"""One persistent pipe writer; old requests cannot send commands to a restarted engine."""
import io
import queue
import threading
import os
import sys
import tempfile
from unittest import mock
import time
import unittest
from serve import server


class Pipe:
    def __init__(self):
        self.text = ""
        self.closed = False
        self.flushes = 0
        self.delay = 0
        self.failure = False

    def write(self, text):
        if self.closed:
            raise ValueError("closed")
        if self.failure:
            self.text += text[:1]
            raise BrokenPipeError("partial write")
        for char in text:
            self.text += char
            if self.delay:
                time.sleep(self.delay)
        return len(text)

    def flush(self):
        self.flushes += 1

    def close(self):
        self.closed = True


class Process:
    def __init__(self):
        self.stdin, self.stdout = Pipe(), io.StringIO()
        self.code = None
        self.waits = []

    def poll(self):
        return self.code

    def wait(self, timeout=None):
        self.waits.append(timeout)
        self.code = 0
        return 0

    def terminate(self):
        self.code = -15

    def kill(self):
        self.code = -9


class SeenLock:
    def __init__(self):
        self.lock = threading.Lock()
        self.waiting = threading.Event()

    def __enter__(self):
        self.waiting.set()
        self.lock.acquire()
        return self

    def __exit__(self, *args):
        self.lock.release()


def engine():
    e = object.__new__(server.StrataEngine)
    e.wlock = threading.Lock()
    e.proc, e.gen = Process(), 1
    e._stdin_closing = False
    e.ended, e.pump, e.log = False, None, None
    e.lines, e.can_stop, e.silence_s = queue.Queue(), True, 0
    e.info = {}
    return e


class EngineWriterTests(unittest.TestCase):
    def test_concurrent_commands_remain_complete_lines(self):
        e = engine()
        e.proc.stdin.delay = .00005
        commands = ["GEN 4 " + ",".join(str(x) for x in range(80)), "STOP", "VRAM 512", "SAVE /tmp/a path"]
        barrier = threading.Barrier(len(commands))
        errors = []
        def send(command):
            try:
                barrier.wait()
                e._send(command)
            except Exception as error:
                errors.append(error)
        threads = [threading.Thread(target=send, args=(c,)) for c in commands]
        for t in threads:
            t.start()
        for t in threads:
            t.join(2)
            self.assertFalse(t.is_alive())
        self.assertFalse(errors)
        self.assertCountEqual(e.proc.stdin.text.splitlines(), commands)
        self.assertEqual(e.proc.stdin.flushes, len(commands))

    def test_queued_old_command_rejected_after_replacement(self):
        e = engine()
        old = e.proc
        e.wlock = SeenLock()
        errors = []
        def send():
            try:
                e._send("STOP")
            except server.EngineDied as error:
                errors.append(error)
        e.wlock.lock.acquire()
        thread = threading.Thread(target=send)
        thread.start()
        self.assertTrue(e.wlock.waiting.wait(1))
        replacement = Process()
        e.proc, e.gen = replacement, 2  # publication while the writer lock is held
        e.wlock.lock.release()
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertEqual(old.stdin.text, "")
        self.assertEqual(replacement.stdin.text, "")
        with self.assertRaises(server.EngineDied):
            e._send("BSTOP 0", born=1)
        self.assertEqual(replacement.stdin.text, "")

    def test_partial_write_never_retried(self):
        e = engine()
        old = e.proc
        old.stdin.failure = True
        with self.assertRaises(server.EngineDied):
            e._send("GEN 1 1", born=1)
        self.assertEqual(old.stdin.text, "G")
        self.assertTrue(e._stdin_closing)
        with self.assertRaises(server.EngineDied):
            e._send("STOP")
        self.assertEqual(old.stdin.text, "G")
        e.close()
        self.assertEqual(old.stdin.text, "G", "shutdown must not append to a partial command")
        e.proc, e.gen = Process(), 2
        self.assertEqual(e.proc.stdin.text, "")

    def test_real_solo_vram_and_session_commands_preserve_bytes(self):
        e = engine()
        e.lines.put("DONE 1 2 3 4 length")
        self.assertEqual(list(e.generate([3, 5], 1, {}, threading.Event())), [])
        e.lines.put("VRAM reserve_mib=512 expert_slots=10")
        self.assertEqual(e.vram(512)["reserve_mib"], 512)
        e.lines.put("SAVED 2 64 1.5")
        self.assertEqual(e.session_file("save", "/tmp/a path"), {"tokens": 2, "bytes": 64, "ms": 1.5})
        self.assertEqual(e.proc.stdin.text, "GEN 1 3,5\nVRAM 512\nSAVE /tmp/a path\n")

    def test_shutdown_frames_and_close_are_one_writer_transaction(self):
        e = engine()
        old, writer = e.proc, e.wlock
        def wait(timeout=None):
            self.assertTrue(writer.acquire(blocking=False), "process wait must not hold writer lock")
            writer.release()
            old.code = 0
            return 0
        old.wait = wait
        e.close()
        self.assertEqual(old.stdin.text, "STOP\nQUIT\n")
        self.assertTrue(old.stdin.closed)
        self.assertIsNone(e.proc)
        self.assertIs(e.wlock, writer)

    def test_lazy_reinitialization_preserves_lock_and_rejects_old_writer(self):
        e = engine()
        writer = e.wlock
        e.__init__("/nonexistent/strata", [], lazy=True)
        self.assertIs(e.wlock, writer)
        with self.assertRaises(server.EngineDied):
            e._send("STOP", born=1)

    def test_admitted_batch_request_cannot_rebind_after_yield_restart(self):
        e = engine()
        old = e.proc
        e.batch = 2
        e.slot_cv, e.ctl = threading.Condition(), threading.Lock()
        e.slot_busy, e.slot_claims, e.waiting = [False, False], set(), 0
        e.slot_held, e.slot_q = [[], []], [queue.Queue(), queue.Queue()]
        e.slot_live, e.slot_used = [None, None], [0, 0]
        e.ctl_epoch, e.admitting = 0, 0
        e.last = {}
        e.pick_slot = lambda prompt, **kwargs: 0
        e._shorter_waiting = lambda plen: True
        acquisitions = []
        def take(*args, **kwargs):
            e.ctl.acquire()
            acquisitions.append(1)
            if len(acquisitions) == 2:
                with e.wlock:
                    e.proc, e.gen = Process(), 2
                    e.slot_held = [[99], [98]]
                    e.slot_live = ["new request", None]
            if False:
                yield None
            return True
        def control(cancel, on_token, **kwargs):
            kwargs["on_quiet"]()
            e._yielded = (0, 1)
            e.last = {"finish": "cancel"}
            yield None
        e._take_control, e._control = take, control
        with self.assertRaisesRegex(server.EngineDied, "after this request was admitted"):
            list(e.generate_batched([1, 2], 4, {}, threading.Event()))
        self.assertEqual(old.stdin.text, "GEN 4 1,2\nBYIELD 0\n")
        self.assertEqual(e.proc.stdin.text, "")
        self.assertEqual(e.slot_held, [[99], [98]])
        self.assertEqual(e.slot_live, ["new request", None])
        self.assertFalse(e.ctl.locked())

    def test_actual_restart_preserves_writer_lock_and_fences_old_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = os.path.join(directory, "fake-engine")
            with open(executable, "w", encoding="utf-8") as out:
                out.write(f"#!{sys.executable}\nimport sys\nprint('READY 4096 stop', flush=True)\n"
                          "for line in sys.stdin:\n    if line.strip() == 'QUIT': break\n")
            os.chmod(executable, 0o700)
            with mock.patch.object(server, "end_leftover_engines"):
                e = server.StrataEngine(executable, [])
                old, generation, writer = e.proc, e.gen, e.wlock
                try:
                    e.restart(tries=1)
                    self.assertIs(e.wlock, writer)
                    self.assertIsNot(e.proc, old)
                    self.assertGreater(e.gen, generation)
                    with self.assertRaises(server.EngineDied):
                        e._send("STOP", born=generation)
                    self.assertIsNone(e.proc.poll())
                finally:
                    e.close()

    def test_old_generator_cancellation_does_not_stop_replacement(self):
        e = engine()
        old = e.proc
        e.lines.put("T 7")
        gen = e.generate([1], 8, {}, threading.Event())
        self.assertEqual(next(gen), 7)
        with e.wlock:
            e.proc, e.gen, e.lines = Process(), 2, queue.Queue()
        gen.close()
        self.assertEqual(old.stdin.text, "GEN 8 1\n")
        self.assertEqual(e.proc.stdin.text, "")


if __name__ == "__main__":
    unittest.main()
