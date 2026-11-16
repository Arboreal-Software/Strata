"""Restart barriers bind readers, cleanup, and destructive actions to one child."""
import queue
import os
import threading
import unittest
from unittest import mock
from serve import server
from serve.test_engine_writer import engine, Process


def batch_engine():
    e = engine()
    e.batch = 2
    e.slot_cv, e.ctl = threading.Condition(), threading.Lock()
    e.slot_busy, e.slot_claims, e.waiting = [False, True], set(), 0
    e.slot_held, e.slot_q = [[], []], [queue.Queue(), queue.Queue()]
    e.slot_live, e.slot_used = [None, None], [0, 0]
    e.slot_releasing, e._orphan_seen = set(), {}
    e.slot_order, e.slot_group, e.slot_groups = [0, 1], [0, 0], 1
    e.slot_gs, e.ctl_epoch, e.admitting = 0, 0, 0
    e.wait_lens, e._yielded, e._ctl_erred = [], None, False
    e.arrivals = server.collections.deque(maxlen=64)  # actual engine admission-window default
    def take(*args, **kwargs):
        e.ctl.acquire()
        if False:
            yield None
        return True
    e._take_control = take
    return e


def replace(e):
    e.proc, e.gen, e.lines = Process(), e.gen + 1, queue.Queue()
    e.slot_q = [queue.Queue(), queue.Queue()]
    e.slot_busy, e.slot_claims = [True, False], {0}
    e.slot_held, e.slot_live = [[99], []], ["successor", None]
    e.slot_used, e.slot_releasing, e._orphan_seen = [0, 0], set(), {}
    e.ended = False


class EngineEpochTests(unittest.TestCase):
    def test_captured_close_mismatch_cannot_touch_successor(self):
        e = batch_engine()
        old = e.proc
        replace(e)
        successor = e.proc
        e._state_lock()
        before = dict(e.__dict__)
        with mock.patch.object(successor, "poll", side_effect=AssertionError("successor polled")), \
             mock.patch.object(successor.stdin, "close", side_effect=AssertionError("successor stdin closed")), \
             mock.patch.object(successor.stdout, "close", side_effect=AssertionError("successor stdout closed")):
            with self.assertRaises(server.EngineStuck):
                e.close(_expected_proc=old)
        self.assertIs(e.proc, successor)
        self.assertEqual(e.__dict__, before)
        # Even the no-child early return must reject a different captured target.
        e.proc = None
        with self.assertRaises(server.EngineStuck):
            e.close(_expected_proc=old)
        self.assertIsNone(e.proc)

    def test_never_started_ordinary_reap_does_not_invent_channels(self):
        e = server.StrataEngine("missing-executable", [], lazy=True)
        epoch = e._capture_epoch()
        self.assertIsNone(epoch.proc)
        self.assertIsNone(epoch.slots.slot_q)
        e._reap_owned_image_transports(epoch)
        self.assertIsNone(epoch.slots.slot_q)
        self.assertIsNone(e.proc)

    def test_missing_strict_or_live_channels_remain_fenced(self):
        import dataclasses
        e = server.StrataEngine("missing-executable", [], lazy=True)
        ordinary = e._capture_epoch()
        for epoch in (dataclasses.replace(ordinary, strict_owned=True),
                      dataclasses.replace(ordinary, proc=Process()),
                      dataclasses.replace(ordinary, receipt_env=(("STRATA_HTTP_TWO_SLOT_RECEIPTS", "1"),))):
            with self.subTest(strict=epoch.strict_owned, live=epoch.proc is not None):
                with self.assertRaises(server.EngineStuck):
                    e._reap_owned_image_transports(epoch)
                self.assertIsNone(epoch.slots.slot_q)

    def test_populated_uncertain_transport_retains_fd_until_original_exit(self):
        e = batch_engine()
        epoch = e._capture_epoch()
        read_fd, write_fd = os.pipe()
        os.close(write_fd)
        channel = epoch.slots.slot_q[0]
        channel._owned_image_proof_lost = True
        channel._owned_image_transport_fd = read_fd
        try:
            with self.assertRaises(server.EngineStuck):
                e._reap_owned_image_transports(epoch)
            os.fstat(read_fd)
            epoch.proc.code = 0
            e._reap_owned_image_transports(epoch)
            self.assertIsNone(channel._owned_image_transport_fd)
            self.assertFalse(channel._owned_image_proof_lost)
            with self.assertRaises(OSError):
                os.fstat(read_fd)
        finally:
            if getattr(channel, "_owned_image_transport_fd", None) is not None:
                os.close(read_fd)

    def test_close_exited_between_polls_closes_captured_pipe_before_wait(self):
        import itertools
        e = engine()
        old = e.proc
        old.poll = mock.Mock(side_effect=itertools.chain([None], itertools.repeat(0)))
        def wait(timeout=None):
            self.assertTrue(old.stdin.closed)
            self.assertIs(e.proc, old)
            return 0
        old.wait = mock.Mock(side_effect=wait)
        e.close()
        self.assertIsNone(e.proc)
        self.assertTrue(old.stdin.closed)
        old.wait.assert_called_once()

    def test_close_initially_exited_closes_stdin_before_eof_reader_join(self):
        e = engine()
        proc = e.proc
        proc.code = 0
        input_read, input_write = os.pipe()
        output_read, output_write = os.pipe()
        proc.stdin.close()
        proc.stdout.close()
        proc.stdin = os.fdopen(input_write, "wb")
        proc.stdout = os.fdopen(output_read, "rb")
        entered = threading.Event()
        def child_input():
            try:
                while os.read(input_read, 4096):
                    pass
            finally:
                os.close(input_read)
                os.close(output_write)
        def read():
            entered.set()
            proc.stdout.readline()
        consumer = threading.Thread(target=child_input, daemon=True)
        e.pump = threading.Thread(target=read, daemon=True)
        consumer.start()
        e.pump.start()
        self.assertTrue(entered.wait(1))
        try:
            with mock.patch.object(server, "ENGINE_QUIT_WAIT_S", .02), \
                 mock.patch.object(server, "ENGINE_TERM_WAIT_S", .1), \
                 mock.patch.object(server, "ENGINE_KILL_WAIT_S", .1):
                e.close()
            self.assertIsNone(e.proc)
            self.assertTrue(proc.stdin.closed)
            self.assertFalse(e.pump.is_alive())
            self.assertIsNone(e._close_owner)
        finally:
            proc.stdin.close()
            consumer.join(1)
            e.pump.join(1)
            if not e.pump.is_alive():
                proc.stdout.close()
        self.assertFalse(consumer.is_alive())
        self.assertFalse(e.pump.is_alive())

    def test_close_failed_writer_waits_for_exit_before_buffered_eof(self):
        e = engine()
        proc = e.proc
        e._stdin_failed = True
        input_read, input_write = os.pipe()
        output_read, output_write = os.pipe()
        proc.stdin.close()
        proc.stdout.close()
        proc.stdin = os.fdopen(input_write, "wb")
        proc.stdout = os.fdopen(output_read, "rb")
        proc.stdin.write(b"partial")  # Real bytes still buffered until close.
        exited, entered = threading.Event(), threading.Event()
        received = bytearray()
        original_close, original_wait = proc.stdin.close, proc.wait
        original_terminate = proc.terminate
        def terminate():
            original_terminate()
            exited.set()
        def wait(timeout=None):
            if not exited.wait(1):
                raise AssertionError("captured child was not terminated")
            return original_wait(timeout=timeout)
        def checked_close():
            self.assertTrue(exited.is_set())
            self.assertIsNotNone(proc.poll())
            self.assertTrue(e.wlock.locked())
            original_close()
        def child_input():
            try:
                while True:
                    chunk = os.read(input_read, 4096)
                    if not chunk:
                        break
                    received.extend(chunk)
            finally:
                os.close(input_read)
                os.close(output_write)
        def read():
            entered.set()
            proc.stdout.readline()
        consumer = threading.Thread(target=child_input, daemon=True)
        e.pump = threading.Thread(target=read, daemon=True)
        consumer.start()
        e.pump.start()
        self.assertTrue(entered.wait(1))
        try:
            with mock.patch.object(proc, "terminate", side_effect=terminate), \
                 mock.patch.object(proc, "wait", side_effect=wait), \
                 mock.patch.object(proc.stdin, "close", side_effect=checked_close), \
                 mock.patch.object(server, "ENGINE_QUIT_WAIT_S", .02), \
                 mock.patch.object(server, "ENGINE_TERM_WAIT_S", .1), \
                 mock.patch.object(server, "ENGINE_KILL_WAIT_S", .1):
                e.close()
            consumer.join(1)
            self.assertIsNone(e.proc)
            self.assertEqual(received, b"partial")
            self.assertNotIn(b"STOP", received)
            self.assertNotIn(b"QUIT", received)
            self.assertIsNone(e._close_owner)
        finally:
            original_close()
            consumer.join(1)
            e.pump.join(1)
            if not e.pump.is_alive():
                proc.stdout.close()
        self.assertFalse(consumer.is_alive())
        self.assertFalse(e.pump.is_alive())

    def test_close_blocked_buffer_retry_retains_same_worker_and_pipe(self):
        e = engine()
        proc = e.proc
        proc.code = 0
        proc.stdin.close()
        read_fd, write_fd = os.pipe()
        os.set_blocking(write_fd, False)
        try:
            while True:
                os.write(write_fd, b"x" * 4096)
        except BlockingIOError:
            pass
        try:
            while True:
                os.write(write_fd, b"x")
        except BlockingIOError:
            pass
        os.set_blocking(write_fd, True)
        proc.stdin = os.fdopen(write_fd, "wb", buffering=4096)
        proc.stdin.write(b"tail")  # Actual close flush blocks on the full pipe.
        e.pump = None
        real_thread = threading.Thread
        workers, callers, outcomes = [], [], []
        entered = threading.Event()
        original_close = proc.stdin.close
        def observed_close():
            entered.set()
            original_close()
        def thread(*args, **kwargs):
            worker = real_thread(*args, **kwargs)
            workers.append(worker)
            return worker
        def attempt(returned):
            try:
                e.close()
            except BaseException as error:
                outcomes.append(error)
            finally:
                returned.set()
        drained = bytearray()
        def drain():
            try:
                while True:
                    chunk = os.read(read_fd, 4096)
                    if not chunk:
                        break
                    drained.extend(chunk)
            finally:
                os.close(read_fd)
        drainer = None
        try:
            with mock.patch.object(server.threading, "Thread", side_effect=thread), \
                 mock.patch.object(proc.stdin, "close", side_effect=observed_close), \
                 mock.patch.object(server, "ENGINE_QUIT_WAIT_S", .02), \
                 mock.patch.object(server, "ENGINE_TERM_WAIT_S", .02), \
                 mock.patch.object(server, "ENGINE_KILL_WAIT_S", .02):
                returned = threading.Event()
                caller = real_thread(target=attempt, args=(returned,), daemon=True)
                callers.append(caller)
                caller.start()
                self.assertTrue(entered.wait(1))
                self.assertTrue(returned.wait(.5), "caller blocked in buffered stdin.close")
                caller.join(1)
                self.assertEqual(len(outcomes), 1)
                self.assertIsInstance(outcomes[0], server.EngineStuck)
                owner = e._close_owner
                self.assertIs(owner[0], proc)
                self.assertIs(owner[1], proc.stdin)
                self.assertEqual(len(workers), 1)
                self.assertIs(owner[4], workers[0])
                self.assertTrue(workers[0].is_alive())
                os.fstat(write_fd)
                returned = threading.Event()
                caller = real_thread(target=attempt, args=(returned,), daemon=True)
                callers.append(caller)
                caller.start()
                self.assertTrue(returned.wait(.5))
                caller.join(1)
                self.assertEqual(len(outcomes), 2)
                self.assertIsInstance(outcomes[1], server.EngineStuck)
                self.assertIs(e._close_owner, owner)
                self.assertEqual(len(workers), 1)
                self.assertIs(e.proc, proc)
                self.assertFalse(proc.stdin.closed)
                os.fstat(write_fd)
            drainer = real_thread(target=drain, daemon=True)
            drainer.start()
            workers[0].join(1)
            self.assertFalse(workers[0].is_alive())
            e.close()
            drainer.join(1)
            self.assertIsNone(e.proc)
            self.assertIsNone(e._close_owner)
            self.assertTrue(drained.endswith(b"tail"))
            with self.assertRaises(OSError):
                os.fstat(write_fd)
        finally:
            if drainer is None:
                drainer = real_thread(target=drain, daemon=True)
                drainer.start()
            for caller in callers:
                caller.join(1)
            for worker in workers:
                worker.join(1)
            if not proc.stdin.closed and not any(worker.is_alive() for worker in workers):
                original_close()
            drainer.join(1)
        self.assertFalse(drainer.is_alive())
        self.assertTrue(all(not worker.is_alive() for worker in workers))
        self.assertTrue(all(not caller.is_alive() for caller in callers))

    def test_strict_admin_single_frame_acquire_borrow_busy_and_errors(self):
        import dataclasses
        from serve.test_batch_control_guard import CountLock
        for operation in ("vram", "save", "restore"):
            for outcome in ("acquire", "borrow", "busy", "native_error"):
                with self.subTest(operation=operation, outcome=outcome):
                    e = batch_engine()
                    e.ctl = CountLock()
                    e.CONTROL_WAIT_S = 0
                    epoch = dataclasses.replace(e._capture_epoch(), strict_owned=True)
                    e._epoch = epoch
                    if outcome in ("borrow", "busy"):
                        self.assertTrue(e.ctl.acquire())
                    if outcome != "busy":
                        epoch.lines.put(("ERR native busy" if operation == "vram" else "SERR memory 0 native busy")
                                        if outcome == "native_error" else
                                        ("VRAM expert_slots=1 vram_free_mib=4" if operation == "vram" else
                                         ("SAVED 7 70 1" if operation == "save" else "RESTORED 7 70 1")))
                    try:
                        with mock.patch.object(e, "_exclusive_control", side_effect=AssertionError("strict context")), \
                             mock.patch.object(server._RequestOwned, "_store", side_effect=AssertionError("strict TLS")):
                            if outcome in ("busy", "native_error"):
                                with self.assertRaisesRegex(ValueError, "request is still running|native busy"):
                                    if operation == "vram":e.vram(None)
                                    else:e.session_file(operation, "/tmp/recording-only")
                            elif operation == "vram":
                                self.assertEqual(e.vram(None, _control=e.ctl if outcome == "borrow" else None)["vram_free_mib"], 4)
                            else:
                                self.assertEqual(e.session_file(operation, "/tmp/recording-only",
                                                                _control=e.ctl if outcome == "borrow" else None)["tokens"], 7)
                        if outcome in ("busy", "borrow"):
                            self.assertTrue(e.ctl.raw.locked())
                            self.assertEqual(e.ctl.releases, 0)
                        else:
                            self.assertFalse(e.ctl.raw.locked())
                            self.assertEqual(e.ctl.releases, 1)
                        if outcome == "busy":self.assertEqual(e.proc.stdin.text, "")
                    finally:
                        if e.ctl.raw.locked():e.ctl.release()

    def test_strict_admin_replacement_releases_only_captured_acquired_owner(self):
        import dataclasses
        from serve.test_batch_control_guard import CountLock
        for operation in ("vram", "save", "restore"):
            with self.subTest(operation=operation):
                e = batch_engine()
                captured = CountLock()
                successor = threading.Lock()
                successor.acquire()
                e.ctl = captured
                epoch = dataclasses.replace(e._capture_epoch(), strict_owned=True)
                e._epoch = epoch
                def change():
                    replace(e)
                    e.ctl = successor
                captured.after_acquire = change
                try:
                    with self.assertRaises(server.EngineDied):
                        if operation == "vram":e.vram(None)
                        else:e.session_file(operation, "/tmp/recording-only")
                    self.assertFalse(captured.raw.locked())
                    self.assertEqual(captured.releases, 1)
                    self.assertTrue(successor.locked())
                    self.assertEqual(epoch.proc.stdin.text, "")
                    self.assertEqual(e.proc.stdin.text, "")
                finally:
                    successor.release()

    def test_strict_admin_refuses_different_borrowed_owner_before_io(self):
        import dataclasses
        for operation in ("vram", "save", "restore"):
            with self.subTest(operation=operation):
                e = batch_engine()
                e._epoch = dataclasses.replace(e._capture_epoch(), strict_owned=True)
                borrowed = threading.Lock()
                borrowed.acquire()
                try:
                    with self.assertRaisesRegex(server.EngineDied, "borrowed control"):
                        if operation == "vram":e.vram(None, _control=borrowed)
                        else:e.session_file(operation, "/tmp/recording-only", _control=borrowed)
                    self.assertTrue(borrowed.locked())
                    self.assertFalse(e.ctl.locked())
                    self.assertEqual(e.proc.stdin.text, "")
                finally:
                    borrowed.release()

    def test_delayed_logprob_read_fences_callback_after_replacement(self):
        e = batch_engine()
        epoch = e._capture_epoch()
        old = e.lines
        old.put("T 7")
        old.put("LP -0.25")
        e._ctl_mode = "solo"
        read = old.get
        def replace_after_lp(*args, **kwargs):
            line = read(*args, **kwargs)
            if line.startswith("LP "):
                replace(e)
                e.lines.put("BIPAY successor request=still-owned")
            return line
        callbacks = []
        with mock.patch.object(old, "get", side_effect=replace_after_lp):
            with self.assertRaises(server.EngineDied):
                next(e._control(threading.Event(), callbacks.append, want_lp=True, epoch=epoch))
        self.assertEqual(callbacks, [], "a blocked LP read cannot call back after replacement")
        self.assertEqual(e.lines.get_nowait(), "BIPAY successor request=still-owned")

    def test_borrowed_control_clear_survives_batch_mode_transition(self):
        e = batch_engine()
        ctl = e.ctl
        ctl.acquire()
        try:
            e._ctl_held = ctl  # only after this actual non-reentrant lock was acquired
            # READY can select ordinary mode on restart. The borrower's real
            # finally must clear the same thread's token in that mode too.
            e.batch = 0
            e._ctl_held = False
            e.batch = 2
            e.CONTROL_WAIT_S = 0
            entered = []
            with self.assertRaisesRegex(ValueError, "request is still running"):
                with e._exclusive_control():
                    entered.append(True)
            self.assertEqual(entered, [], "an old TLS owner cannot reappear as permission")
            self.assertTrue(ctl.locked())
        finally:
            e._ctl_held = False
            ctl.release()

    def test_borrowed_control_is_not_authority_for_another_thread(self):
        e = batch_engine()
        e.CONTROL_WAIT_S = 0
        ctl = e.ctl
        ctl.acquire()
        e._ctl_held = ctl
        result = []
        def other_request():
            try:
                with e._exclusive_control():
                    result.append("entered")
            except ValueError:
                result.append("busy")
        worker = threading.Thread(target=other_request)
        try:
            worker.start()
            worker.join(2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(result, ["busy"])
            self.assertTrue(ctl.locked())
        finally:
            e._ctl_held = False
            ctl.release()
            worker.join(2)

    def test_unpublished_epoch_releases_actual_acquired_control_once(self):
        e = batch_engine()
        acquired = mock.Mock(wraps=threading.Lock())
        successor = threading.Lock()
        successor.acquire()
        e.ctl = acquired
        # Capture the original epoch, then install a child that has not reached
        # READY. Exercise the real _capture_epoch refusal after acquisition.
        e._capture_epoch()
        replace(e)
        e._stdin_closing = True
        def take(*args, **kwargs):
            acquired.acquire()
            e.ctl = successor
            if False:
                yield None
            return acquired
        e._take_control = take
        gen = e.generate_batched([1, 2], 3, {}, threading.Event())
        try:
            with self.assertRaisesRegex(server.EngineDied, "ready process epoch"):
                next(gen)
            self.assertFalse(acquired.locked())
            self.assertTrue(successor.locked())
            self.assertEqual(e.proc.stdin.text, "")
            self.assertEqual(e.slot_claims, {0})
            gen.close()
            acquired.release.assert_called_once_with()
        finally:
            gen.close()
            successor.release()

    def test_control_setup_trace_failure_releases_control_once(self):
        # Both allocation errors and generator termination must leave no lock
        # behind before the normal request-finally locals exist.
        for error in (MemoryError("trace allocation"), GeneratorExit()):
            with self.subTest(error=type(error).__name__):
                e = batch_engine()
                acquired = mock.Mock(wraps=threading.Lock())
                e.ctl = acquired
                gen = e.generate_batched([1, 2], 3, {}, threading.Event())
                with mock.patch.object(server, "btrace", side_effect=error):
                    with self.assertRaises(type(error)):
                        next(gen)
                gen.close()
                self.assertFalse(acquired.locked())
                acquired.release.assert_called_once_with()
                self.assertEqual(e.proc.stdin.text, "")
                self.assertEqual(e.slot_claims, set())

    def test_paused_batch_reader_never_consumes_successor(self):
        e = batch_engine()
        e.lines.put("DONE 0 2 1 1 length")
        e.lines.put("BADM 0 1")
        old_q = e.slot_q[0]
        send = e._send
        def admit(text, **kwargs):
            send(text, **kwargs)
            if text.startswith("BGEN "):
                old_q.put("BT 0 7")
        e._send = admit
        gen = e.generate_batched([1, 2], 3, {}, threading.Event())
        self.assertEqual(next(gen), 7)
        replace(e)
        old_q.put(None)
        e.slot_q[0].put("BT 0 99")
        try:
            with self.assertRaises(server.EngineDied):
                next(gen)
            self.assertEqual(e.slot_q[0].get_nowait(), "BT 0 99")
            self.assertEqual(e.slot_live[0], "successor")
            self.assertTrue(e.slot_busy[0])
        finally:
            gen.close()

    def test_restart_after_bstop_never_rebinds_waiter(self):
        e = batch_engine()
        old_q, old_busy = e.slot_q[0], e.slot_busy
        old_busy[0] = True
        old_q.put("BDONE 0 1 cancel 1")
        def send(*args, **kwargs):
            replace(e)
            e.slot_q[0].put("BDONE 0 1 stop 1")
        e._send = send
        threads = []
        real_thread = threading.Thread
        def thread(*args, **kwargs):
            t = real_thread(*args, **kwargs)
            threads.append(t)
            return t
        with mock.patch.object(server.threading, "Thread", side_effect=thread):
            e._release_slot_when_done(0, [1, 2], born=1)
        for t in threads:
            t.join(1)
            self.assertFalse(t.is_alive())
        self.assertTrue(e.slot_busy[0])
        self.assertEqual(e.slot_held[0], [99])
        self.assertEqual(e.slot_q[0].get_nowait(), "BDONE 0 1 stop 1")

    def test_old_cleanup_table_mutation_cannot_clear_successor(self):
        e = batch_engine()
        e.lines.put("DONE 0 2 1 1 length")
        e.lines.put("BADM 0 1")
        old_q = e.slot_q[0]
        send = e._send
        def admit(text, **kwargs):
            send(text, **kwargs)
            if text.startswith("BGEN "):
                old_q.put("BT 0 7")
        e._send = admit
        class RestartOnClear(list):
            def __setitem__(self, key, value):
                if value is None:
                    replace(e)
                super().__setitem__(key, value)
        e.slot_live = RestartOnClear(e.slot_live)
        gen = e.generate_batched([1, 2], 3, {}, threading.Event())
        self.assertEqual(next(gen), 7)
        gen.close()
        self.assertEqual(e.slot_live[0], "successor")
        self.assertTrue(e.slot_busy[0])
        self.assertEqual(e.slot_claims, {0})

    def test_delayed_pp_does_not_publish_progress_or_invoke_callback(self):
        e = batch_engine()
        e._ctl_mode = "solo"
        class PausedReply:
            def get(self, **kwargs):
                replace(e)
                e.progress = (99, 100)
                return "PP 1 2 3 4"
        e.lines = PausedReply()
        called = []
        gen = e._control(threading.Event(), lambda t: called.append(t), on_quiet=lambda: called.append("quiet"))
        with self.assertRaises(server.EngineDied):
            next(gen)
        self.assertEqual(called, [])
        self.assertEqual(e.progress, (99, 100))

    def test_delayed_give_way_cannot_claim_successor_slot(self):
        e = batch_engine()
        e.slot_busy = [False, False]
        e._shorter_waiting = lambda plen: True
        def control(cancel, on_token, **kwargs):
            replace(e)
            kwargs["on_quiet"]()
            if False:
                yield None
        e._control = control
        with self.assertRaises(server.EngineDied):
            list(e.generate_batched([1, 2], 4, {}, threading.Event()))
        self.assertEqual(e.slot_claims, {0})
        self.assertEqual(e.slot_busy, [True, False])
        self.assertEqual(e.proc.stdin.text, "")

    def test_old_stop_timeout_cannot_kill_successor(self):
        e = engine()
        old = e.proc
        e.silence_s = .01
        class PausedTimeout:
            n = 0
            def get(self, **kwargs):
                self.n += 1
                if self.n == 1:
                    return "T 7"
                replace(e)
                raise queue.Empty
        e.lines = PausedTimeout()
        gen = e.generate([1], 8, {}, threading.Event())
        self.assertEqual(next(gen), 7)
        with self.assertRaises(server.EngineSilent):
            gen.close()
        self.assertIsNone(e.proc.poll())
        self.assertFalse(e.ended)
        self.assertEqual(old.stdin.text, "GEN 8 1\nSTOP\n")

    def test_session_wait_never_reads_or_kills_successor(self):
        e = engine()
        class PausedReply:
            def get(self, **kwargs):
                replace(e)
                e.lines.put("SAVED 9 90 1")
                return "ERR stale"
        e.lines = PausedReply()
        with self.assertRaises(server.EngineDied):
            e.session_file("save", "/tmp/old")
        self.assertIsNone(e.proc.poll())
        self.assertFalse(e.ended)
        self.assertEqual(e.lines.get_nowait(), "SAVED 9 90 1")

    def test_blocked_writer_close_terminates_only_captured_child(self):
        e = engine()
        old = e.proc
        entered, released = threading.Event(), threading.Event()
        def write(text):
            entered.set()
            released.wait(2)
            raise BrokenPipeError("child terminated")
        old.stdin.write = write
        def terminate():
            old.code = -15
            released.set()
        old.terminate = terminate
        errors = []
        def send():
            try:
                e._send("GEN 8 1")
            except server.EngineDied as error:
                errors.append(error)
        t = threading.Thread(target=send)
        t.start()
        self.assertTrue(entered.wait(1))
        try:
            with mock.patch.object(server, "ENGINE_QUIT_WAIT_S", .02), mock.patch.object(server, "ENGINE_TERM_WAIT_S", .1):
                e.close()
            self.assertIsNone(e.proc)
            self.assertEqual(old.poll(), 0)
        finally:
            released.set()
            t.join(1)
        self.assertFalse(t.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertNotIn("STOP", old.stdin.text)

    def test_cleanup_releases_captured_control_lock_only(self):
        e = batch_engine()
        e.ctl = mock.Mock(wraps=threading.Lock())
        e.slot_busy = [False, False]
        e._shorter_waiting = lambda plen: False
        e.lines.put("PP 1 2 3 4")
        gen = e.generate_batched([1, 2], 4, {}, threading.Event())
        self.assertIsNone(next(gen))
        old_ctl = e.ctl
        self.assertTrue(old_ctl.locked())
        replace(e)
        e.ctl = threading.Lock()
        e.ctl.acquire()
        gen.close()
        self.assertFalse(old_ctl.locked())
        gen.close()
        old_ctl.release.assert_called_once_with()
        self.assertTrue(e.ctl.locked())
        e.ctl.release()

    def test_never_admitted_waiter_can_take_successor(self):
        e = batch_engine()
        old = e.proc
        def take(*args, **kwargs):
            replace(e)
            e.slot_busy = [False, True]
            e.slot_claims.clear()
            e.ctl.acquire()
            if False:
                yield None
            return e.ctl
        e._take_control = take
        send = e._send
        def admit(text, **kwargs):
            send(text, **kwargs)
            if text.startswith("BGEN "):
                e.lines.put("DONE 0 2 1 1 length")
                e.lines.put("BADM 0 0")
        e._send = admit
        self.assertEqual(list(e.generate_batched([1, 2], 2, {}, threading.Event())), [])
        self.assertEqual(old.stdin.text, "")
        self.assertEqual(e.proc.stdin.text, "BGEN 0 2 1,2\n")
        self.assertFalse(e.ctl.locked())

    def test_blocked_output_reader_fences_retirement_and_allows_retry(self):
        e = engine()
        proc = e.proc
        proc.code = 0
        read_fd, retained_writer = os.pipe()
        proc.stdout.close()
        proc.stdout = os.fdopen(read_fd, "r")
        entered, returned = threading.Event(), threading.Event()
        errors = []
        def read():
            entered.set()
            proc.stdout.readline()
        def close():
            try:
                e.close()
            except Exception as error:
                errors.append(error)
            finally:
                returned.set()
        e.pump = threading.Thread(target=read, daemon=True)
        e.pump.start()
        self.assertTrue(entered.wait(1))
        closer = threading.Thread(target=close, daemon=True)
        with mock.patch.object(server, "ENGINE_QUIT_WAIT_S", .02), \
             mock.patch.object(server, "ENGINE_TERM_WAIT_S", .02), \
             mock.patch.object(server, "ENGINE_KILL_WAIT_S", .02):
            try:
                closer.start()
                # Background close keeps the regression bounded even on the
                # former implementation, whose stdout.close blocks here.
                self.assertTrue(returned.wait(.5), "shutdown blocked on buffered output reader")
                self.assertEqual(len(errors), 1)
                self.assertIsInstance(errors[0], server.EngineStuck)
                self.assertIn("output reader", str(errors[0]))
                self.assertIs(e.proc, proc)
                self.assertFalse(proc.stdout.closed)
                self.assertFalse(e.wlock.locked())
                self.assertFalse(e._close_lock.locked())
            finally:
                os.close(retained_writer)
                e.pump.join(timeout=1)
                closer.join(timeout=1)
        self.assertFalse(e.pump.is_alive())
        self.assertFalse(closer.is_alive())
        e.close()
        self.assertIsNone(e.proc)
        self.assertTrue(proc.stdout.closed)
        self.assertTrue(proc.stdin.closed)

    def test_unreleased_writer_fences_restart(self):
        e = engine()
        e.wlock.acquire()
        try:
            with mock.patch.object(server, "ENGINE_QUIT_WAIT_S", .01), mock.patch.object(server, "ENGINE_TERM_WAIT_S", .01):
                with self.assertRaises(server.EngineStuck):
                    e.close()
            self.assertIsNotNone(e.proc)
            self.assertTrue(e._stdin_closing)
            self.assertEqual(e.proc.stdin.text, "")
        finally:
            e.wlock.release()

    def test_queued_close_never_retargets_successor(self):
        e = engine()
        old = e.proc
        reached, release = threading.Event(), threading.Event()
        class CloseBarrier:
            def acquire(self, **kwargs):
                reached.set()
                return release.wait(1)
            def release(self):
                pass
        e._close_lock = CloseBarrier()
        errors = []
        def close():
            try:
                e.close()
            except Exception as error:
                errors.append(error)
        t = threading.Thread(target=close)
        t.start()
        self.assertTrue(reached.wait(1))
        replace(e)
        successor = e.proc
        release.set()
        t.join(1)
        self.assertFalse(t.is_alive())
        self.assertEqual(errors, [])
        self.assertIs(e.proc, successor)
        self.assertIsNone(successor.poll())
        self.assertEqual(successor.stdin.text, "")
        self.assertIsNotNone(old.poll())

    def test_batch_prompt_and_admission_tokens_keep_order(self):
        e = batch_engine()
        send = e._send
        def admit(text, **kwargs):
            send(text, **kwargs)
            if text.startswith("BGEN "):
                for line in ["PP 1 2 3 4", "T 5", "DONE 1 2 3 4 length", "BADM 0 1"]:
                    e.lines.put(line)
                e.slot_q[0].put("BT 0 7")
                e.slot_q[0].put("BDONE 0 2 length 3")
        e._send = admit
        self.assertEqual(list(e.generate_batched([1, 2], 2, {}, threading.Event())), [None, 5, 7])
        self.assertEqual(e.proc.stdin.text, "BGEN 0 2 1,2\n")
