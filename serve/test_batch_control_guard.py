import threading
import unittest
from unittest import mock
from serve import server
from serve.test_engine_epoch import batch_engine


class CountLock:
    def __init__(self):
        self.raw, self.releases = threading.Lock(), 0
        self.after_acquire = lambda: None

    def acquire(self, *args, **kwargs):
        ok = self.raw.acquire(*args, **kwargs)
        if ok:
            self.after_acquire()
        return ok

    def release(self):
        self.raw.release()
        self.releases += 1


def engine():
    e = batch_engine()
    del e._take_control  # Use the real admission method.
    e.ctl = CountLock()
    e._shorter_waiting = lambda plen: False
    return e


class BatchControlTests(unittest.TestCase):
    def test_failed_capture_and_next_admission(self):
        e = engine()
        acquired = e.ctl
        successor = threading.Lock()
        successor.acquire()
        def unpublished():
            e.ctl = successor
            e._stdin_closing = True
        acquired.after_acquire = unpublished
        try:
            with self.assertRaisesRegex(server.EngineDied, "ready process epoch"):
                next(e.generate_batched([1, 2], 4, {}, threading.Event()))
            self.assertFalse(acquired.raw.locked())
            self.assertEqual(acquired.releases, 1)
            self.assertTrue(successor.locked())
            e.ctl, e._stdin_closing = acquired, False
            acquired.after_acquire = lambda: None
            with self.assertRaises(StopIteration) as done:
                next(e._take_control(threading.Event(), 2))
            self.assertIs(done.exception.value, acquired)
            acquired.release()
        finally:
            successor.release()

    def test_acquired_trace_failure(self):
        e = engine()
        error = RuntimeError("acquired trace")
        with mock.patch.object(server, "btrace", side_effect=error):
            with self.assertRaises(RuntimeError) as caught:
                next(e.generate_batched([1], 4, {}, threading.Event()))
        self.assertIs(caught.exception, error)
        self.assertEqual(e.ctl.releases, 1)

    def test_cleanup_failure_keeps_original_exception(self):
        e = engine()
        original, cleanup = ValueError("body"), RuntimeError("cleanup trace")
        def trace(*args):
            if args[0] == "finally phase":
                raise cleanup
        with mock.patch.object(e, "_check_epoch", side_effect=original), \
             mock.patch.object(server, "btrace", side_effect=trace):
            with self.assertRaises(ValueError) as caught:
                next(e.generate_batched([1], 4, {}, threading.Event()))
        self.assertIs(caught.exception, original)
        self.assertIs(caught.exception.__cause__, cleanup)
        self.assertEqual(e.ctl.releases, 1)

    def test_cleanup_trace_cannot_skip_protocol_drain(self):
        e = engine()
        e.slot_busy = [False, False]
        e.lines.put("PP 1 2 3 4")
        e._drain_control = mock.Mock()
        gen = e.generate_batched([1, 2], 4, {}, threading.Event())
        next(gen)
        def trace(*args):
            if args[0] == "finally phase":
                raise RuntimeError("trace")
        with mock.patch.object(server, "btrace", side_effect=trace):
            gen.close()
        e._drain_control.assert_called_once_with("DONE", born=1)
        self.assertEqual(e.ctl.releases, 1)

    def test_close_throw_cancel_in_each_phase(self):
        for phase in ("solo", "admit", "slot"):
            for ending in ("close", "throw", "cancel"):
                with self.subTest(phase=phase, ending=ending):
                    e, cancel = engine(), threading.Event()
                    e._drain_control = mock.Mock(return_value="BADM 0 0")
                    e._release_slot_when_done = mock.Mock()
                    e.slot_busy = [False, phase != "solo"]
                    send = e._send
                    def start(text, **kwargs):
                        send(text, **kwargs)
                        if not text.startswith(("GEN ", "BGEN ")):
                            return
                        if phase == "slot":
                            e.lines.put("DONE 0 2 1 1 length")
                            e.lines.put("BADM 0 1")
                            e.slot_q[0].put("BT 0 7")
                        else:
                            e.lines.put("PP 1 2 3 4")
                    e._send = start
                    gen = e.generate_batched([1, 2], 4, {}, cancel)
                    next(gen)
                    if ending == "close":
                        gen.close()
                    elif ending == "throw":
                        error = RuntimeError("consumer")
                        with self.assertRaises(RuntimeError) as caught:
                            gen.throw(error)
                        self.assertIs(caught.exception, error)
                    else:
                        cancel.set()
                        e.lines.put("DONE 0 2 1 1 cancel")
                        e.lines.put("BADM 0 0")
                        e.slot_q[0].put("BDONE 0 1 cancel 1")
                        list(gen)
                    self.assertFalse(e.ctl.raw.locked())
                    self.assertEqual(e.ctl.releases, 1)

    def test_failed_drain_keeps_consumer_error_and_releases_old_lock(self):
        e = engine()
        e.slot_busy = [False, False]
        e.lines.put("PP 1 2 3 4")
        original, cleanup = ValueError("consumer"), RuntimeError("drain")
        e._drain_control = mock.Mock(side_effect=cleanup)
        acquired = e.ctl
        gen = e.generate_batched([1, 2], 4, {}, threading.Event())
        next(gen)
        successor = threading.Lock()
        successor.acquire()
        e.ctl = successor
        try:
            with self.assertRaises(ValueError) as caught:
                gen.throw(original)
            self.assertIs(caught.exception, original)
            self.assertIs(caught.exception.__cause__, cleanup)
            e._drain_control.assert_called_once_with("DONE", born=1)
            self.assertFalse(acquired.raw.locked())
            self.assertEqual(acquired.releases, 1)
            self.assertTrue(successor.locked())
            gen.close()
            self.assertEqual(acquired.releases, 1)
        finally:
            gen.close()
            successor.release()

    def test_trace_failure_after_normal_completion_is_reported(self):
        e = engine()
        e.slot_busy = [False, False]
        e.lines.put("DONE 0 2 1 1 length")
        failure = RuntimeError("final trace")
        def trace(*args):
            if args[0] == "finally phase":
                raise failure
        with mock.patch.object(server, "btrace", side_effect=trace):
            with self.assertRaises(RuntimeError) as caught:
                list(e.generate_batched([1, 2], 4, {}, threading.Event()))
        self.assertIs(caught.exception, failure)
        self.assertFalse(e.ctl.raw.locked())
        self.assertEqual(e.ctl.releases, 1)

    def test_cleanup_after_slot_handoff_never_releases_new_holder(self):
        e = engine()
        e.lines.put("DONE 0 2 1 1 length")
        e.lines.put("BADM 0 1")
        e.slot_q[0].put("BT 0 7")
        e._release_slot_when_done = mock.Mock(side_effect=RuntimeError("slot cleanup"))
        gen = e.generate_batched([1, 2], 4, {}, threading.Event())
        self.assertEqual(next(gen), 7)
        self.assertEqual(e.ctl.releases, 1)  # admission handed control back
        self.assertTrue(e.ctl.acquire(blocking=False))  # another request owns it
        original = ValueError("consumer")
        try:
            with self.assertRaises(ValueError) as caught:
                gen.throw(original)
            self.assertIs(caught.exception, original)
            self.assertTrue(e.ctl.raw.locked())
            self.assertEqual(e.ctl.releases, 1)
            e._release_slot_when_done.assert_called_once()
        finally:
            gen.close()
            e.ctl.release()

    def test_release_failure_does_not_mask_setup_failure_or_retry(self):
        e = engine()
        original, cleanup = MemoryError("setup"), RuntimeError("release")
        release = e.ctl.release
        def broken_release():
            release()
            raise cleanup
        with mock.patch.object(e.ctl, "release", side_effect=broken_release) as called, \
             mock.patch.object(e, "_capture_epoch", side_effect=original):
            gen = e.generate_batched([1, 2], 4, {}, threading.Event())
            with self.assertRaises(MemoryError) as caught:
                next(gen)
            self.assertIs(caught.exception, original)
            self.assertIs(caught.exception.__cause__, cleanup)
            gen.close()
            called.assert_called_once_with()
        self.assertFalse(e.ctl.raw.locked())
        self.assertEqual(e.ctl.releases, 1)
