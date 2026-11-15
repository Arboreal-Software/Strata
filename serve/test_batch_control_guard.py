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
        send = e._send
        def admit(text, **kwargs):
            send(text, **kwargs)
            # Real admission discards queued output before sending BGEN.
            # A reply can only arrive after that command reaches the child.
            if text.startswith("BGEN "):
                e.slot_q[0].put("BT 0 7")
        e._send = admit
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

    def _owned_case(self, action):
        import tempfile
        from serve.test_server import OwnedImageTransport
        with tempfile.TemporaryDirectory() as directory:
            OwnedImageTransport._lease_case(self, lambda: action(directory))

    def _owned_pair(self, directory, phase):
        from serve.test_server import OwnedImageTransport
        return OwnedImageTransport._same_epoch_batch(self, directory, CountLock(), phase)

    def _elastic_service(self, e, running):
        from types import SimpleNamespace
        svc = server.Service.__new__(server.Service)
        svc.engine, svc.fifo = e, threading.Lock()
        svc.vision_held, svc.last_image_at = running, 0
        svc.vision_no_shrink = svc.vision_no_shrink_said = False  # actual Service constructor defaults
        svc.vram_reserve_asked, svc.vram_reserve = False, None
        svc.loaded = e.alive
        svc.free_vram_mib = lambda: None  # no telemetry/GPU access
        vision = SimpleNamespace(elastic=True, elastic_mib=2048, elastic_idle_s=60,
                                 running=running, starts=0, unloads=0)
        vision.alive = lambda: vision.running
        def restart():
            vision.running = True; vision.starts += 1
        def unload():
            vision.running = False; vision.unloads += 1
        vision.restart, vision.unload = restart, unload
        svc.vision = vision  # encoder endpoint recording only
        return svc

    def test_same_epoch_owned_images_serialize_all_reply_phases_and_badm(self):
        def exercise(directory):
            for phase in ("BIPAY", "BICAPTURED", "BICLOSED", "BADM"):
                with self.subTest(phase=phase):
                    e, state = self._owned_pair(directory, phase)
                    first, _ = state.generator("aaaaaaaaaaaa")
                    peer, _ = state.generator("bbbbbbbbbbbb")
                    attempted, decoded, finish_peer = threading.Event(), threading.Event(), threading.Event()
                    errors, output = [], []
                    acquired = e.ctl
                    original_acquire = acquired.acquire
                    def acquire(*args, **kwargs):
                        if threading.current_thread().name == "same-epoch-peer":
                            attempted.set()
                        return original_acquire(*args, **kwargs)  # actual real lock
                    acquired.acquire = acquire
                    def worker():
                        try:
                            while True:
                                value = next(peer)
                                if value is not None:
                                    output.append(value); decoded.set(); break
                            self.assertTrue(finish_peer.wait(2))
                            output.extend(peer)
                        except BaseException as error:
                            errors.append(error); decoded.set()
                    thread = threading.Thread(target=worker, name="same-epoch-peer", daemon=True)
                    with mock.patch.object(server.time, "monotonic", side_effect=state.clock):
                        try:
                            self.assertIsNone(next(first))
                            self.assertEqual(state.phase, phase)
                            self.assertIs(state.epoch.ctl, acquired)
                            self.assertTrue(acquired.raw.locked())
                            thread.start()
                            self.assertTrue(attempted.wait(1))
                            self.assertEqual(sum(f.startswith("BIPREP ") for f in state.frames), 1)
                            self.assertFalse(any("bbbbbbbbbbbb" in str(line) for _, line in state.reads))
                            state.release()
                            self.assertEqual(next(first), 7)
                            self.assertTrue(decoded.wait(2))
                            self.assertEqual(errors, [])
                            self.assertEqual(output, [9])
                            self.assertTrue(all(state.epoch.slots.slot_busy))
                            self.assertEqual(sum(f.startswith("BIPREP ") for f in state.frames), 2)
                            self.assertEqual(acquired.releases, 2)  # each released only after BADM
                            # Reverse slot completion; slot0 remains decoding while slot1 admitted.
                            state.finish(1); finish_peer.set(); thread.join(2)
                            self.assertFalse(thread.is_alive()); self.assertEqual(errors, [])
                            state.finish(0); self.assertEqual(list(first), [])
                            self.assertFalse(any(state.epoch.slots.slot_busy))
                            self.assertFalse(state.epoch.proc.stdin.closed)
                        finally:
                            finish_peer.set(); state.abort()
                            first.close()
                            if thread.ident is not None:
                                thread.join(2)
                                self.assertFalse(thread.is_alive())
                            peer.close()
        self._owned_case(exercise)

    def test_same_epoch_owned_image_waiting_cancel_and_capture_cleanup(self):
        def exercise(directory):
            for ending in ("queued_cancel", "active_cancel", "capture_close"):
                with self.subTest(ending=ending):
                    e, state = self._owned_pair(directory, "BICAPTURED")
                    first, cancel = state.generator("aaaaaaaaaaaa")
                    peer, peer_cancel = state.generator("bbbbbbbbbbbb")
                    attempted, peer_done = threading.Event(), threading.Event()
                    errors, peer_values = [], []
                    original = e.ctl.acquire
                    def acquire(*args, **kwargs):
                        if threading.current_thread().name == "canceling-peer":attempted.set()
                        return original(*args, **kwargs)
                    e.ctl.acquire = acquire
                    def worker():
                        try:peer_values.extend(peer)
                        except BaseException as error:errors.append(error)
                        finally:peer_done.set()
                    thread = threading.Thread(target=worker, name="canceling-peer", daemon=True)
                    with mock.patch.object(server.time, "monotonic", side_effect=state.clock):
                        try:
                            self.assertIsNone(next(first))
                            producer = state.records[0]["producer"]
                            server.os.fstat(producer)  # held behind the pending actual ACK
                            thread.start(); self.assertTrue(attempted.wait(1))
                            peer_cancel.set()
                            self.assertTrue(peer_done.wait(2)); thread.join(2)
                            self.assertFalse(thread.is_alive()); self.assertEqual(errors, [])
                            self.assertEqual(peer_values, [])
                            self.assertTrue(e.ctl.raw.locked())
                            self.assertFalse(any("bbbbbbbbbbbb" in frame for frame in state.frames))
                            if ending == "active_cancel":cancel.set()
                            state.release()  # genuine recording ACK, actual borrower already closed
                            if ending == "capture_close":first.close()
                            elif ending == "active_cancel":self.assertEqual(list(first), [])
                            else:
                                self.assertEqual(next(first), 7)
                                state.finish(0); self.assertEqual(list(first), [])
                            self.assertFalse(e.ctl.raw.locked())
                            self.assertEqual(e.ctl.releases, 1)
                            self.assertFalse(state.epoch.proc.stdin.closed)
                            with self.assertRaises(OSError):server.os.fstat(producer)
                            if ending != "queued_cancel":
                                self.assertTrue(any(f.startswith("BICLOSE 0 ") and f.endswith("cancel=1") for f in state.frames))
                            self.assertFalse(any(state.epoch.slots.slot_busy))
                        finally:
                            peer_cancel.set(); state.abort(); first.close()
                            if thread.ident is not None:
                                thread.join(2); self.assertFalse(thread.is_alive())
                            peer.close()
        self._owned_case(exercise)

    def test_same_epoch_owned_image_eof_replacement_and_unsolicited_replies(self):
        def exercise(directory):
            from serve.test_engine_epoch import replace
            cases = ("eof", "replacement", "close_confirmation", "wrong_slot", "stale_request", "malformed")
            for ending in cases:
                with self.subTest(ending=ending):
                    phase = "BICAPTURED" if ending in ("eof", "replacement") else "BICLOSED" if ending == "close_confirmation" else "BIPAY"
                    e, state = self._owned_pair(directory, phase)
                    first, _ = state.generator("aaaaaaaaaaaa")
                    acquired, old = e.ctl, state.epoch
                    with mock.patch.object(server.time, "monotonic", side_effect=state.clock):
                        try:
                            self.assertIsNone(next(first))
                            if ending == "replacement":
                                replace(e)
                                successor = e._capture_epoch()
                                successor.lines.put("untouched successor control")
                                with self.assertRaises(server.EngineDied):next(first)
                                self.assertEqual(successor.lines.get_nowait(), "untouched successor control")
                                self.assertEqual(successor.proc.stdin.text, "")
                                self.assertFalse(successor.proc.stdin.closed)
                                self.assertTrue(e.alive())
                            elif ending == "close_confirmation":
                                first.close()  # sent close without receipt: original epoch fenced
                                self.assertTrue(old.proc.stdin.closed)
                            else:
                                messages = {"eof": None,
                                    "wrong_slot": "BIPAY 1 request=bbbbbbbbbbbb incarnation=8 bytes=52 tokens=3 budget=32768",
                                    "stale_request": "BIPAY 0 request=cccccccccccc incarnation=7 bytes=52 tokens=3 budget=32768",
                                    "malformed": "BIPAY 0 request=aaaaaaaaaaaa incarnation=7"}
                                old.lines.put(messages[ending])
                                with self.assertRaises(server.EngineDied):next(first)
                                self.assertTrue(old.proc.stdin.closed)
                                self.assertFalse(e.alive())
                            self.assertFalse(acquired.raw.locked()); self.assertEqual(acquired.releases, 1)
                            for channel in old.slots.slot_q:
                                self.assertIsNone(getattr(channel, "_owned_image_transport_fd", None))
                            for record in state.records.values():
                                if record["producer"] is not None:
                                    with self.assertRaises(OSError):server.os.fstat(record["producer"])
                        finally:
                            state.abort(); first.close()
        self._owned_case(exercise)

    def test_same_epoch_elastic_idle_leaves_image_payment_unread_until_release(self):
        def exercise(directory):
            e, state = self._owned_pair(directory, "BIPAY")
            svc = self._elastic_service(e, True)
            first, _ = state.generator("aaaaaaaaaaaa")
            with mock.patch.object(server.time, "monotonic", side_effect=state.clock):
                try:
                    self.assertIsNone(next(first))
                    state.release()  # a valid payment is now in the SAME real control queue
                    result = svc._vision_idle_down(force=True)
                    self.assertEqual(result, "busy", f"frames={state.frames}; reads={state.reads}")
                    self.assertEqual(svc.vision.unloads, 0)
                    self.assertTrue(svc.vision_held)
                    self.assertFalse(any(f.startswith("VRAM") for f in state.frames))
                    self.assertEqual(state.epoch.lines.qsize(), 1)
                    self.assertEqual(next(first), 7)
                    state.finish(0); self.assertEqual(list(first), [])
                    self.assertEqual(svc._vision_idle_down(force=True), "down")
                    self.assertEqual(svc.vision.unloads, 1)
                    self.assertEqual([f for f in state.frames if f.startswith("VRAM")], ["VRAM 0"])
                    self.assertFalse(e.ctl.raw.locked())
                    self.assertTrue(svc.fifo.acquire(blocking=False)); svc.fifo.release()
                finally:
                    state.abort(); first.close()
        self._owned_case(exercise)

    def test_same_epoch_elastic_start_waits_then_releases_captured_control(self):
        def exercise(directory):
            e, state = self._owned_pair(directory, "BIPAY")
            svc = self._elastic_service(e, False)
            first, _ = state.generator("aaaaaaaaaaaa")
            original = e.ctl
            attempted, finished = threading.Event(), threading.Event()
            errors = []
            acquire = original.acquire
            def taking(*args, **kwargs):
                if threading.current_thread().name == "elastic-start":attempted.set()
                return acquire(*args, **kwargs)
            original.acquire = taking
            successor = threading.Lock(); successor.acquire()
            def vram_started():
                self.assertTrue(original.raw.locked())
                e.ctl = successor  # only the acquired object may be released
            state.on_vram = vram_started
            def worker():
                try:
                    with svc._turn():svc._vision_up()
                except BaseException as error:errors.append(error)
                finally:finished.set()
            thread = threading.Thread(target=worker, name="elastic-start", daemon=True)
            with mock.patch.object(server.time, "monotonic", side_effect=state.clock),                     mock.patch.object(server.time, "sleep", return_value=None):
                try:
                    self.assertIsNone(next(first))
                    thread.start(); self.assertTrue(attempted.wait(1))
                    self.assertEqual(svc.vision.starts, 0)
                    self.assertFalse(any(f.startswith("VRAM") for f in state.frames))
                    state.release(); self.assertEqual(next(first), 7)
                    self.assertTrue(finished.wait(2)); thread.join(2)
                    self.assertFalse(thread.is_alive()); self.assertEqual(errors, [])
                    self.assertEqual(svc.vision.starts, 1)
                    self.assertTrue(svc.vision_held)
                    self.assertEqual([f for f in state.frames if f.startswith("VRAM")], ["VRAM 2048"])
                    self.assertFalse(original.raw.locked())
                    self.assertTrue(successor.locked())
                    state.finish(0); self.assertEqual(list(first), [])
                    self.assertTrue(svc.fifo.acquire(blocking=False)); svc.fifo.release()
                finally:
                    state.abort(); first.close()
                    if thread.ident is not None:
                        thread.join(2); self.assertFalse(thread.is_alive())
                    e.ctl = original
                    successor.release()
        self._owned_case(exercise)

    def test_new_ready_epoch_reserve_cannot_consume_prepared_image_payment(self):
        def exercise(directory):
            for outcome in ("reply", "refusal", "busy", "none"):
                with self.subTest(outcome=outcome):
                    e, state = self._owned_pair(directory, "BIPAY")
                    svc = self._elastic_service(e, True)
                    del svc.loaded  # actual Service.loaded -> actual StrataEngine.alive
                    svc.before_load, svc.min_free_vram_mib, svc.restore_after_load = None, 0, None
                    svc.vram_reserve = None if outcome == "none" else 512
                    if outcome == "busy":svc.vram_wait_s = 0  # actual failed acquire, no wait/cap increase
                    if outcome == "refusal":state.vram_reply = "ERR VRAM native batch busy"
                    old_epoch, captured = state.epoch, e.ctl
                    first, _ = state.generator("aaaaaaaaaaaa")  # source prepared BEFORE restart, not yet sent
                    e.unload()  # actual close of recording Process, no model/engine binary
                    ready, resume, progressed, attempted, finished = (threading.Event() for _ in range(5))
                    errors = []
                    successor = threading.Lock(); successor.acquire()
                    def published():
                        ready.set()
                        self.assertTrue(resume.wait(2))
                    state.on_ready = published
                    acquire = captured.acquire
                    def acquiring(*args, **kwargs):
                        if threading.current_thread().name == "post-ready-loader":
                            attempted.set(); progressed.set()
                        return acquire(*args, **kwargs)
                    captured.acquire = acquiring
                    def vram_seen():
                        # Do not raise before the native-format reply: old-block
                        # negative must actually read/discard the queued payment.
                        e.ctl = successor
                    if outcome == "reply":state.on_vram = vram_seen
                    def loader():
                        try:
                            with svc._turn():svc.ensure_loaded()
                        except BaseException as error:errors.append(error)
                        finally:finished.set(); progressed.set()
                    thread = threading.Thread(target=loader, name="post-ready-loader", daemon=True)
                    with mock.patch.object(server.time, "monotonic", side_effect=state.clock):
                        try:
                            thread.start(); self.assertTrue(ready.wait(1))
                            self.assertTrue(svc.loaded())
                            self.assertTrue(svc.fifo.locked())
                            self.assertIsNot(state.epoch, old_epoch)
                            self.assertIsNot(state.epoch.proc, old_epoch.proc)
                            self.assertIsNot(state.epoch.lines, old_epoch.lines)
                            self.assertEqual(state.epoch.gen, old_epoch.gen+1)
                            self.assertIs(state.epoch.ctl, captured)
                            self.assertIsNone(next(first))  # actual NEW-epoch BIPREP, holds ctl
                            state.release()  # valid new payment queued while image reader is paused
                            resume.set(); self.assertTrue(progressed.wait(1))
                            self.assertFalse(any(f.startswith("VRAM") for f in state.frames),
                                             f"unsafe reserve frames={state.frames}; reads={state.reads}")
                            self.assertEqual(state.epoch.lines.qsize(), 1)
                            if outcome == "none":
                                self.assertTrue(finished.wait(2)); self.assertEqual(errors, [])
                                self.assertFalse(attempted.is_set())
                            else:
                                self.assertTrue(attempted.is_set())
                            if outcome == "busy":
                                self.assertTrue(finished.wait(2))
                                self.assertEqual(len(errors), 1)
                                self.assertIsInstance(errors[0], server.ModelBusy)
                                self.assertTrue(captured.raw.locked())  # image owner untouched
                            self.assertEqual(next(first), 7)  # real capture/producer close/BICLOSED/BADM
                            self.assertTrue(finished.wait(2)); thread.join(2)
                            self.assertFalse(thread.is_alive())
                            if outcome != "busy":self.assertEqual(errors, [])
                            expected = [] if outcome in ("none", "busy") else ["VRAM 512"]
                            self.assertEqual([f for f in state.frames if f.startswith("VRAM")], expected)
                            self.assertFalse(captured.raw.locked())
                            self.assertEqual(captured.releases, 1 if outcome in ("none", "busy") else 2)
                            if outcome == "reply":self.assertTrue(successor.locked())
                            state.finish(0); self.assertEqual(list(first), [])
                            self.assertFalse(state.epoch.proc.stdin.closed)
                            self.assertTrue(svc.fifo.acquire(blocking=False)); svc.fifo.release()
                        finally:
                            resume.set(); state.abort(); first.close()
                            thread.join(2); self.assertFalse(thread.is_alive())
                            e.ctl = captured; successor.release()
        self._owned_case(exercise)
