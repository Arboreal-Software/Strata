"""Actual server pump over a recorded native pipe; no model/native-fit claims."""
import dataclasses
import json
import os
import sys
from pathlib import Path
import unittest
from unittest import mock
from serve import server
from serve.test_engine_epoch import batch_engine
from serve.two_slot_receipts import OwnedSlotObservations, OwnedTerminalPairing, history_digest
A='aaaaaaaaaaaa';B='bbbbbbbbbbbb';C='cccccccccccc'
def event(kind,**row):return kind+' '+json.dumps(row)+'\n'
def accept(slot,key,occupancy):
    return event('BOWN_ACCEPT',slot=slot,request=key,occupancy=occupancy,leg=slot+1,
                 input_count=1,input_sha256=history_digest([7]),full_budget=8,
                 cut=1,forwarded=1,accepted=2,watermark=0,history=history_digest([7,8]))
def retired(slot,key,occupancy,reason='cancel'):
    return event('BOWN_DONE',slot=slot,request=key,occupancy=occupancy,reason=reason,
                 watermark=1,history=history_digest([7,8,9]))
def observations():
    result=OwnedSlotObservations();result.register(0,A,[7],8);result.register(1,B,[7],8);return result
class PairingTests(unittest.TestCase):
    def ready(self):
        obs=observations();pair=OwnedTerminalPairing(obs);pair.observe(accept(0,A,11));pair.observe(accept(1,B,22));return obs,pair
    def test_both_native_orders_release_only_complete_pair(self):
        for terminal_first in (False,True):
            obs,pair=self.ready();terminal='BDONE 0 1 cancel 0.0\n';done=retired(0,A,11)
            first,second=(terminal,done)if terminal_first else(done,terminal)
            self.assertIsNone(pair.observe(first));self.assertEqual(pair.observe(second),(0,terminal));self.assertIn(1,obs.active);self.assertNotIn(0,obs.active)
    def test_duplicate_terminal_refuses(self):
        _,p=self.ready();line='BDONE 0 1 cancel 0.0\n';p.observe(line)
        with self.assertRaises(ValueError):p.observe(line)
    def test_duplicate_retirement_refuses(self):
        _,p=self.ready();line=retired(0,A,11);p.observe(line)
        with self.assertRaises(ValueError):p.observe(line)
    def test_reason_mismatch_refuses(self):
        _,p=self.ready();p.observe('BDONE 0 1 length 0.0\n')
        with self.assertRaises(ValueError):p.observe(retired(0,A,11,'nonfinite'))
    def test_foreign_nonce_refuses(self):
        o,p=self.ready();p.observe('BDONE 0 1 cancel 0.0\n')
        with self.assertRaises(ValueError):p.observe(retired(0,A,99))
        self.assertTrue(o.failed)
    def test_unbound_terminal_refuses(self):
        with self.assertRaises(ValueError):OwnedTerminalPairing(observations()).observe('BDONE 0 1 cancel 0.0\n')
    def test_admission_cannot_replace_unpaired_retired_occupancy(self):
        o,p=self.ready();p.observe(retired(0,A,11));o.register(0,C,[7],8)
        with self.assertRaises(ValueError):p.observe(accept(0,C,33))
    def test_retired_lane_cannot_emit_while_peer_continues(self):
        _,p=self.ready();p.observe(retired(0,A,11));self.assertIsNone(p.observe('BT 1 9\n'))
        for kind in ('BT 0 9\n',event('BOWN_COMMIT',slot=0),event('BOWN_KNOWN',slot=0)):
            _,p=self.ready();p.observe('BDONE 0 1 length 0.0\n')
            with self.assertRaises(ValueError):p.observe(kind)
    def test_old_retirement_cannot_complete_new_occupancy(self):
        o,p=self.ready();p.observe(retired(0,A,11));p.observe('BDONE 0 1 cancel 0.0\n');o.register(0,C,[7],8);p.observe(accept(0,C,33));p.observe('BDONE 0 1 cancel 0.0\n')
        with self.assertRaises(ValueError):p.observe(retired(0,A,11))
    def test_late_cancel_supersession_requires_all_actual_records(self):
        for reason in ('length','stop'):
            for retirement_first in (False,True):
                _,p=self.ready();self.assertIsNone(p.observe('BDONE 0 1 %s 1.0\n'%reason));cancel='BDONE 0 1 cancel 0.0\n';done=retired(0,A,11)
                first,second=(done,cancel)if retirement_first else(cancel,done)
                self.assertIsNone(p.observe(first));self.assertEqual(p.observe(second),(0,cancel))
    def test_replacement_wrong_count_refuses(self):
        _,p=self.ready();p.observe('BDONE 0 1 length 1.0\n');p.observe(retired(0,A,11))
        with self.assertRaises(ValueError):p.observe('BDONE 0 2 cancel 0.0\n')
    def test_provisional_cancel_never_tolerates_duplicate(self):
        _,p=self.ready();p.observe('BDONE 0 1 length 1.0\n');p.observe('BDONE 0 1 cancel 0.0\n')
        with self.assertRaises(ValueError):p.observe('BDONE 0 1 cancel 0.0\n')
    def test_cancel_candidate_cannot_replace_non_cancel_retirement(self):
        _,p=self.ready();p.observe('BDONE 0 1 length 1.0\n');p.observe('BDONE 0 1 cancel 0.0\n')
        with self.assertRaises(ValueError):p.observe(retired(0,A,11,'length'))
    def test_native_source_has_both_orders_and_no_post_terminal_commits(self):
        text=(Path(__file__).resolve().parents[1]/'src/program/generate.cpp').read_text()
        stop=text[text.index('        auto accept_batch_stop ='):text.index('        static const bool refill_parts',text.index('        auto accept_batch_stop ='))]
        self.assertLess(stop.index('report_http2_done('),stop.index('std::printf("BDONE'))
        step=text[text.index('        auto batch_step ='):text.index('        // ---- --batch-groups:',text.index('        auto batch_step ='))]
        self.assertIn('bs[(size_t)b].active',step);terminal=step[step.index('if (fin != nullptr) {'):];self.assertLess(terminal.index('sl.active = false;'),terminal.index('sl.repair_finishing=true'))
        # Direct call sites for these observations remain confined to batch_step.
        self.assertEqual(text.count('report_http2_known('),step.count('report_http2_known('));self.assertEqual(text.count('report_http2_commit('),step.count('report_http2_commit('))
class PumpTests(unittest.TestCase):
    def engine(self):
        e=batch_engine();epoch=e._capture_epoch();epoch.slots.owned_observations=observations();epoch=dataclasses.replace(epoch,strict_owned=True,receipt_env=());e._epoch=epoch;e.proc.terminate=mock.Mock(wraps=e.proc.terminate);return e,epoch
    def test_real_pump_cancel_retirement_first_keeps_peer_alive(self):
        e,epoch=self.engine();terminal='BDONE 0 1 cancel 0.0\n'
        def native():
            yield accept(0,A,11);yield accept(1,B,22);yield retired(0,A,11)
            self.assertTrue(e.slot_q[0].empty());self.assertFalse(e.proc.terminate.called)
            yield 'BT 1 9\n';yield terminal
            self.assertEqual(e.slot_q[0].get_nowait(),terminal);self.assertTrue(e.slot_q[0].empty());self.assertFalse(e.proc.terminate.called);self.assertIn(1,epoch.slots.owned_observations.active)
            yield 'BT 1 10\n'
        e.proc.stdout=native();e._pump(epoch);self.assertFalse(e.proc.terminate.called);self.assertEqual(e.slot_q[1].get_nowait(),'BT 1 9\n');self.assertEqual(e.slot_q[1].get_nowait(),'BT 1 10\n');self.assertIsNone(e.slot_q[0].get_nowait())
    def test_real_pump_terminal_first_waits_for_actual_retirement(self):
        e,epoch=self.engine();terminal='BDONE 0 1 length 1.0\n'
        def native():
            yield accept(0,A,11);yield accept(1,B,22);yield terminal
            self.assertTrue(e.slot_q[0].empty());yield 'BT 1 9\n';yield retired(0,A,11,'length')
            self.assertEqual(e.slot_q[0].get_nowait(),terminal);self.assertFalse(e.proc.terminate.called)
        e.proc.stdout=native();e._pump(epoch);self.assertFalse(e.proc.terminate.called)
    def test_real_pump_late_cancel_delivers_only_actual_replacement(self):
        for retirement_first in (False,True):
            e,epoch=self.engine();cancel='BDONE 0 1 cancel 0.0\n'
            def native():
                yield accept(0,A,11);yield accept(1,B,22);yield 'BDONE 0 1 length 1.0\n'
                self.assertTrue(e.slot_q[0].empty());first,second=(retired(0,A,11),cancel)if retirement_first else(cancel,retired(0,A,11))
                yield first;self.assertTrue(e.slot_q[0].empty());yield 'BT 1 9\n';yield second
                self.assertEqual(e.slot_q[0].get_nowait(),cancel);self.assertTrue(e.slot_q[0].empty());self.assertFalse(e.proc.terminate.called)
            e.proc.stdout=native();e._pump(epoch);self.assertFalse(e.proc.terminate.called)
    def test_missing_half_never_synthesizes_success(self):
        for terminal_only in (False,True):
            e,epoch=self.engine();e.proc.stdout=iter([accept(0,A,11),'BDONE 0 1 cancel 0.0\n'if terminal_only else retired(0,A,11)]);e._pump(epoch);self.assertIsNone(e.slot_q[0].get_nowait());self.assertTrue(e.slot_q[0].empty())
    def test_raw_two_slot_recorder_preserves_arrival_order(self):
        e,epoch=self.engine();e.proc.pid=4321
        epoch=dataclasses.replace(epoch,receipt_env=(("STRATA_HTTP_TWO_SLOT_RECEIPTS","1"),));e._epoch=epoch
        received=[accept(0,A,11),retired(0,A,11),'BDONE 0 1 cancel 0.0\n'];e.proc.stdout=iter(received)
        recorder=mock.Mock()
        with mock.patch('serve.two_slot_receipts.TwoSlotReceipt.enabled',return_value=recorder):e._pump(epoch)
        self.assertEqual([call.args[0]for call in recorder.observe.call_args_list],received)
        self.assertEqual(e.slot_q[0].get_nowait(),received[-1]);recorder.close.assert_called_once()
    def test_legacy_unowned_terminal_routes_without_retirement_protocol(self):
        e,epoch=self.engine();epoch=dataclasses.replace(epoch,strict_owned=False);e._epoch=epoch
        e.slot_busy[0]=True;e.slot_claims.add(0)  # Genuine in-flight ordinary lane, not stale output.
        terminal='BDONE 0 1 length 1.0\n';e.proc.stdout=iter([terminal]);e._pump(epoch)
        self.assertEqual(e.slot_q[0].get_nowait(),terminal);self.assertFalse(e.proc.terminate.called)
    def test_protocol_conflict_fails_captured_epoch(self):
        e,epoch=self.engine();e.proc.stdout=iter([accept(0,A,11),'BDONE 0 1 length 0.0\n',retired(0,A,11,'nonfinite')]);e._pump(epoch);e.proc.terminate.assert_called_once();self.assertIn('protocol failure',e.silent_note);self.assertIsNone(e.slot_q[0].get_nowait())

    @unittest.skipUnless(sys.platform.startswith('linux') and hasattr(os, 'memfd_create'),
                         'Linux sealed transport boundary')
    def test_peer_retirement_at_each_actual_image_wait_and_next_control(self):
        # Actual generate_batched/delivery/reply/pump and sealed producer FD.
        # Native payment/borrower replies and diagnostic/scheduler sinks are
        # recorded, not allocator authority or a native qualification.
        import os
        import queue
        import tempfile
        from serve.test_server import OwnedImageTransport
        from serve.test_batch_control_guard import CountLock
        for phase in ('BIPAY', 'BICAPTURED', 'BICLOSED'):
            for terminal_first in (False, True):
                def exercise():
                    with tempfile.TemporaryDirectory() as directory:
                        e, state = OwnedImageTransport._same_epoch_batch(self, directory, CountLock(), phase)
                        epoch = state.epoch
                        epoch.proc.pid = 4321
                        epoch = dataclasses.replace(epoch, receipt_env=(
                            ('STRATA_HTTP_NATIVE_RECEIPTS','1'), ('STRATA_HTTP_TWO_SLOT_RECEIPTS','1')))
                        # Same captured slots/queues/proc, no managed thread or
                        # scheduler authority is fabricated for this image caller.
                        e._epoch = epoch
                        first, _ = state.generator(A)
                        native_diagnostic, two_diagnostic = mock.Mock(), mock.Mock()
                        received = []
                        with mock.patch.object(server.time, 'monotonic', state.clock), \
                             mock.patch('serve.native_receipts.NativeReceipt.enabled', return_value=native_diagnostic), \
                             mock.patch('serve.two_slot_receipts.TwoSlotReceipt.enabled', return_value=two_diagnostic):
                            try:
                                self.assertIsNone(next(first))
                                self.assertEqual(state.phase, phase)
                                epoch.slots.owned_observations.register(1, B, [7], 8)
                                epoch.slots.slot_busy[1] = True
                                epoch.slots.slot_claims.add(1)
                                terminal = 'BDONE 1 1 cancel 0.0\n'
                                done = retired(1, B, 22)
                                def native():
                                    received.append(accept(1, B, 22)); yield received[-1]
                                    for line in ((terminal, done) if terminal_first else (done, terminal)):
                                        received.append(line); yield line
                                    self.assertEqual(epoch.slots.slot_q[1].get_nowait(), terminal)
                                    self.assertTrue(epoch.slots.slot_q[1].empty())
                                    self.assertTrue(epoch.slots.slot_q[0].empty())
                                    acknowledgement, slot = state.pending
                                    state.pending = None
                                    received.append(acknowledgement); yield acknowledgement
                                    if phase == 'BICLOSED':
                                        epoch.lines.put('DONE 0 3 0 0 length')
                                        epoch.lines.put('BADM 0 1')
                                        epoch.slots.slot_q[0].put('BT 0 7')
                                    self.assertEqual(next(first), 7)
                                    descriptor = state.records[0]['producer']
                                    with self.assertRaises(OSError): os.fstat(descriptor)
                                    self.assertFalse(getattr(epoch.slots.slot_q[0], '_owned_image_proof_lost', False))
                                    self.assertFalse(getattr(epoch.slots.slot_q[0], '_owned_image_close_uncertain', False))
                                    self.assertTrue(epoch.lines.empty())
                                    state.finish(0)
                                    self.assertEqual(list(first), [])
                                    self.assertFalse(e.ctl.raw.locked())
                                    self.assertEqual(e.vram(None)['vram_free_mib'], 4)
                                    self.assertTrue(epoch.lines.empty())
                                epoch.proc.stdout = native()
                                e._pump(epoch)
                                for diagnostic in (native_diagnostic, two_diagnostic):
                                    self.assertEqual([call.args[0] for call in diagnostic.observe.call_args_list], received)
                                    diagnostic.close.assert_called_once()
                                self.assertFalse(epoch.slots.owned_observations.failed)
                            finally:
                                state.abort()
                                first.close()
                with self.subTest(phase=phase, terminal_first=terminal_first):
                    OwnedImageTransport._lease_case(self, exercise)

    @unittest.skipUnless(sys.platform.startswith('linux') and hasattr(os, 'memfd_create'),
                         'Linux sealed transport boundary')
    def test_stale_diagnostics_during_expected_image_and_accepted_continuation(self):
        # Actual caller/pump/sealed FD; native records and diagnostic write sinks
        # are recorded. Callback delivery is not artifact inclusion or credit.
        import tempfile
        from serve.test_server import OwnedImageTransport
        from serve.test_batch_control_guard import CountLock
        from serve.native_receipts import NativeReceipt
        from serve.two_slot_receipts import TwoSlotReceipt, PREFIXES
        for phase in ('BIPAY', 'BICAPTURED', 'BICLOSED'):
            for terminal_first in (False, True):
                def exercise():
                    with tempfile.TemporaryDirectory() as directory:
                        e, state = OwnedImageTransport._same_epoch_batch(
                            self, directory, CountLock(), phase, pause_request=C, defer_admit_request=C)
                        epoch = state.epoch
                        epoch.proc.pid = 4321
                        epoch = dataclasses.replace(epoch, receipt_env=(
                            ('STRATA_HTTP_NATIVE_RECEIPTS', '1'), ('STRATA_HTTP_TWO_SLOT_RECEIPTS', '1')))
                        e._epoch = epoch
                        first, _ = state.generator(C)
                        native_diagnostic, two_diagnostic = mock.Mock(), mock.Mock()
                        native_diagnostic.observe.side_effect = lambda line: NativeReceipt.observe(native_diagnostic, line)
                        two_diagnostic.observe.side_effect = lambda line: TwoSlotReceipt.observe(two_diagnostic, line)
                        received = []
                        old_commit = dict(request=A, slot=0, occupancy=11, op=1, cut=2,
                                          forwarded=2, accepted=3, watermark=1, history=history_digest([7,8,9]))
                        stale = (event('BOWN_COMMIT', **old_commit, sample=9),
                                 event('BOWN_KNOWN', **old_commit, next_known=9))
                        with mock.patch.object(server.time, 'monotonic', state.clock), \
                             mock.patch('serve.native_receipts.NativeReceipt.enabled', return_value=native_diagnostic), \
                             mock.patch('serve.two_slot_receipts.TwoSlotReceipt.enabled', return_value=two_diagnostic):
                            try:
                                epoch.slots.owned_observations.register(0, A, [7], 8)
                                def native():
                                    received.append(accept(0, A, 11)); yield received[-1]
                                    terminal, done = 'BDONE 0 1 cancel 0.0\n', retired(0, A, 11)
                                    for line in ((terminal, done) if terminal_first else (done, terminal)):
                                        received.append(line); yield line
                                    self.assertEqual(epoch.slots.slot_q[0].get_nowait(), terminal)
                                    self.assertTrue(epoch.slots.slot_q[0].empty())
                                    self.assertNotIn(0, epoch.slots.owned_observations.active)
                                    self.assertIsNone(next(first))
                                    self.assertEqual(state.phase, phase)
                                    observations = epoch.slots.owned_observations
                                    self.assertEqual(observations.expected[0]['request'], C)
                                    self.assertNotIn(0, observations.active)
                                    with self.assertRaises(ValueError): observations.stop(0, C)
                                    before = (dict(observations.active), {key: dict(value) for key, value in observations.expected.items()},
                                              dict(observations.retired), dict(vars(epoch.slots.slot_q[0])),
                                              observations.stop(0, A), state.pending, tuple(state.frames))
                                    for line in stale:
                                        received.append(line); yield line
                                    self.assertEqual((dict(observations.active), {key: dict(value) for key, value in observations.expected.items()},
                                                      dict(observations.retired), dict(vars(epoch.slots.slot_q[0])),
                                                      observations.stop(0, A), state.pending, tuple(state.frames)), before)
                                    self.assertTrue(epoch.lines.empty())
                                    self.assertTrue(epoch.slots.slot_q[0].empty())
                                    # This advances the actual waiter without any C acknowledgement.
                                    self.assertIsNone(next(first))
                                    self.assertEqual(state.phase, phase)
                                    self.assertEqual(state.pending, before[5])
                                    state.release()  # Existing genuine ACK/FD handshake.
                                    self.assertIsNone(next(first))
                                    self.assertIsNotNone(state.deferred_admit)
                                    self.assertEqual(state.deferred_admit[1], 0)
                                    self.assertTrue(epoch.lines.empty())
                                    self.assertTrue(epoch.slots.slot_q[0].empty())
                                    self.assertEqual(observations.expected[0]['request'], C)
                                    self.assertNotIn(0, observations.active)
                                    self.assertTrue(any(frame.startswith('BICLOSE ') for frame in state.frames))
                                    with self.assertRaises(OSError): os.fstat(state.records[0]['producer'])
                                    # Native order: image handoff, ACCEPT, then BADM/token data.
                                    accepted = event('BOWN_ACCEPT', slot=0, request=C, occupancy=33, leg=2,
                                                     input_count=3, input_sha256=history_digest([1,1,2]), full_budget=32768,
                                                     cut=1, forwarded=3, accepted=4, watermark=1,
                                                     history=history_digest([1,1,2,7]))
                                    received.append(accepted); yield accepted
                                    self.assertEqual(observations.active[0].request, C)
                                    self.assertEqual(observations.stop(0, C), 'BSTOP 0 occupancy=33')
                                    before = (dict(observations.active), {key: dict(value) for key, value in observations.expected.items()},
                                              dict(observations.retired), dict(vars(epoch.slots.slot_q[0])),
                                              observations.stop(0, C), tuple(state.frames))
                                    for line in stale:
                                        received.append(line); yield line
                                    self.assertEqual((dict(observations.active), {key: dict(value) for key, value in observations.expected.items()},
                                                      dict(observations.retired), dict(vars(epoch.slots.slot_q[0])),
                                                      observations.stop(0, C), tuple(state.frames)), before)
                                    self.assertTrue(epoch.lines.empty())
                                    self.assertTrue(epoch.slots.slot_q[0].empty())
                                    admit, slot = state.deferred_admit
                                    state.deferred_admit = None
                                    admit(slot)  # Release the existing BADM/BT producer only now.
                                    self.assertEqual(next(first), 7)
                                    for line in ('BDONE 0 1 length 0\n',
                                                 event('BOWN_DONE', slot=0, request=C, occupancy=33, reason='length',
                                                       watermark=1, history=history_digest([1,1,2,7]))):
                                        received.append(line); yield line
                                    self.assertEqual(list(first), [])
                                    self.assertFalse(e.ctl.raw.locked())
                                    self.assertNotIn(0, observations.active)
                                    self.assertEqual(e.vram(None)['vram_free_mib'], 4)
                                    self.assertTrue(epoch.lines.empty())
                                epoch.proc.stdout = native()
                                e._pump(epoch)
                                for diagnostic in (native_diagnostic, two_diagnostic):
                                    self.assertEqual([call.args[0] for call in diagnostic.observe.call_args_list], received)
                                    diagnostic.close.assert_called_once()
                                # Execute actual artifact filters with recording write sinks:
                                # TwoSlotReceipt includes BOWN; NativeReceipt excludes it.
                                self.assertEqual([call.args[0]['line'] for call in two_diagnostic.write.call_args_list
                                                  if call.args[0]['kind'] == 'native'],
                                                 [line.rstrip('\n') for line in received if line.startswith(PREFIXES)])
                                self.assertEqual([call.args[0]['line'] for call in native_diagnostic.write.call_args_list
                                                  if call.args[0]['kind'] == 'native'],
                                                 [line.rstrip('\n') for line in received
                                                  if line.startswith(('HTTP_ACCEPT ', 'T ', 'DONE', 'ERR', 'BT ', 'BDONE '))])
                                self.assertFalse(epoch.slots.owned_observations.failed)
                            finally:
                                state.abort()
                                first.close()
                with self.subTest(phase=phase, terminal_first=terminal_first):
                    OwnedImageTransport._lease_case(self, exercise)

    def test_scheduler_sink_gets_only_validated_actual_retired_request(self):
        import queue
        from types import SimpleNamespace
        for terminal_first in (False, True):
            e, epoch = self.engine()
            # Transport sink only, not a fake issued thread/operation manager.
            sink = SimpleNamespace(observe=lambda line: False, retirements=queue.Queue(), close=mock.Mock())
            epoch = dataclasses.replace(epoch, scheduler=sink)
            e._epoch = epoch
            terminal, done = 'BDONE 1 1 cancel 0.0\n', retired(1,B,22)
            e.proc.stdout = iter([accept(0,A,11), accept(1,B,22),
                                  *((terminal,done) if terminal_first else (done,terminal))])
            e._pump(epoch)
            self.assertEqual(sink.retirements.get_nowait(), B)
            self.assertTrue(sink.retirements.empty())
            self.assertIn(0, epoch.slots.owned_observations.active)
            self.assertNotIn(1, epoch.slots.owned_observations.active)
            self.assertEqual(epoch.slots.slot_q[1].get_nowait(), terminal)
            self.assertIsNone(epoch.slots.slot_q[0].get_nowait())

    def test_optional_native_notification_inventory_is_consumed(self):
        e, epoch = self.engine()
        commit = dict(request=A, slot=0, occupancy=11, op=1, cut=2, forwarded=2,
                      accepted=3, watermark=1, history=history_digest([7,8,9]))
        received = [accept(0,A,11),
                    event('BOWN_GROUP', op=1, stages=2, active_rows=1, rows=[dict(slot=0, occupancy=11, position=1)]),
                    event('BOWN_COMMIT', **commit, sample=9), event('BOWN_KNOWN', **commit, next_known=9),
                    event('BOWN_ADMISSION_CANCEL', request=A, slot=0, phase='active'),
                    event('BOWN_STOP', request=A, slot=0, occupancy=11, last_op=1)]
        def native():
            yield received[0]
            before_active = dict(epoch.slots.owned_observations.active)
            before_expected = dict(epoch.slots.owned_observations.expected)
            before_retired = dict(epoch.slots.owned_observations.retired)
            yield from received[1:]
            # These informational payloads have no baseline schema validator.
            # Routing must not turn them into a nonce/cancel/admission authority.
            yield 'BOWN_GROUP {}\n'
            yield event('BOWN_STOP', request=B, slot=0, occupancy=99, last_op=1)
            yield event('BOWN_ADMISSION_CANCEL', request=C, slot=1, phase='queued')
            self.assertEqual(epoch.slots.owned_observations.active, before_active)
            self.assertEqual(epoch.slots.owned_observations.expected, before_expected)
            self.assertEqual(epoch.slots.owned_observations.retired, before_retired)
            self.assertTrue(epoch.lines.empty())
            self.assertFalse(epoch.slots.owned_observations.failed)
        e.proc.stdout = native(); e._pump(epoch)
        self.assertIsNone(epoch.lines.get_nowait()); self.assertTrue(epoch.lines.empty())
        e.proc.terminate.assert_not_called()

    def test_unknown_malformed_foreign_duplicate_notifications_fail_closed(self):
        # Only inherited ACCEPT/DONE and post-terminal COMMIT/KNOWN checks.
        # Informational record contents do not acquire new schema authority.
        invalid = [accept(0,A,11), retired(0,A,99),
                   'BOWN_DONE {"slot":0,"slot":0}\n',
                   'BOWN_ACCEPT {"slot":0}\n']
        for line in invalid:
            with self.subTest(line=line):
                e, epoch = self.engine()
                e.proc.stdout = iter([accept(0,A,11), line])
                e._pump(epoch)
                e.proc.terminate.assert_called_once()
                self.assertTrue(epoch.slots.owned_observations.failed)
                self.assertIsNone(epoch.lines.get_nowait())
                self.assertTrue(epoch.lines.empty())
                self.assertIsNone(epoch.slots.slot_q[0].get_nowait())

    @unittest.skipUnless(sys.platform.startswith('linux') and hasattr(os, 'memfd_create'),
                         'Linux sealed transport boundary')
    def test_bad_peer_notification_retains_actual_submitted_image_fd(self):
        import os
        import subprocess
        import tempfile
        from serve.test_server import OwnedImageTransport
        from serve.test_batch_control_guard import CountLock
        for bad in ('BOWN_FUTURE {}\n', 'BOWN_ERROR invalid_done\n', 'BOWN_DONE\n',
                    retired(1,B,99), accept(1,B,22)):
            def exercise():
                with tempfile.TemporaryDirectory() as directory:
                    e, state = OwnedImageTransport._same_epoch_batch(self, directory, CountLock(), 'BICAPTURED')
                    epoch = state.epoch
                    first, _ = state.generator(A)
                    try:
                        with mock.patch.object(server.time, 'monotonic', state.clock):
                            self.assertIsNone(next(first))
                            descriptor = state.records[0]['producer']
                            os.fstat(descriptor)
                            epoch.slots.owned_observations.register(1,B,[7],8)
                            epoch.proc.stdout = iter([accept(1,B,22), bad])
                            with mock.patch.object(epoch.proc, 'wait', side_effect=subprocess.TimeoutExpired('recorded child', 2)):
                                e._pump(epoch)
                                with self.assertRaises(server.EngineDied): next(first)
                            self.assertEqual(epoch.slots.owned_observations.failed,
                                             bad.startswith(('BOWN_DONE ', 'BOWN_ACCEPT ')))
                            self.assertTrue(epoch.slots.slot_q[0]._owned_image_proof_lost)
                            self.assertEqual(epoch.slots.slot_q[0]._owned_image_transport_fd, descriptor)
                            os.fstat(descriptor)
                            self.assertTrue(e.ended)
                            if not epoch.slots.owned_observations.failed:
                                # Unknown/error/missing-space reached the strict
                                # reply rejection; pump EOF still waits behind it.
                                self.assertIsNone(epoch.lines.get_nowait())
                            self.assertTrue(epoch.lines.empty())
                            epoch.proc.code = 0
                            e._reap_owned_image_transports(epoch)
                            self.assertIsNone(epoch.slots.slot_q[0]._owned_image_transport_fd)
                            with self.assertRaises(OSError): os.fstat(descriptor)
                    finally:
                        state.abort()
                        first.close()
                        if getattr(epoch.slots.slot_q[0], '_owned_image_transport_fd', None) is not None:
                            epoch.proc.code = 0
                            e._reap_owned_image_transports(epoch)
            with self.subTest(bad=bad): OwnedImageTransport._lease_case(self, exercise)

if __name__=='__main__':unittest.main()
