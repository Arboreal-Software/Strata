"""Actual server pump over a recorded native pipe; no model/native-fit claims."""
import dataclasses
import json
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
        self.assertIn('bs[(size_t) b].active',step);terminal=step[step.index('if (fin != nullptr) {'):];self.assertLess(terminal.index('sl.active = false;'),terminal.index('sl.repair_finishing=true'))
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
        terminal='BDONE 0 1 length 1.0\n';e.proc.stdout=iter([terminal]);e._pump(epoch)
        self.assertEqual(e.slot_q[0].get_nowait(),terminal);self.assertFalse(e.proc.terminate.called)
    def test_protocol_conflict_fails_captured_epoch(self):
        e,epoch=self.engine();e.proc.stdout=iter([accept(0,A,11),'BDONE 0 1 length 0.0\n',retired(0,A,11,'nonfinite')]);e._pump(epoch);e.proc.terminate.assert_called_once();self.assertIn('protocol failure',e.silent_note);self.assertIsNone(e.slot_q[0].get_nowait())
if __name__=='__main__':unittest.main()
