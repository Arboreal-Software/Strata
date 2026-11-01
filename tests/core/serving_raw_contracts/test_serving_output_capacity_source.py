#!/usr/bin/env python3
"""Source wiring checks complement the compiled production storage/body tests."""
from pathlib import Path
import unittest
from generate_serving_output_capacity_test import generate
from test_serving_thread_bridge_source import body

ROOT = Path(__file__).resolve().parents[3]


class OutputCapacitySource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / 'src/program/generate.cpp').read_text()
        cls.controller = (ROOT / 'src/program/resident_repair_controller.inc').read_text()

    def test_full_bgen_budget_checked_before_media_or_native_work(self):
        begin = self.source.index('const int64_t n = (int64_t) ids.size();', self.source.index('ERR bad request:'))
        check = self.source.index('ids.size(),uint64_t(admit_max_new),uint64_t(o.max_context)', begin)
        self.assertLess(check, self.source.index('req_imgs.clear();', begin))
        owned = body(self.source, 'auto admit_owned_repair_slot =')
        self.assertIn('serving_output_fits(', owned)
        self.assertLess(owned.index('serving_output_fits('), owned.index('fresh_owned_slot('))

    def test_initial_storage_prepared_before_state_install(self):
        start = self.source.index('rrd::ResidentRepairController::ReceiptHistory initial_receipts;')
        self.assertLess(self.source.index('initial_receipts.prepare(', start), self.source.index('session.restore(out.splice.bundle()', start))
        self.assertIn('repair->receipts_.commit(receipt_growth)', self.source[start:])

    def test_continuation_preparation_precedes_displacement(self):
        prepare = body(self.source, 'static bool prepare_continuation_request(')
        self.assertIn('c.receipts_.prepare(required', prepare)
        self.assertIn('serving_output_history_quote(incoming.size(),budget,bytes)', prepare)
        self.assertIn('publication.accepted_tokens.reserve', prepare)
        activate = body(self.source, 'static bool activate_retained_owner(')
        self.assertLess(activate.index('prepare_continuation_request('), activate.index('session.capture('))
        self.assertLess(activate.index('prepare_continuation_request('), activate.index('registry.activate_parked_owner('))
        adopt = body(self.source, 'static void adopt_continuation_request(')
        self.assertNotIn('.reserve(', adopt)
        self.assertIn('c.receipts_.commit(prepared.receipts)', adopt)
        self.assertIn('prepared.publication.cut!=owner.published.cut', adopt)

    def test_receipt_growth_does_not_change_region_or_other_mode_guards(self):
        self.assertIn('ReceiptHistory receipts_;', self.controller)
        self.assertIn('std::array<core::RepairRegionRootBinding,64> region_roots{};', self.controller)
        self.assertIn('receipts_.snapshot(receipt_count_', self.controller)
        self.assertIn('fatal("cohort RAW commit refused after authentic group completion")', self.controller)
        eligibility = (ROOT / 'include/strata/core/native_owned_batch_serving.hpp').read_text()
        for restriction in ['p.spec==0', 'p.batch_groups==1', 'p.prefill_chunk==16', '!p.vision']:
            self.assertIn(restriction, eligibility)
        for removed in ['facts.max_generated>64', 'budget>64||!fresh_owned_slot', 'bimg||mn>64']:
            self.assertNotIn(removed, self.source)

    def test_actual_extracted_cpp_contains_production_bodies(self):
        actual = generate(ROOT)
        self.assertIn(body(self.source, 'static bool prepare_continuation_request('), actual)
        self.assertIn(body(self.source, 'static void adopt_continuation_request('), actual)
        self.assertIn('ERR owned BGEN prompt/output exceeds context', actual)


if __name__ == '__main__':
    unittest.main()
