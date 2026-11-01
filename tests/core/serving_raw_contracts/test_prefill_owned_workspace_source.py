#!/usr/bin/env python3
"""Wiring checks only; these do not prove CUDA allocation/layout or destruction."""
from pathlib import Path
import unittest
from test_serving_thread_bridge_source import body
from generate_prefill_owned_workspace_test import generate

ROOT = Path(__file__).resolve().parents[3]


class PaidPrefillWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cpp = (ROOT / 'src/prefill/prefill.cpp').read_text()
        cls.driver = (ROOT / 'src/program/generate.cpp').read_text()
        cls.header = (ROOT / 'include/strata/prefill/prefill.hpp').read_text()

    def test_actual_driver_quotes_after_owner_selection_before_init(self):
        setup = body(self.driver, 'auto init_prompt_paths =')
        self.assertLess(setup.index('st.sp.set_stage('), setup.index('admit_prefill_workspace(st.sp'))
        self.assertLess(setup.index('OnDevice on(st.dev)'), setup.index('admit_prefill_workspace(st.sp'))
        self.assertLess(setup.index('st.sp.set_cpu_pool('), setup.index('admit_prefill_workspace(st.sp'))
        self.assertLess(setup.index('admit_prefill_workspace(st.sp'), setup.index('st.sp.init('))
        self.assertLess(setup.index('sp.set_stage('), setup.index('admit_prefill_workspace(sp,'))
        self.assertLess(setup.index('owned_primary_prefill_device'), setup.index('admit_prefill_workspace(sp,'))
        self.assertLess(setup.index('admit_prefill_workspace(sp,'), setup.index('sp.init(wt'))
        callback = body(self.driver, 'auto admit_prefill_workspace =')
        self.assertIn('model_budget->reserve(resources,strata::core::ReservationPurpose::foreground)', callback)
        self.assertIn('resources.device_bytes[size_t(device)]=quote.device', callback)
        self.assertIn('if(owned_batch_allocation)', callback)

    def test_release_build_still_executes_helper_contract_checks(self):
        test = (ROOT / 'tests/core/serving_raw_contracts/test_prefill_owned_workspace.cpp').read_text()
        self.assertLess(test.index('#undef NDEBUG'), test.index('#include'))

    def test_producer_quotes_actual_shape_and_rejects_extra_consumers(self):
        producer = body(self.cpp, 'bool Prefill::admit_owned_workspace(')
        for actual in ['sizeof(Impl)', 'sizeof(Stager::Job)', 'core::PinnedRing::kSlots',
                       'core::PinnedRing::minimum_slot_bytes()', 'bytes_needed_owned(g,ss,chunk,source!=nullptr)',
                       'source->transient(layer,expert)', 'workspace_credit_=std::move(credit)']:
            self.assertIn(actual, producer)
        self.assertIn('cpu_pool_||helper_||impl_->pp||impl_->help_pp||embd_rows||on_routes||on_taps', producer)
        self.assertLess(producer.index('owned_workspace_quote('), producer.index('admit(quote,credit)'))

    def test_init_checks_identity_before_any_allocation(self):
        init = body(self.cpp, 'bool Prefill::init(')
        check = init.index('if(workspace_credit_&&')
        self.assertLess(check, init.index('retirement_init_started_ = true'))
        for field in ['&g!=workspace_geometry_', '&ss!=workspace_session_', 'src!=workspace_source_',
                      'chunk!=workspace_chunk_', 'bool(borrow)!=workspace_borrowed_', 'bool(next_)!=workspace_successor_']:
            self.assertIn(field, init[:init.index('retirement_init_started_ = true')])

    def test_real_cleanup_precedes_credit_release(self):
        release = body(self.cpp, 'void Prefill::release()')
        last = release.rindex('workspace_credit_.reset()')
        for completion in ['impl_->hand_worker.reset()', 'impl_->stager.reset()', 'impl_->up_ring.retire_strict()',
                           'for (void* p : impl_->owned)', 'if (strict_retirement_) impl_.reset()']:
            self.assertLess(release.index(completion), last)
        self.assertLess(self.header.index('workspace_credit_;'), self.header.index('std::unique_ptr<Impl> impl_;'))
        retry = self.driver[self.driver.index('const int r = init_prompt_paths();'):]
        self.assertLess(retry.index('stp->sp.reset()'), retry.index('o.prefill_chunk = next'))
        self.assertLess(retry.index('sp.reset()'), retry.index('o.prefill_chunk = next'))

    def test_first_unit_keeps_eligibility_closed(self):
        eligibility = (ROOT / 'include/strata/core/native_owned_batch_serving.hpp').read_text()
        self.assertIn('p.prefill_chunk==16', eligibility)
        self.assertIn('!p.prefill_auto', eligibility)
        generated = generate(ROOT)
        self.assertIn(body(self.cpp, 'bool Prefill::admit_owned_workspace('), generated)
        self.assertIn(body(self.driver, 'auto admit_prefill_workspace ='), generated)


if __name__ == '__main__':
    unittest.main()
