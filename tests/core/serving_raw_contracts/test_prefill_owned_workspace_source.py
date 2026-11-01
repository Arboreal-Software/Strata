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
        producer = body(self.cpp, 'WorkspaceAdmission Prefill::admit_owned_workspace(')
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

    def test_paid_init_resolves_policy_without_observer_side_effects(self):
        init = body(self.cpp, 'bool Prefill::init(')
        resolved = init.index('if(workspace_credit_&&cpu_share_env()!=0.0)')
        self.assertLess(resolved, init.index('retirement_init_started_ = true'))
        observer = body(self.cpp, 'core::PrefillCpuShareObservation Prefill::cpu_share_policy() const noexcept')
        self.assertNotIn('cpu_share_env()', observer)
        self.assertNotIn('cpu_share_explicit()', observer)
        self.assertIn('r.explicit_resolved = g_share_explicit_observed', observer)

    def test_preinstall_refusals_keep_same_request_running_for_exact_tail(self):
        helper = body(self.driver, 'auto sfx_resume_exact =')
        self.assertIn('serving_raw_session.get()!=main.get()', helper)
        self.assertIn('main->cut()!=prefix_cut', helper)
        self.assertIn('!main->request_open()', helper)
        self.assertLess(helper.index('main->begin_work('), helper.index('serving_ordinary_fence=true'))
        self.assertNotIn('close_request(', helper)
        self.assertIn('serving_raw_fence_running||serving_ordinary_fence||serving_suffix_active', helper)
        self.assertEqual(self.driver.count('sfx_resume_exact(prefix_main,prefix_cut)'), 3)
        section = self.driver[self.driver.index('if (to == sfx_repair_at && serving_raw_fence_running)'):]
        self.assertLess(section.index('const auto prefix_main=serving_raw_session'), section.index('drain_work('))
        for refusal in ['prefix schedule refused before install', 'prefix capture refused',
                        'prefix base refused before install']:
            branch = section[section.index(refusal):]
            self.assertIn('sfx_resume_exact(prefix_main,prefix_cut)', branch[:2400])
        legacy = section[section.index('refused before the "'):]
        self.assertLess(legacy.index('sfx_release_donor()'), legacy.index('sfx_failstop_if_running()'))
        owned = section.index('if(owned_batch_allocation&&admit_slot>=0)')
        exit_owned = section.index('goto owned_batch_request_done', owned)
        legacy_run = section.index('const auto result = strata::program::ServingSuffixRepairOperation::run(')
        self.assertLess(owned, exit_owned)
        self.assertLess(exit_owned, legacy_run)

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

    def test_bounded_profiles_require_managed_safe_unit_policy(self):
        eligibility = (ROOT / 'include/strata/core/native_owned_batch_serving.hpp').read_text()
        self.assertIn('p.prefill_chunk>0 && p.prefill_chunk<=32768', eligibility)
        for restriction in ['p.short_read==64', 'p.spec==0', 'p.batch_groups==1', '!p.vision']:
            self.assertIn(restriction, eligibility)
        self.assertIn('owned_batch_requested&&(o.prefill_chunk!=16||o.prefill_auto)', self.driver)
        self.assertIn('larger/auto owned foreground prefill requires explicit managed safe-unit policy and limits', self.driver)
        self.assertIn('serving_prefill_safe_turn', self.driver)
        self.assertIn('shared_prefix_refill(true', self.driver)
        generated = generate(ROOT)
        self.assertIn(body(self.cpp, 'WorkspaceAdmission Prefill::admit_owned_workspace('), generated)
        self.assertIn(body(self.driver, 'auto admit_prefill_workspace ='), generated)


if __name__ == '__main__':
    unittest.main()
