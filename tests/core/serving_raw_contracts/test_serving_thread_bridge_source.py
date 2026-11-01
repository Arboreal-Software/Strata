#!/usr/bin/env python3
"""Actual source-hook guards only; no compile, device or runtime qualification."""
import argparse
from pathlib import Path
import re
import unittest

ROOT=Path(__file__).resolve().parents[3]

def body(source,marker):
    start=source.index(marker)
    # Ignore comment/string braces while retaining original source positions.
    pattern=r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
    stripped=re.sub(pattern,lambda m:' '*len(m[0]),source,flags=re.S)
    opening=stripped.index('{',start);depth=0
    for i in range(opening,len(source)):
        if stripped[i]=='{':depth+=1
        elif stripped[i]=='}':
            depth-=1
            if not depth:return source[opening:i+1]
    raise AssertionError('unterminated actual source body: '+marker)

class ActualServingThreadSeams(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=(ROOT/'src/program/generate.cpp').read_text()
        cls.header=(ROOT/'src/program/serving_thread_bridge.hpp').read_text()
        cls.prefill=(ROOT/'src/prefill/prefill.cpp').read_text()

    def test_control_registry_never_runs_on_detached_reader(self):
        reader=body(self.source,'std::thread([st = in_shared]')
        self.assertIn('st->controls.push_back(l)',reader)
        for forbidden in ('serving_process_controls','priorities_.','dependencies_.','serving_threads->'):
            self.assertNotIn(forbidden,reader)
        process=body(self.source,'auto serving_process_controls=')
        self.assertIn('serving_threads->command(command,serving_thread_now_ms())',process)

    def test_ready_order_uses_actual_os_nonce(self):
        normal=self.source.index('std::printf("READY %lld stop')
        self.assertGreater(self.source.index('serving_threads->ready(stdout)',normal),normal)
        self.assertIn('scheduler_epoch=strata::program::serving_thread_epoch()',self.source)
        self.assertIn('ServingThreadBridge>(scheduler_epoch,',self.source)
        self.assertNotIn('ServingThreadBridge>(owned_runtime->remote_policy_incarnation',self.source)

    def test_managed_startup_and_bounded_input_do_not_activate_private_loader(self):
        marker='if(!serving_threads_config.valid || (serving_threads_config.enabled&&'
        start=self.source.index(marker);guard=self.source[start:self.source.index('if(serving_threads_config.enabled){',start)]
        self.assertNotIn('verified_inputs_active',guard)
        self.assertIn('!owned_batch_allocation',guard);self.assertIn('!model_budget',guard)
        start=self.source.index('size_t input_max_line_bytes=0, input_max_queue_lines=0;')
        end=self.source.index('const auto in_shared = std::make_shared<StdinShared>();',start)
        envelope=self.source[start:end]
        self.assertIn('if(serving_threads)',envelope)
        self.assertIn('if(!o.verified_inputs_active)serving_input_budget=model_budget.get()',envelope)
        self.assertIn('serving_input_budget->reserve',envelope)

    def test_first_setup_gated_by_bound_ready_operation(self):
        begin=self.source.index('bool serving_pending_request_completed=false;')
        gate=self.source.index('serving_threads->request_ready(slot,owned_admission_scope.request)',begin)
        first_native=self.source.index('if (!adapt_tick(true))',begin)
        self.assertLess(gate,first_native)
        self.assertIn('BADM %zu 0',self.source[gate:first_native])

    def test_idle_expiry_has_timed_wake_even_without_requests(self):
        wait=body(self.source,'auto next_line =')
        self.assertIn('serving_threads->next_expiry()',wait)
        self.assertIn('in_cv.wait_for',wait)
        self.assertIn('serving_process_controls()',wait)
        self.assertIn('if(!in_shared->controls.empty())continue',wait)

    def test_prefill_control_is_reactor_side_between_joined_exact_chunks(self):
        run=body(self.source,'auto run = [&](int64_t q,int64_t r)')
        self.assertLess(run.index('serving_process_controls()'),run.index('run_piece(at,end)'))
        self.assertLess(run.index('run_piece(at,end)'),run.index('cudaDeviceSynchronize()'))
        self.assertLess(run.index('cudaDeviceSynchronize()'),run.index('acknowledge_prefill('))
        self.assertGreater(run.rindex('serving_process_controls()'),run.index('acknowledge_prefill('))
        self.assertIn('stage_forwarded[i]=done',run)
        self.assertIn('on_stage_chunk=std::move(prior[i])',run)
        self.assertTrue(run.rstrip().endswith('return run_piece(q,r);\n                }'))
        actual=body(self.prefill,'bool Prefill::run(const int64_t* tokens')
        self.assertLess(actual.index('run_impl('),actual.index('drain_pipeline('))
        self.assertLess(actual.index('drain_pipeline('),actual.index('if (!body_ok)'))

    def test_original_whole_segment_plan_flags_survive_splitting(self):
        plan=self.source.index('if(serving_threads&&plan.chunk<=0)')
        end=self.source.index('const bool planned = plan.chunk > 0;',plan)
        self.assertIn('plan.single=b0-a0<=plan.chunk;plan.chunked=false;',self.source[plan:end])
        self.assertIn('event.chunk=plan.chunk;',body(self.source,'auto run_piece ='))

    def test_foreground_only_path_exists_while_main_fence_open(self):
        batch=body(self.source,'auto batch_step =')
        self.assertIn('!in_prompt_read&&!managed_main_fence&&owned_slot_maintenance()',batch)
        self.assertIn('(in_prompt_read||managed_main_fence))serving_foreground_only_cohort()',batch)
        peer=body(self.source,'auto serving_foreground_only_cohort=')
        self.assertIn('foreground_peer(selected,units[i])',peer)
        self.assertNotIn('repair_unit(',peer)
        self.assertNotIn('AdmissionUnit::repair',peer)

    def test_cohort_filter_and_each_actual_member_ack(self):
        batch=body(self.source,'auto batch_step =')
        self.assertIn('!serving_thread_cohort[size_t(b)]',batch)
        self.assertNotIn('serving_selected_thread_unit->slot!=size_t(b)',batch)
        self.assertGreaterEqual(batch.count('serving_ack_foreground_lane(size_t(b))'),3)
        ack=body(self.source,'auto serving_ack_foreground_lane=')
        self.assertIn('*serving_thread_cohort[slot]',ack)
        self.assertIn('lane.owned_raw->slot_occupancy_incarnation(),lane.owned_raw->cut().id',ack)

    def test_alignment_uses_actual_occupancy_before_first_known_row(self):
        fresh=body(self.source,'auto fresh_owned_slot =')
        self.assertLess(fresh.index('open_request(error)'),fresh.index('serving_threads->attach('))
        admission=body(self.source,'auto admit_owned_repair_slot =')
        self.assertLess(admission.index('fresh_owned_slot(index,error)'),admission.index('owned_alignment_step()'))
        self.assertIn('serving_ack_foreground_lane(size_t(b))',body(self.source,'auto batch_step ='))

    def test_cancelled_graph_does_not_release_native_owner_by_metadata(self):
        stop=body(self.source,'auto accept_batch_stop =')
        self.assertLess(stop.index('retire_owned_slot(err)'),stop.index('serving_release_thread_lane(slot,true)'))
        batch=body(self.source,'auto batch_step =')
        self.assertIn('cancellation_pending(i)',batch)
        self.assertIn('accept_batch_stop("BSTOP "+std::to_string(i)',batch)
        dispatch=body(self.header,'ServingThreadReply dispatch(')
        self.assertNotIn('bindings_[slot]={};',dispatch)

    def test_prefill_bootstrap_is_explicit_and_not_fabricated_statecut(self):
        run=body(self.source,'auto run = [&](int64_t q,int64_t r)')
        self.assertIn('proof.source_cut=serving_raw_session->cut().id',run)
        self.assertIn('proof.source_bootstrap=proof.source_cut==0&&serving_raw_cut==0&&serving_ordinary_fence&&serving_start_zeroed',run)
        self.assertIn('proof.request_fence_open=',run)
        self.assertNotIn('proof.source_cut=1',run)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path)
    args,rest=parser.parse_known_args()
    if args.source_root:ROOT=args.source_root.resolve()
    unittest.main(argv=[__file__,*rest])
