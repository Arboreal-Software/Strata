    bool entry_set_logit_bias(std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
        ++effects; return true;
    }
    bool entry_init(std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
        ++effects; return true;
    }
    bool entry_record_window(std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
        ++effects; return true;
    }
    bool entry_capture(std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
        ++effects; return true;
    }
    bool entry_capture_commit(std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
        ++effects; return true;
    }
    bool entry_prelaunch(std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
        ++effects; return true;
    }
    bool entry_run(std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
        ++effects; return true;
    }
    bool entry_window_logprobs(std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
        ++effects; return true;
    }
    bool entry_commit(std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
        ++effects; return true;
    }
    bool entry_init_slots(std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
        ++effects; return true;
    }
    bool entry_capture_batch(std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
        ++effects; return true;
    }
    bool entry_capture_commit_batch(std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
        ++effects; return true;
    }
    bool entry_evict_batch_graph(std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
        ++effects; return true;
    }
    bool entry_instantiate_evicting(std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
        ++effects; return true;
    }
    bool entry_stage_batch(std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
        ++effects; return true;
    }
    bool entry_run_slots(std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
        ++effects; return true;
    }
    bool entry_run_slot_rows(std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
        ++effects; return true;
    }
    bool entry_commit_slots(std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
        ++effects; return true;
    }
    bool entry_commit_slot_prefixes(std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
        ++effects; return true;
    }
    bool entry_sample_rows(std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
        ++effects; return true;
    }
    bool entry_batch_launch(std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
        ++effects; return true;
    }
    int entry_batch_serve(std::string& err) {
    if (terminal_copy_query_failed_.load(std::memory_order_acquire)) return -1;
        ++effects; return 1;
    }
    int entry_batch_poll(std::string& err) {
    if (refuse_copy_query_failure(err)) return -1;
        ++effects; return 1;
    }
    bool entry_copy_logits(std::string& err) {
    if (terminal_copy_query_failed_.load(std::memory_order_acquire)) return false;
        ++effects; return true;
    }
    bool entry_capture_all(std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
        ++effects; return true;
    }
    bool entry_prestage(std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
        ++effects; return true;
    }
    bool entry_pl_launch(std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
        ++effects; return true;
    }
    int entry_service(std::string& err) {
    if (refuse_copy_query_failure(err)) return -1;
        ++effects; return 1;
    }
    bool entry_done(std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
        ++effects; return true;
    }
    bool entry_pl_finish(std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
        ++effects; return true;
    }
    bool entry_pl_commit_async(std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
        ++effects; return true;
    }
    bool check_entry_guards(std::string& err) {
        return entry_set_logit_bias(err) == false
            && entry_init(err) == false
            && entry_record_window(err) == false
            && entry_capture(err) == false
            && entry_capture_commit(err) == false
            && entry_prelaunch(err) == false
            && entry_run(err) == false
            && entry_window_logprobs(err) == false
            && entry_commit(err) == false
            && entry_init_slots(err) == false
            && entry_capture_batch(err) == false
            && entry_capture_commit_batch(err) == false
            && entry_evict_batch_graph(err) == false
            && entry_instantiate_evicting(err) == false
            && entry_stage_batch(err) == false
            && entry_run_slots(err) == false
            && entry_run_slot_rows(err) == false
            && entry_commit_slots(err) == false
            && entry_commit_slot_prefixes(err) == false
            && entry_sample_rows(err) == false
            && entry_batch_launch(err) == false
            && entry_batch_serve(err) == -1
            && entry_batch_poll(err) == -1
            && entry_copy_logits(err) == false
            && entry_capture_all(err) == false
            && entry_prestage(err) == false
            && entry_pl_launch(err) == false
            && entry_service(err) == -1
            && entry_done(err) == false
            && entry_pl_finish(err) == false
            && entry_pl_commit_async(err) == false;
    }
