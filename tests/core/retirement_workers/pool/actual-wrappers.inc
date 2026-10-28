namespace strata::kernels::cpu {
void ExpertPool::run(ExpertJob* jobs, int n) {
    DispatchGuard guard(*this);
    run_impl(jobs, n);
    guard.complete = true;
}
void ExpertPool::run_split(ExpertJob* jobs, int n) {
    DispatchGuard guard(*this);
    run_split_impl(jobs, n);
    guard.complete = true;
}
void ExpertPool::run_split_multi(ExpertJobMulti* jobs, int n) {
    DispatchGuard guard(*this);
    run_split_multi_impl(jobs, n);
    guard.complete = true;
}
void ExpertPool::run_split_multi_native(const NativeFmt& f, ExpertJobMulti* jobs, int n) {
    DispatchGuard guard(*this);
    run_split_multi_native_impl(f, jobs, n);
    guard.complete = true;
}
} // namespace strata::kernels::cpu
