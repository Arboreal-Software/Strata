    DriverRuntimeObject<strata::kernels::cpu::ExpertPool> pool_storage(
#if defined(STRATA_WINDOW_STATE_AVAILABLE)
        owned_runtime?&owned_runtime->pool:nullptr
#else
        nullptr
#endif
        ,o.pool_workers,true,!o.no_host_worker,o.pool_affinity,o.pool_tasks,
#if defined(STRATA_WINDOW_STATE_AVAILABLE)
        bool(owned_runtime)
#else
        false
#endif
        );
