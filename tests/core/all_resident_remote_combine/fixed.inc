        // The zero-doorbell graph never resets/populates remote sums.
        if (remote_opt_ && !ar_on()) remote_opt_->combine(bo_ + tb * N, y_dummy_ + tb * N, tb, n,
                              device_plan_ ? skip_ + grp : nullptr, ring, cs);
