            // A terminal input failure forbids new units; draining an already
            // submitted pipeline group remains required and permitted.

            if(may_start && in_shared->input_error.load())return false;
