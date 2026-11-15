void VmmRange::release() { if(retirement_generation_){if(!retirement_complete_)retirement_failed("uncommitted sealed VMM range");return;}if(managed_)release_managed_metadata(); }
