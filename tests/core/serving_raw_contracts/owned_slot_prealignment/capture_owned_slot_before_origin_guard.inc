SuffixRawResult ServingRawSession::capture_owned_slot(BundleHandle& out) {
    if(alignment_known_||slot_controller_bound_)return invalid("slot capture requires no pending preparation and its sole metadata authority");
    if(!slot_occupancy_||!slot_occupancy_->active())return invalid("no live owned slot occupancy");
    const auto metadata=endpoint_metadata(slot_current_key_,tx_->cut_,slot_tokens_);
    // Freeze edit-origin metadata before capture/native mutation. This is an actual
    // progressed donor endpoint, never a relabelled original source image.
    std::unique_ptr<WindowKey> edit_origin;
    if(slot_current_key_!=tx_->identity_&&!alignment_edit_origin_) {
        if(alignment_known_||tx_->cut_.forwarded%4)return invalid("alignment donor not at real page endpoint");
        try{edit_origin=std::make_unique<WindowKey>(slot_current_key_);}catch(...){return invalid("alignment key allocation refused");}
    }
    auto result=capture(metadata,out);
    if(result&&edit_origin){alignment_edit_origin_=std::move(edit_origin);alignment_donor_=out;}
    return result;
}
