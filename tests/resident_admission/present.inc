bool present(const StateDigest& digest) {
    return std::any_of(digest.begin(),digest.end(),[](uint8_t b){return b!=0;});
}
