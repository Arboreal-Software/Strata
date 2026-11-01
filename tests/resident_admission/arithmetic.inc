SuffixRawResult invalid(const char* why) { return {SuffixRawStatus::invalid,why,{}}; }
bool add(uint64_t& a,uint64_t b) {
    if(b>UINT64_MAX-a) return false;
    a+=b;return true;
}
bool product(uint64_t a,uint64_t b,uint64_t& result) {
    if(a && b>UINT64_MAX/a) return false;
    result=a*b;return true;
}
