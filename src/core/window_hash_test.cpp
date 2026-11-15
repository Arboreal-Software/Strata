#include "strata/core/window_hash.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace strata::core;
namespace {
std::string hex(const StateDigest& bytes) {
    std::string s;
    for (auto b : bytes) { s += "0123456789abcdef"[b>>4]; s += "0123456789abcdef"[b&15]; }
    return s;
}
void check(bool ok,const char* what) {
    if (!ok) { std::fprintf(stderr,"FAIL: %s\n",what); std::exit(1); }
}
}
int main() {
    WindowHasher h;
    check(hex(h.digest())=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","empty SHA-256 vector");
    h.update("abc",3);
    check(hex(h.digest())=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad","abc SHA-256 vector");
    h.update("defghijklmnopqrstuvwxyz",23);
    check(hex(h.digest())=="71c480df93d6ae2f1efad1447c66c9525e316218cf51fc8d9ed832f2daf18b73","digest does not consume streaming state");
    WindowHasher million;
    const std::string chunk(1000,'a');
    for(int i=0;i<1000;++i) million.update(chunk.data(),chunk.size());
    check(hex(million.digest())=="cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0","million-a SHA-256 vector");
    for (size_t n : {55,56,63,64,65,1000}) {
        WindowHasher bulk,pieces;
        std::string text(n,'a'); bulk.update(text.data(),text.size());
        for (char c : text) pieces.update(&c,1);
        check(bulk.digest()==pieces.digest(),"stream splits preserve block/padding boundaries");
    }
    check(window_history_digest({7,8,9})!=window_history_digest({7,9,8}),"history order bound");
    check(window_history_digest({7,8,9})!=window_history_digest({7,8,9,0}),"history length bound");
    // Independently generated with Python hashlib and explicit little-endian struct packing.
    check(hex(window_history_digest({7,8,9}))=="de54a67f3bf952a7ef3beb4386277672b1bf5ef880da9025dac89af5a94d6c9b",
          "history schema agrees with independent SHA-256 implementation");
    std::puts("window_hash_test passed");
}
