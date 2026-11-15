// src/kernels/cpu/hadamard.cpp - see include/strata/kernels/cpu/hadamard.hpp.
#include "strata/kernels/cpu/hadamard.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <atomic>
#include <stdexcept>
#include <charconv>

namespace strata::kernels::cpu {
namespace {
HadamardSpec g_spec;
std::atomic<uint64_t> spec_generation{0};
}

const HadamardSpec& hadamard_spec() { return g_spec; }
uint64_t hadamard_generation() noexcept {return spec_generation.load(std::memory_order_acquire);}
void hadamard_set(HadamardSpec h) { spec_generation.fetch_add(1,std::memory_order_acq_rel);g_spec = std::move(h); }

const float* hadamard_signs(int width) {
    const auto it = g_spec.signs.find(width);
    return it == g_spec.signs.end() ? nullptr : it->second.data();
}

bool hadamard_parse(const std::string& path, int64_t n_layers, HadamardSpec& h, std::string& err,const HadamardParseLimits* limits) {
    try {
    HadamardSpec candidate;auto& target=limits?candidate:h;
    if(limits&&(!limits->max_layers||limits->max_layers>65536||!limits->max_line_bytes||limits->max_line_bytes>(16ull<<20)||!limits->max_total_bytes||limits->max_total_bytes>(64ull<<20)||!limits->max_vectors||limits->max_vectors>1024||!limits->max_elements||limits->max_elements>(4ull<<20)||n_layers<=0||uint64_t(n_layers)>limits->max_layers)){err="Hadamard layer bound refused";return false;}
    uint64_t used=0,elements=0;
    std::ifstream in(path);
    if (!in) { err = "cannot read " + path; return false; }
    target = HadamardSpec{};
    target.roles.assign((size_t) n_layers, 0);
    std::vector<uint8_t> seen((size_t) n_layers, 0);
    std::string line;
    bool versioned = false;
    auto next_line=[&](){
        if(!limits)return bool(std::getline(in,line));
        line.clear();char c;
        while(in.get(c)){if(++used>limits->max_total_bytes)throw std::length_error("Hadamard metadata byte limit");if(c=='\n')return true;if(line.size()>=limits->max_line_bytes)throw std::length_error("Hadamard line limit");line.push_back(c);}
        if(in.bad())throw std::runtime_error("Hadamard metadata read failed");
        return !line.empty();
    };
    while (next_line()) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            static const char tag[] = "# strata hadamard v";
            if (line.compare(0, sizeof tag - 1, tag) == 0) {
                int version=0;
                if(limits){const char* first=line.data()+sizeof tag-1;auto parsed=std::from_chars(first,line.data()+line.size(),version);if(parsed.ec!=std::errc{})version=0;}
                else version=std::atoi(line.c_str()+sizeof tag-1);
                if (version != 1) {
                    err = path + ": an unknown hadamard.txt version (this engine reads v1)";
                    return false;
                }
                versioned = true;
            }
            continue;
        }
        std::istringstream ss(line);
        std::string key;
        ss >> key;
        if (key == "block") {
            long long b = 0;
            if (!(ss >> b) || b < 2 || b > 4096 || (b & (b - 1)) != 0 || target.block) {
                err = path + ": a bad or repeated block line: " + line;
                return false;
            }
            target.block = (int) b;
        } else if (key == "signs") {
            long long w = 0;
            if (!(ss >> w) || w <= 0 || w > (1 << 20) || target.signs.count((int) w)) {
                err = path + ": a bad or repeated signs line";
                return false;
            }
            if(limits&&(target.signs.size()>=limits->max_vectors||uint64_t(w)>limits->max_elements-elements)){err="Hadamard sign allocation bound refused";return false;}
            elements+=uint64_t(w);
            std::vector<float> v((size_t) w);
            for (auto& x : v) {
                int s = 0;
                if (!(ss >> s) || (s != 1 && s != -1)) { err = path + ": sign values must be +1 or -1"; return false; }
                x = (float) s;
            }
            std::string extra;
            if (ss >> extra) { err = path + ": a signs line longer than its width"; return false; }
            target.signs.emplace((int) w, std::move(v));
        } else if (key == "layer") {
            long long l = -1;
            int g = -1, u = -1, d = -1;
            if (!(ss >> l >> g >> u >> d) || l < 0 || l >= n_layers || (g | u | d) & ~1 || seen[(size_t) l]) {
                err = path + ": a bad or repeated layer line: " + line;
                return false;
            }
            if (g != u) {   // gate and up read one activation: a transform for one of them only is not runnable
                err = path + ": layer " + std::to_string(l) + " folds only one of gate/up";
                return false;
            }
            seen[(size_t) l] = 1;
            target.roles[(size_t) l] = (uint8_t) (g | (u << 1) | (d << 2));
        } else {
            err = path + ": an unknown line: " + line;
            return false;
        }
    }
    if (!versioned || target.block == 0) { err = path + ": no version header or no block line"; return false; }
    for (const auto& [w, v] : target.signs)
        if (w % target.block) { err = path + ": sign width " + std::to_string(w) + " is not whole blocks"; return false; }
    if(limits)h=std::move(candidate);
    return true;
    }catch(const std::exception&){if(!limits)throw;err.clear();return false;}
}

void hadamard_rotate(float* x, int n, int block, const float* signs) {
    if (signs)
        for (int i = 0; i < n; ++i) x[i] *= signs[i];
    const float scale = 1.0f / std::sqrt((float) block);
    for (int b0 = 0; b0 < n; b0 += block) {
        float* v = x + b0;
        // the in-place fast Walsh-Hadamard transform in natural (Sylvester) order: entry (-1)^popcount(r & c)
        for (int len = 1; len < block; len <<= 1)
            for (int i = 0; i < block; i += len << 1)
                for (int j = i; j < i + len; ++j) {
                    const float a = v[j], c = v[j + len];
                    v[j] = a + c;
                    v[j + len] = a - c;
                }
        for (int i = 0; i < block; ++i) v[i] *= scale;
    }
}

}  // namespace strata::kernels::cpu
