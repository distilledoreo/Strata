// Experimental selected-token prefill: compact KV cells, original rotary positions.
#pragma once
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::program {
// A cached recurrent/KV prefix is reusable only while every rotary coordinate agrees.
inline int64_t selected_prefix_limit(const std::vector<int32_t>& previous,
        const std::vector<int32_t>& next, int64_t tokens) {
    int64_t matched = 0;
    const int64_t cells = (int64_t) std::min(previous.size(), next.size()) / 3;
    while (matched < tokens && matched < cells) {
        bool same = true;
        for (int axis = 0; axis < 3; ++axis)
            same = same && previous[(size_t) matched * 3 + axis] == next[(size_t) matched * 3 + axis];
        if (!same) break;
        ++matched;
    }
    return matched;
}
inline bool selected_prefill_positions(const std::vector<int64_t>& positions, int64_t tokens,
        int64_t original_length, int64_t max_new, int64_t context,
        std::vector<int32_t>& table, std::string& error) {
    if (tokens < 1 || (int64_t) positions.size() != tokens || original_length < tokens ||
        original_length > context || max_new < 1 || max_new > context - original_length - 8) {
        error = "selected prefill: lengths or original prompt plus output exceed context";
        return false;
    }
    int64_t previous = -1;
    for (auto p : positions) {
        if (p <= previous || p < 0 || p >= original_length) {
            error = "selected prefill: positions must be strictly increasing within original prompt";
            return false;
        }
        previous = p;
    }
    // Include the same speculative look-ahead cells as the existing vision table.
    table.resize((size_t) (context + 64) * 3);
    for (int64_t cell = 0; cell < context + 64; ++cell) {
        const auto pos = (int32_t) (cell < tokens ? positions[(size_t) cell]
                                                : original_length + cell - tokens);
        for (int axis = 0; axis < 3; ++axis) table[(size_t) cell * 3 + axis] = pos;
    }
    return true;
}
}  // namespace strata::program
