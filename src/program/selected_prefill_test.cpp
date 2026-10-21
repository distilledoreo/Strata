#include "strata/program/selected_prefill.hpp"
#include <cassert>
#include <cstdio>
int main() {
    using strata::program::selected_prefill_positions;
    std::vector<int32_t> table;
    std::string error;
    assert(selected_prefill_positions({0,2,4}, 3, 7, 3, 100, table, error));
    for (int i=0;i<3;++i) for (int axis=0;axis<3;++axis) assert(table[i*3+axis]==i*2);
    for (int i=3;i<12;++i) for (int axis=0;axis<3;++axis) assert(table[i*3+axis]==7+i-3);
    assert(selected_prefill_positions({0,1,2}, 3, 3, 3, 100, table, error));
    for (int i=0;i<100;++i) assert(table[i*3]==i);
    for (auto pos : std::vector<std::vector<int64_t>>{{}, {0,2}, {0,0,4}, {4,2,0}, {-1,2,4}, {0,2,7}})
        assert(!selected_prefill_positions(pos, 3, 7, 3, 100, table, error));
    assert(!selected_prefill_positions({0,2,4}, 3, 2, 3, 100, table, error));
    assert(!selected_prefill_positions({0,2,4}, 3, 7, 86, 100, table, error));
    assert(!selected_prefill_positions({0,2,4}, 3, 7, 0, 100, table, error));
    using strata::program::selected_prefix_limit;
    std::vector<int32_t> previous{0,0,0,32,32,32,64,64,64};
    assert(selected_prefix_limit(previous, previous, 3) == 3);
    auto changed = previous;
    changed[4] = 33;
    assert(selected_prefix_limit(previous, changed, 3) == 1);
    changed[0] = 1;
    assert(selected_prefix_limit(previous, changed, 3) == 0);
    assert(selected_prefix_limit({}, previous, 3) == 0);
    std::puts("selected_prefill_test OK");
}
