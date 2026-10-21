#include "policy.h"

#include <cmath>
#include <cstdio>

#ifdef NDEBUG
#error Policy tests require assertions
#endif

int main() {
    spark_cost_policy policy;
    for (int i = 0; i < 4; ++i) {
        assert(!policy.use_lookup());
        policy.observe(false, 400, 4);
    }
    assert(policy.use_lookup());
    policy.observe(true, 1200, 13);
    assert(policy.use_lookup());
    assert(std::abs(policy.lookup_us_per_token - 1200.0 / 13) < 1e-9);
    policy.observe(true, 1000, 2);
    assert(std::abs(policy.lookup_us_per_token - (0.75 * 1200.0 / 13 + 125)) < 1e-9);
    for (int i = 0; i < 15; ++i) {
        assert(!policy.use_lookup());
    }
    assert(policy.use_lookup());
    assert(!policy.use_lookup());
    assert(policy.selected_lookup == 3 && policy.bypassed_lookup == 20);

    policy = {};
    assert(policy.started_us == 0 && policy.mtp_rounds == 0);
    assert(policy.lookup_us_per_token == 0 && policy.skipped_lookup == 0);
    assert(!policy.use_lookup());

    for (int i = 0; i < 4; ++i) {
        policy.observe(false, 412, 4);
    }
    policy.observe(true, 1300, 13);
    assert(policy.use_lookup()); // exactly 3% faster
    policy.observe(true, 1301, 13);
    assert(!policy.use_lookup());
    std::puts("Policy warmup, bootstrap, EWMA, probe cadence, reset and threshold checks passed");
}
