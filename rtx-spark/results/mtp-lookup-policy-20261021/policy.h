#pragma once

#include <cassert>
#include <cstdint>

struct spark_cost_policy {
    int64_t started_us = 0;
    double mtp_us_per_token = 0;
    double lookup_us_per_token = 0;
    uint32_t mtp_rounds = 0;
    uint32_t skipped_lookup = 0;
    uint32_t selected_lookup = 0;
    uint32_t bypassed_lookup = 0;

    bool use_lookup() {
        if (mtp_rounds < 4) {
            bypassed_lookup++;
            return false;
        }
        if (lookup_us_per_token == 0 || lookup_us_per_token <= mtp_us_per_token / 1.03 || ++skipped_lookup == 16) {
            skipped_lookup = 0;
            selected_lookup++;
            return true;
        }
        bypassed_lookup++;
        return false;
    }

    void observe(bool lookup, int64_t elapsed_us, uint32_t committed) {
        assert(elapsed_us > 0 && committed > 0);
        double & estimate = lookup ? lookup_us_per_token : mtp_us_per_token;
        const double sample = double(elapsed_us) / committed;
        estimate = estimate == 0 ? sample : 0.75 * estimate + 0.25 * sample;
        mtp_rounds += !lookup;
    }
};
