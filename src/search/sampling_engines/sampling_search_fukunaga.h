#ifndef SEARCH_ENGINES_SAMPLING_SEARCH_FUKUNAGA_H
#define SEARCH_ENGINES_SAMPLING_SEARCH_FUKUNAGA_H

#include "sampling_search_base.h"

#include <vector>
#include <random>


/* SamplingSearchFukunaga
 * ----------------------
 * Similar to SamplingSearchFerber, but heuristics are not
 * estimated by a teacher search.
 */


namespace options {
class Options;
}


namespace sampling_engine {

std::mt19937 seeded_engine() {
    std::random_device rd; // Used to obtain a seed for the RNG
    return std::mt19937(rd()); // Standard mersenne_twister_engine seeded with rd()
}

enum SelectStateMethod {
    RANDOM_STATE,
    ENTIRE_PLAN,
    INIT_STATE,
};

class SamplingSearchFukunaga : public SamplingSearchBase {
protected:
    const SelectStateMethod select_state_method;
    const bool store_plan_cost;
    const bool store_state;
    const bool store_operator;
    const std::vector<FactPair> relevant_facts;
    const std::string header;

    std::mt19937 eng = seeded_engine();

    virtual std::vector<std::string> extract_samples() override;
    virtual std::string construct_header() const;
    virtual std::string sample_file_header() const override;

public:
    explicit SamplingSearchFukunaga(const options::Options &opts);
    virtual ~SamplingSearchFukunaga() override = default;
};
}
#endif
