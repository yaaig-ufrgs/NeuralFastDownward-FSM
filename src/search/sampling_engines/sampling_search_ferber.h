#ifndef SEARCH_ENGINES_SAMPLING_SEARCH_FERBER_H
#define SEARCH_ENGINES_SAMPLING_SEARCH_FERBER_H

#include "sampling_search_base.h"

#include <vector>
#include <random>


/* SamplingSearchFerber
 * --------------------
 * Same as SamplingSearchSimple, but reduced and with
 * arguements for random_state, entire_plan, and init_state
 * sampling methods.
 */


namespace options {
class Options;
}

namespace sampling_engine {

enum SelectStateMethod {
    RANDOM_STATE,
    ENTIRE_PLAN,
    INIT_STATE,
};

class SamplingSearchFerber : public SamplingSearchBase {
protected:
    const SelectStateMethod select_state_method;
    const bool store_plan_cost;
    const bool store_state;
    const bool store_operator;
    const std::vector<FactPair> relevant_facts;
    const std::string header;

    std::string extract_single_sample(Trajectory trajectory, size_t idx_t, Plan plan, OperatorsProxy ops, int *cost);
    virtual std::vector<std::string> extract_samples() override;
    virtual std::string construct_header() const;
    virtual std::string sample_file_header() const override;

public:
    explicit SamplingSearchFerber(const options::Options &opts);
    virtual ~SamplingSearchFerber() override = default;
};
}
#endif
