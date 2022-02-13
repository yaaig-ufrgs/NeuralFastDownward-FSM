#ifndef SEARCH_ENGINES_SAMPLING_SEARCH_YAAIG_H
#define SEARCH_ENGINES_SAMPLING_SEARCH_YAAIG_H

#include "sampling_search_base.h"

#include <vector>
#include <random>

#include <unordered_map>
#include <unordered_set>

/* SamplingSearchYaaig
 * ----------------------
 * Similar to SamplingSearchFerber, but heuristics are not
 * estimated by a teacher search.
 */

namespace options {
class Options;
}

namespace sampling_engine {

class SamplingSearchYaaig : public SamplingSearchBase {
protected:
    const bool store_plan_cost;
    const bool store_state;
    const std::string state_representation;
    const bool minimization;
    const int assignments_by_undefined_state;
    const int contrasting_samples;
    const int avi_k;
    const int avi_its;
    const std::vector<FactPair> relevant_facts;
    const std::string header;
    std::shared_ptr<utils::RandomNumberGenerator> rng;

    virtual std::vector<std::string> extract_samples() override;
    virtual std::string construct_header() const;
    virtual std::string sample_file_header() const override;


public:
    explicit SamplingSearchYaaig(const options::Options &opts);
    virtual ~SamplingSearchYaaig() override = default;

private:
    void approximate_value_iteration();
    std::unordered_map<std::string,int> do_minimization(std::unordered_map<std::string,int>& state_value);
    std::vector<State> assign_undefined_state(std::shared_ptr<PartialAssignment>& pa, int max_attempts);
    void create_contrasting_samples(std::vector<std::pair<int,std::vector<int>>>& values_set, int percentage);
    std::vector<std::string> values_to_samples(std::vector<std::pair<int,std::vector<int>>> values_set);
};
}
#endif
