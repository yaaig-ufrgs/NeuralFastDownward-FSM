#ifndef SEARCH_ENGINES_SAMPLING_SEARCH_YAAIG_H
#define SEARCH_ENGINES_SAMPLING_SEARCH_YAAIG_H

#include "sampling_search_base.h"

#include <vector>
#include <random>
#include <limits.h>

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
    const std::string random_sample_state_representation;
    const std::string sai;
    const int assignments_by_undefined_state;
    const int sui_k;
    const SearchRule sui_rule;
    const double sui_epsilon;
    const bool sort_h;
    const std::string mse_hstar_file;
    const std::string mse_result_file;
    const std::shared_ptr<Evaluator> evaluator;
    const bool use_evaluator;
    const std::vector<FactPair> relevant_facts;
    StateRegistry registry;
    const std::string header;
    std::shared_ptr<utils::RandomNumberGenerator> rng;

    virtual std::vector<std::string> extract_samples() override;
    virtual std::string construct_header() const;
    virtual std::string sample_file_header() const override;


public:
    explicit SamplingSearchYaaig(const options::Options &opts);
    virtual ~SamplingSearchYaaig() override = default;

private:
    void create_trie_statespace();
    double mse(std::vector<std::shared_ptr<PartialAssignment>>& samples, bool root = false);
    void log_mse(int updates);
    void successor_improvement();
    void sample_improvement(std::vector<std::shared_ptr<PartialAssignment>>& states);
    //void sample_improvement(std::vector<std::pair<int,std::pair<std::vector<int>,std::string>>>& states);
    void sample_improvement(std::vector<std::pair<int,std::string>>& states);
    std::vector<State> assign_undefined_state(std::shared_ptr<PartialAssignment>& pa, int max_attempts);
    void create_random_samples(
          std::vector<std::pair<int,std::pair<std::vector<int>,std::string>>>& values_set_eval, std::vector<std::pair<int,std::string>>& values_set, int percentage);
    std::vector<std::string> values_to_samples(
        std::vector<std::pair<int,std::pair<std::vector<int>,std::string>>> values_set_eval, std::vector<std::pair<int,std::string>> values_set);
    void replace_h_with_evaluator(
        std::vector<std::pair<int,std::pair<std::vector<int>,std::string>>>& values_set);
    void compute_sampling_statistics(std::vector<std::pair<int,std::pair<std::vector<int>,std::string>>> samples_eval, std::vector<std::pair<int,std::string>> samples);
    std::vector<int> binary_to_values(std::string bin);
};

class SuiNode {
public:
    std::vector<std::shared_ptr<PartialAssignment>> samples;
    std::vector<std::pair<size_t,int>> successors;
    int best_h = INT_MAX;
};

class SuiNodePtrCompare {
public:
    bool operator()(SuiNode* const first, SuiNode* const second) {
        return first->best_h > second->best_h;
    };
};
}
#endif
