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
    const bool sai_partial;
    const bool sai_complete;
    const bool sui;
    const SearchRule sui_rule;
    const std::string statespace_file;
    const std::string random_value;
    const int random_multiplier;
    const std::shared_ptr<Evaluator> evaluator;
    const std::string evaluate_file;
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
    std::vector<std::string> format_output(std::vector<std::shared_ptr<PartialAssignment>>& samples);
    std::vector<int> binary_to_values(std::string bin);
    void create_trie_statespace();
    void successor_improvement(std::vector<std::shared_ptr<PartialAssignment>>& samples);
    void sample_improvement(std::vector<std::shared_ptr<PartialAssignment>>& samples);
    void replace_h_with_evaluator(std::vector<std::shared_ptr<PartialAssignment>>& samples);
    void create_random_samples(
          std::vector<std::shared_ptr<PartialAssignment>>& samples, int num_random_samples);
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
