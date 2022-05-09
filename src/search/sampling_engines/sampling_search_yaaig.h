#ifndef SEARCH_ENGINES_SAMPLING_SEARCH_YAAIG_H
#define SEARCH_ENGINES_SAMPLING_SEARCH_YAAIG_H

#include "sampling_search_base.h"

#include <vector>
#include <random>
#include <limits.h>

#include <unordered_map>
#include <unordered_set>

#include "../trie/trie.h"

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
    const std::string minimization;
    const int assignments_by_undefined_state;
    const int contrasting_samples;
    const int avi_k;
    const int avi_its;
    const std::string avi_rule;
    const double avi_epsilon;
    const bool avi_unit_cost;
    const bool sort_h;
    const std::string mse_hstar_file;
    const std::string mse_result_file;
    const std::shared_ptr<Evaluator> evaluator;
    const bool use_evaluator;
    const std::vector<FactPair> relevant_facts;
    StateRegistry registry;
    const std::string header;
    std::shared_ptr<utils::RandomNumberGenerator> rng;
    trie::trie<int> trie_statespace; // with h*

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
    void approximate_value_iteration();
    void old_approximate_value_iteration(
        std::vector<std::pair<int,std::pair<std::vector<int>,std::string>>> sample_pairs =
            std::vector<std::pair<int,std::pair<std::vector<int>,std::string>>>()
    );
    void do_minimization(std::vector<std::shared_ptr<PartialAssignment>>& states);
    void do_minimization(std::vector<std::pair<int,std::pair<std::vector<int>,std::string>>>& states);
    std::vector<State> assign_undefined_state(std::shared_ptr<PartialAssignment>& pa, int max_attempts);
    void create_contrasting_samples(
        std::vector<std::pair<int,std::pair<std::vector<int>,std::string>>>& values_set, int percentage);
    std::vector<std::string> values_to_samples(
        std::vector<std::pair<int,std::pair<std::vector<int>,std::string>>> values_set);
    void replace_h_with_evaluator(
        std::vector<std::pair<int,std::pair<std::vector<int>,std::string>>>& values_set);
    void compute_sampling_statistics(std::vector<std::pair<int,std::pair<std::vector<int>,std::string>>> samples);
    std::vector<int> binary_to_values(std::string bin);
};

class AviNode {
public:
    std::vector<std::shared_ptr<PartialAssignment>> samples;
    std::vector<std::pair<std::string,int>> predecessors;
    std::vector<std::pair<std::string,int>> successors;
    int best_h = INT_MAX;

    bool operator()(AviNode& a, AviNode& b) {
        return a.best_h < b.best_h;
    }
};
}
#endif
