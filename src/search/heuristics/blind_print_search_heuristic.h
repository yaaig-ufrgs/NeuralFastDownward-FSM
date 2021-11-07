#ifndef HEURISTICS_BLIND_PRINT_SEARCH_HEURISTIC_H
#define HEURISTICS_BLIND_PRINT_SEARCH_HEURISTIC_H

#include "../heuristic.h"

namespace blind_print_search_heuristic {
class BlindPrintSearchHeuristic : public Heuristic {
    const std::vector<FactPair> relevant_facts;
    int min_operator_cost;
    const std::string prefix = "@";
protected:
    virtual int compute_heuristic(const State &ancestor_state) override;
public:
    BlindPrintSearchHeuristic(const options::Options &opts);
    ~BlindPrintSearchHeuristic();
};
}

#endif
