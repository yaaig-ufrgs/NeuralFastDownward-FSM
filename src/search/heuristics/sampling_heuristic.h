#ifndef HEURISTICS_SAMPLING_HEURISTIC_H
#define HEURISTICS_SAMPLING_HEURISTIC_H

#include <unordered_map>
#include <random>

#include "../heuristic.h"
#include "../utils/rng.h"

using namespace std;

namespace sampling_heuristic {
class SamplingHeuristic : public Heuristic {
    const std::vector<FactPair> relevant_facts;
    unordered_map<string,int> h;
    utils::RandomNumberGenerator rng;
protected:
    virtual int compute_heuristic(const State &ancestor_state) override;
public:
    SamplingHeuristic(const options::Options &opts);
    ~SamplingHeuristic();
};
}

#endif
