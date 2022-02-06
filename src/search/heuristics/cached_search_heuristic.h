#ifndef HEURISTICS_CACHED_SEARCH_HEURISTIC_H
#define HEURISTICS_CACHED_SEARCH_HEURISTIC_H

#include <unordered_map>
#include <string>

#include "../heuristic.h"

namespace cached_search_heuristic {
class CachedSearchHeuristic : public Heuristic {
    std::unordered_map<std::string,int> h;
    int max_value;
protected:
    virtual int compute_heuristic(const State &ancestor_state) override;
public:
    CachedSearchHeuristic(const options::Options &opts);
    ~CachedSearchHeuristic();
};
}

#endif
