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

class SamplingSearchFukunaga : public SamplingSearchBase {
protected:
    const bool store_plan_cost;
    const bool store_state;
    const bool use_full_state;
    const bool match_heuristics;
    const std::vector<FactPair> relevant_facts;
    const std::string header;

    virtual std::vector<std::string> extract_samples() override;
    virtual std::string construct_header() const;
    virtual std::string sample_file_header() const override;

public:
    explicit SamplingSearchFukunaga(const options::Options &opts);
    virtual ~SamplingSearchFukunaga() override = default;
};
}
#endif
