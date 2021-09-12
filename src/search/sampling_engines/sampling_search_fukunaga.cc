#include "sampling_search_fukunaga.h"

#include "sampling_search_base.h"
#include "sampling_engine.h"

#include "../option_parser.h"
#include "../plugin.h"

#include "../task_utils/task_properties.h"

#include <sstream>
#include <string>

using namespace std;

namespace sampling_engine {

string SamplingSearchFukunaga::construct_header() const {
    ostringstream oss;

    if (store_plan_cost){
        oss << "#<PlanCost>=single integer value" << endl;
    }
    if (store_state) {
        oss << "#<State>=";
        for (const FactPair &fp: relevant_facts) {
            oss << task->get_fact_name(fp) << state_separator;
        }
        oss.seekp(-1,oss.cur);
    }

    return oss.str();
}

string SamplingSearchFukunaga::sample_file_header() const {
    return header;
}

vector<string> SamplingSearchFukunaga::extract_samples() {
    vector<string> samples;
    for (std::shared_ptr<PartialAssignment>& task: sampling_technique::modified_tasks) {
        ostringstream oss;

        if (store_plan_cost) {
            oss << task->estimated_heuristic << field_separator;
        }

        if (store_state) {
            vector<int> values;
            if (use_full_state) {
                State s = task->get_full_state(true, *rng).second;
                s.unpack();
                values = s.get_values();
            } else {
                values = task->get_values();
            }
            for (const FactPair &fp: relevant_facts) {
                // if (values[fp.var] == fp.value)
                    // oss << this->task->get_fact_name(fp) << state_separator;
                oss << (values[fp.var] == fp.value ? 1 : 0);
            }

            oss.seekp(-1, oss.cur);
            oss << field_separator;
        }

        string s = oss.str();
        s.pop_back();

        samples.push_back(s);
    }
    return samples;
}

SamplingSearchFukunaga::SamplingSearchFukunaga(const options::Options &opts)
    : SamplingSearchBase(opts),
      store_plan_cost(opts.get<bool>("store_plan_cost")),
      store_state(opts.get<bool>("store_state")),
      use_full_state(opts.get<bool>("use_full_state")),
      relevant_facts(task_properties::get_strips_fact_pairs(task.get())),
      header(construct_header()){
}


static shared_ptr<SearchEngine> _parse_sampling_search_fukunaga(OptionParser &parser) {
    parser.document_synopsis("Sampling Search Manager", "");

    sampling_engine::SamplingSearchBase::add_sampling_search_base_options(parser);
    sampling_engine::SamplingEngine::add_sampling_options(parser);
    sampling_engine::SamplingStateEngine::add_sampling_state_options(
            parser, "fields", "pddl", ";", ";");

    parser.add_option<bool>(
            "store_plan_cost",
            "Store for every state its cost along the plan to the goal",
            "true");
    parser.add_option<bool>(
            "store_state",
            "Store every state along the plan",
            "true");
    parser.add_option<bool>(
            "use_full_state",
            "Transform partial assignment to full state.",
            "false");

    SearchEngine::add_options_to_parser(parser);
    Options opts = parser.parse();
    shared_ptr<sampling_engine::SamplingSearchFukunaga> engine;
    if (!parser.dry_run()) {
        engine = make_shared<sampling_engine::SamplingSearchFukunaga>(opts);
    }

    return engine;
}

static Plugin<SearchEngine> _plugin_search("sampling_search_fukunaga", _parse_sampling_search_fukunaga);

}
