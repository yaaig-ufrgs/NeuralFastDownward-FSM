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
        for (unsigned i = 0; i < relevant_facts.size(); i++) {
            if ((state_representation == "undefined") &&
                (i == 0 || relevant_facts[i].var != relevant_facts[i-1].var))
                oss << "Atom undefined()" << state_separator;
            oss << task->get_fact_name(relevant_facts[i]) << state_separator;
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
    for (std::shared_ptr<PartialAssignment>& partialAssignment: sampling_technique::modified_tasks) {
        ostringstream oss;

        if (store_plan_cost) {
            oss << partialAssignment->estimated_heuristic << field_separator;
        }

        if (store_state) {
            vector<int> values;
            if (state_representation == "complete") {
                State s = partialAssignment->get_full_state(true, *rng).second;
                s.unpack();
                values = s.get_values();
            } else if (state_representation == "partial" || state_representation == "undefined") {
                values = partialAssignment->get_values();
            }
            for (unsigned i = 0; i < relevant_facts.size(); i++) {
                if ((state_representation == "undefined") &&
                    (i == 0 || relevant_facts[i].var != relevant_facts[i-1].var))
                    oss << (values[relevant_facts[i].var] == PartialAssignment::UNASSIGNED);
                oss << (values[relevant_facts[i].var] == relevant_facts[i].value ? 1 : 0);
            }
            oss << field_separator;
        }

        string s = oss.str();
        s.pop_back();

        samples.push_back(s);
    }
    if (match_heuristics) {
        unordered_map<string,int> pairs;
        string state;
        int h, cut;
        for (string& s : samples) {
            cut = s.find_first_of(";");
            h = stoi(s.substr(0, cut));
            string state = s.substr(cut+1);
            if (pairs.count(state) == 0 || h < pairs[state])
                pairs[state] = h;
        }
        for (string& s : samples) {
            cut = s.find_first_of(";");
            string state = s.substr(cut+1);
            s = to_string(pairs[state]) + ";" + state;
        }
    }
    return samples;
}

SamplingSearchFukunaga::SamplingSearchFukunaga(const options::Options &opts)
    : SamplingSearchBase(opts),
      store_plan_cost(opts.get<bool>("store_plan_cost")),
      store_state(opts.get<bool>("store_state")),
      state_representation(opts.get<string>("state_representation")),
      match_heuristics(opts.get<bool>("match_heuristics")),
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
    parser.add_option<string>(
            "state_representation",
            "State facts representation format (complete, partial, or undefined).",
            "complete");
    parser.add_option<bool>(
            "match_heuristics",
            "Identical states receive the best heuristic value assigned between them.",
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
