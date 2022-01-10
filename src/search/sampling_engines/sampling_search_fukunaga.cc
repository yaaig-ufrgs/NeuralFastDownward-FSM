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
            if ((state_representation == "undefined") && (i == 0 || relevant_facts[i].var != relevant_facts[i-1].var))
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

unordered_map<string,int> SamplingSearchFukunaga::create_smaller_h_mapping() {
    // If match_heuristics then then all identical samples receive the smallest h value among them
    unordered_map<string,int> state_value;
    for (shared_ptr<PartialAssignment>& partialAssignment: sampling_technique::modified_tasks) {
        string state = partialAssignment->to_string();
        int h = partialAssignment->estimated_heuristic;
        if (state_value.count(state) == 0 || h < state_value[state])
            state_value[state] = h;
    }
    return state_value;
}

vector<State> SamplingSearchFukunaga::assign_undefined_state(shared_ptr<PartialAssignment>& pa, int max_attempts) {
    // Each partial state generates `assignments_by_undefined_state` full states
    vector<State> states;
    static int rand_value = 0;
    for (int attempts = 0; attempts < max_attempts; attempts++) {
        utils::RandomNumberGenerator rand(rand_value++);
        pair<bool,State> full_state = pa->get_full_state(true, rand);
        if (!full_state.first)
            continue;
        State s = full_state.second;
        if (count(states.begin(), states.end(), s) == 0) {
            states.push_back(s);
            if (states.size() >= (unsigned)assignments_by_undefined_state)
                break;
            attempts = 0;
        }
    }
    return states;
}

void SamplingSearchFukunaga::create_contrasting_samples(
    vector<pair<int,vector<int>>>& values_set,
    int percentage
) {
    if (percentage == 0)
        return;
    assert(percentage > 0 && percentage <= 100);

    const size_t n_atoms = sampling_technique::modified_tasks[0]->get_values().size();
    PartialAssignment pa(
        *sampling_technique::modified_tasks[0],
        vector<int>(n_atoms, PartialAssignment::UNASSIGNED)
    );

    // Biggest h found in the search
    int max_h = 0;
    for (auto& p : values_set)
        if (p.first > max_h)
            max_h = p.first;
    max_h++; // contrasting h = max_h + 1

    int samples_to_be_created;
    if (percentage == 100) {
        // TODO: if 100%, the samples generation step is useless
        samples_to_be_created = values_set.size();
        values_set.clear();
    } else {
        samples_to_be_created =
            (sampling_technique::modified_tasks.size()*percentage) / (100.0 - percentage);
    }

    unordered_map<string,int> state_value;
    if (match_heuristics) {
        for (auto& p : values_set) {
            string s; // values to string
            for (int& v : p.second)
                s += '0' + v;
            // All identical states have the same h when match_heuristics=true,
            // so we just add in the first occurrence
            if (state_value.count(s) == 0)
                state_value[s] = p.first;
        }
    }

    while (samples_to_be_created > 0) {
        pair<bool,State> fs = pa.get_full_state(true, *rng);
        if (!fs.first)
            continue;
        State s = fs.second;
        s.unpack();
        vector<int> values = s.get_values();

        int h = max_h;
        if (match_heuristics) {
            string s;
            for (int& v : values)
                s += '0' + v;
            if (state_value.count(s) != 0)
                h = state_value[s];   
        }
        
        values_set.push_back(make_pair(h, values));
        samples_to_be_created--;
    }
}

vector<string> SamplingSearchFukunaga::values_to_samples(vector<pair<int,vector<int>>> values_set) {
    vector<string> samples;
    for (pair<int,vector<int>>& p : values_set) {
        ostringstream oss;
        if (store_plan_cost)
            oss << p.first << field_separator;
        if (store_state) {
            for (unsigned i = 0; i < relevant_facts.size(); i++) {
                if ((state_representation == "undefined") &&
                    (i == 0 || relevant_facts[i].var != relevant_facts[i-1].var))
                    oss << (p.second[relevant_facts[i].var] == PartialAssignment::UNASSIGNED);
                oss << (p.second[relevant_facts[i].var] == relevant_facts[i].value ? 1 : 0);
            }
        }
        samples.push_back(oss.str());
    }
    return samples;
}

vector<string> SamplingSearchFukunaga::extract_samples() {
    unordered_map<string,int> state_value;
    if (match_heuristics)
        state_value = create_smaller_h_mapping();

    vector<pair<int,vector<int>>> values_set;
    for (shared_ptr<PartialAssignment>& partialAssignment: sampling_technique::modified_tasks) {
        int h = -1;
        if (store_plan_cost) {
            h = partialAssignment->estimated_heuristic;
            if (match_heuristics)
                h = state_value[partialAssignment->to_string()];
        }

        if (state_representation == "complete") {
            State s = partialAssignment->get_full_state(true, *rng).second;
            if (task_properties::is_goal_state(task_proxy, s)) h = 0;
            s.unpack();
            values_set.push_back(make_pair(h, s.get_values()));
        } else if (state_representation == "partial" || state_representation == "undefined") {
            if (task_properties::is_goal_state(
                    task_proxy, partialAssignment->get_full_state(true, *rng).second))
                h = 0;
            values_set.push_back(make_pair(h, partialAssignment->get_values()));
        } else if (state_representation == "assign_undefined") {
            for (State &s : assign_undefined_state(partialAssignment, 5*assignments_by_undefined_state)) {
                if (task_properties::is_goal_state(task_proxy, s)) h = 0;
                s.unpack();
                values_set.push_back(make_pair(h, s.get_values()));
            }
        }
    }

    if (contrasting_samples > 0)
        create_contrasting_samples(values_set, contrasting_samples);

    return values_to_samples(values_set);
}

SamplingSearchFukunaga::SamplingSearchFukunaga(const options::Options &opts)
    : SamplingSearchBase(opts),
      store_plan_cost(opts.get<bool>("store_plan_cost")),
      store_state(opts.get<bool>("store_state")),
      state_representation(opts.get<string>("state_representation")),
      match_heuristics(opts.get<bool>("match_heuristics")),
      assignments_by_undefined_state(opts.get<int>("assignments_by_undefined_state")),
      contrasting_samples(opts.get<int>("contrasting_samples")),
      relevant_facts(task_properties::get_strips_fact_pairs(task.get())),
      header(construct_header()),
      rng(utils::parse_rng_from_options(opts)) {
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
            "State facts representation format (complete, partial, or undefined, assign_undefined).",
            "complete");
    parser.add_option<bool>(
            "match_heuristics",
            "Identical states receive the best heuristic value assigned between them.",
            "false");
    parser.add_option<int>(
            "assignments_by_undefined_state",
            "Number of states generated from each undefined state (only with assign_undefined).",
            "10"
    );
    parser.add_option<int>(
            "contrasting_samples",
            "Generate new random samples with h = L+1. (Percentage of those obtained with the search).",
            "0"
    );

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
