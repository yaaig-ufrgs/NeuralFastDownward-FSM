#include "sampling_search_yaaig.h"

#include "sampling_search_base.h"
#include "sampling_engine.h"

#include "../option_parser.h"
#include "../plugin.h"

#include "../task_utils/task_properties.h"
#include "../task_utils/successor_generator.h"

#include <sstream>
#include <string>

using namespace std;

namespace sampling_engine {

string SamplingSearchYaaig::construct_header() const {
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

string SamplingSearchYaaig::sample_file_header() const {
    return header;
}

unordered_map<string,int> SamplingSearchYaaig::do_minimization(unordered_map<string,int>& state_value) {
    // If minimization then then all identical samples receive the smallest h value among them
    for (shared_ptr<PartialAssignment>& partialAssignment: sampling_technique::modified_tasks) {
        string bin = partialAssignment->to_binary();
        int h = partialAssignment->estimated_heuristic;
        if (state_value.count(bin) == 0 || h < state_value[bin])
            state_value[bin] = h;
    }
    return state_value;
}

vector<State> SamplingSearchYaaig::assign_undefined_state(shared_ptr<PartialAssignment>& pa, int max_attempts) {
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

void SamplingSearchYaaig::create_contrasting_samples(
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
    if (minimization) {
        for (pair<int,vector<int>>& p : values_set) {
            string s = PartialAssignment(*task, vector<int>(p.second)).to_binary();
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
        if (minimization) {
            string bin = s.to_binary();
            if (state_value.count(bin) != 0)
                h = state_value[bin];
        }
        
        values_set.push_back(make_pair(h, values));
        samples_to_be_created--;
    }
}

vector<string> SamplingSearchYaaig::values_to_samples(vector<pair<int,vector<int>>> values_set) {
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

void SamplingSearchYaaig::approximate_value_iteration(unordered_map<string,int>& state_value) {
    if (avi_k <= 0)
        return;

    // do_minimization to create state_value mapping
    // TODO: AVI without minimization (for now it's not necessary)
    do_minimization(state_value);

    const std::unique_ptr<successor_generator::SuccessorGenerator> succ_generator =
        utils::make_unique_ptr<successor_generator::SuccessorGenerator>(task_proxy);
    const OperatorsProxy operators = task_proxy.get_operators();

    unordered_set<string> already_computed;
    for (shared_ptr<PartialAssignment>& partialAssignment: sampling_technique::modified_tasks) {
        string pa_str = partialAssignment->to_binary();
        if (already_computed.count(pa_str) != 0)
            continue;
        already_computed.insert(pa_str);
        vector<OperatorID> applicable_operators;
        succ_generator->generate_applicable_ops(*partialAssignment, applicable_operators);
        for (OperatorID& op_id : applicable_operators) {
            OperatorProxy op_proxy = operators[op_id];
            string succ_pa_str = partialAssignment->get_partial_successor(op_proxy).to_binary();
            if (state_value.count(succ_pa_str) != 0) {
                state_value[succ_pa_str] = min(
                    state_value[succ_pa_str],
                    state_value[pa_str] + op_proxy.get_cost()
                );
            }
        }
    }
}

vector<string> SamplingSearchYaaig::extract_samples() {
    unordered_map<string,int> state_value;
    if (avi_k > 0)
        approximate_value_iteration(state_value);
    else if (minimization)
        do_minimization(state_value);

    vector<pair<int,vector<int>>> values_set;
    for (shared_ptr<PartialAssignment>& partialAssignment: sampling_technique::modified_tasks) {
        int h = -1;
        if (store_plan_cost) {
            h = (state_value.empty()) ?
                partialAssignment->estimated_heuristic :
                state_value[partialAssignment->to_binary()];
        }

        if (state_representation == "complete" || state_representation == "complete_no_mutex") {
            State s = partialAssignment->get_full_state(
                state_representation != "complete_no_mutex", *rng).second;
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

SamplingSearchYaaig::SamplingSearchYaaig(const options::Options &opts)
    : SamplingSearchBase(opts),
      store_plan_cost(opts.get<bool>("store_plan_cost")),
      store_state(opts.get<bool>("store_state")),
      state_representation(opts.get<string>("state_representation")),
      minimization(opts.get<bool>("minimization")),
      assignments_by_undefined_state(opts.get<int>("assignments_by_undefined_state")),
      contrasting_samples(opts.get<int>("contrasting_samples")),
      avi_k(opts.get<int>("avi_k")),
      relevant_facts(task_properties::get_strips_fact_pairs(task.get())),
      header(construct_header()),
      rng(utils::parse_rng_from_options(opts)) {
    assert(contrasting_samples >= 0 && contrasting_samples <= 100);
    assert(assignments_by_undefined_state > 0);
    assert(avi_k == 0 || avi_k == 1);
}

static shared_ptr<SearchEngine> _parse_sampling_search_yaaig(OptionParser &parser) {
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
            "State facts representation format (complete, complete_no_mutex, partial, or undefined, assign_undefined).",
            "complete");
    parser.add_option<bool>(
            "minimization",
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
    parser.add_option<int>(
            "avi_k",
            "Correct h-values using AVI via K-step forward repeatedly",
            "0"
    );

    SearchEngine::add_options_to_parser(parser);
    Options opts = parser.parse();
    shared_ptr<sampling_engine::SamplingSearchYaaig> engine;
    if (!parser.dry_run()) {
        engine = make_shared<sampling_engine::SamplingSearchYaaig>(opts);
    }

    return engine;
}

static Plugin<SearchEngine> _plugin_search("sampling_search_yaaig", _parse_sampling_search_yaaig);

}
