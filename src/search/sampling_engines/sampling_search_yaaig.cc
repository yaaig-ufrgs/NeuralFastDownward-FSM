#include "sampling_search_yaaig.h"

#include "sampling_search_base.h"
#include "sampling_engine.h"

#include "../option_parser.h"
#include "../plugin.h"

#include "../task_utils/task_properties.h"
#include "../task_utils/successor_generator.h"

#include <sstream>
#include <fstream>
#include <string>
#include <limits.h>

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

double SamplingSearchYaaig::mse(trie::trie<int> trie_mse, bool root) {
    double sum = 0.0;
    for (shared_ptr<PartialAssignment>& pa: sampling_technique::modified_tasks) {
        int best_h = INT_MAX;
        for (int& hs: trie_mse.find_all_compatible(pa->get_values(), true)) {
            best_h = min(best_h, hs);
        }
        assert(best_h != INT_MAX);
        int err = best_h - pa->estimated_heuristic;
        sum += (err * err);
    }
    double e = sum / sampling_technique::modified_tasks.size();
    return root ? sqrt(e) : e;
}

void SamplingSearchYaaig::approximate_value_iteration(
    trie::trie<shared_ptr<PartialAssignment>> trie,
    trie::trie<int> trie_mse
) {
    if (avi_k <= 0 || avi_its <= 0)
        return;

    ofstream mse_result(mse_result_file);
    if (!trie_mse.empty()) {
        double e = mse(trie_mse);
        cout << endl << "[AVI] RMSE #0: " << sqrt(e) << endl;
        mse_result << "avi_it,mse,rmse" << endl;
        mse_result << 0 << "," << e << "," << sqrt(e) << endl;
    }

    const std::unique_ptr<successor_generator::SuccessorGenerator> succ_generator =
        utils::make_unique_ptr<successor_generator::SuccessorGenerator>(task_proxy);
    const OperatorsProxy operators = task_proxy.get_operators();

    for (int i = 0; i < avi_its; i++) {
        for (shared_ptr<PartialAssignment>& pa: sampling_technique::modified_tasks) {
            vector<OperatorID> applicable_operators;
            succ_generator->generate_applicable_ops(*pa, applicable_operators);
            for (OperatorID& op_id : applicable_operators) {
                OperatorProxy op_proxy = operators[op_id];
                PartialAssignment succ_pa = pa->get_partial_successor(op_proxy);
                for (shared_ptr<PartialAssignment>& _pa_succ: trie.find_all_compatible(succ_pa.get_values())) {
                    _pa_succ->estimated_heuristic = min(
                        _pa_succ->estimated_heuristic,
                        pa->estimated_heuristic + op_proxy.get_cost()
                    );
                }
            }
        }
        if (!trie_mse.empty()) {
            double e = mse(trie_mse);
            cout << "[AVI] RMSE #" << (i+1) << ": " << sqrt(e) << endl;
            mse_result << (i+1) << "," << e << "," << sqrt(e) << endl;

        }
    }
    mse_result.close();
}

struct SortIncreasingH {
  bool operator()(shared_ptr<PartialAssignment>& object1, shared_ptr<PartialAssignment>& object2) {
    return(object1->estimated_heuristic < object2->estimated_heuristic);
  }
};

vector<string> SamplingSearchYaaig::extract_samples() {
    if (sort_h) {
        sort(sampling_technique::modified_tasks.begin(), sampling_technique::modified_tasks.end(), SortIncreasingH());
    }

    trie::trie<int> trie_mse;
    if (compute_mse) {
        string h_sample;
        ifstream f(mse_hstar_file);
        while (getline(f, h_sample)) {
            if (h_sample[0] == '#')
                continue;
            int h = stoi(h_sample.substr(0, h_sample.find(';')));
            vector<int> key;
            for (char& b : h_sample.substr(h_sample.find(';') + 1, h_sample.size()))
                key.push_back((int)b - '0');
            trie_mse.insert(key, h);
        }
        f.close();
    }

    unordered_map<string,int> state_value;
    if (avi_k > 0) {
        trie::trie<shared_ptr<PartialAssignment>> trie;
        for (shared_ptr<PartialAssignment>& partialAssignment: sampling_technique::modified_tasks) {
            string bin = partialAssignment->to_binary();
            int h = partialAssignment->estimated_heuristic;
            // Maintain a mapping to keep the smallest h-value
            // in the trie in case there are duplicate samples
            if (state_value.count(bin) == 0 || h < state_value[bin]) {
                trie.insert(partialAssignment->get_values(), partialAssignment);
                state_value[bin] = h;
            }
        }
        approximate_value_iteration(trie, trie_mse);
    }
    if (minimization) {
        for (shared_ptr<PartialAssignment>& partialAssignment: sampling_technique::modified_tasks) {
            string bin = partialAssignment->to_binary();
            int h = partialAssignment->estimated_heuristic;
            if (state_value.count(bin) == 0 || h < state_value[bin]) {
                state_value[bin] = h;
            }
        }
        do_minimization(state_value);
    }

    vector<pair<int,vector<int>>> values_set;
    for (shared_ptr<PartialAssignment>& partialAssignment: sampling_technique::modified_tasks) {
        int h = -1;
        if (store_plan_cost) {
            h = (minimization) ?
                state_value[partialAssignment->to_binary()] :
                partialAssignment->estimated_heuristic;
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
      avi_its(opts.get<int>("avi_its")),
      sort_h(opts.get<bool>("sort_h")),
      mse_hstar_file(opts.get<string>("mse_hstar_file")),
      mse_result_file(opts.get<string>("mse_result_file")),
      relevant_facts(task_properties::get_strips_fact_pairs(task.get())),
      header(construct_header()),
      rng(utils::parse_rng_from_options(opts)),
      compute_mse(mse_hstar_file != "none") {
    assert(contrasting_samples >= 0 && contrasting_samples <= 100);
    assert(assignments_by_undefined_state > 0);
    assert(avi_k == 0 || avi_k == 1);
    assert(avi_its > 0);
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
            "10");
    parser.add_option<int>(
            "contrasting_samples",
            "Generate new random samples with h = L+1. (Percentage of those obtained with the search).",
            "0");
    parser.add_option<int>(
            "avi_k",
            "Correct h-values using AVI via K-step forward repeatedly",
            "0");
    parser.add_option<int>(
            "avi_its",
            "Number of AVI repeats.",
            "1");
    parser.add_option<bool>(
            "sort_h",
            "Sort samples by increasing h-values.",
            "false");
    parser.add_option<string>(
            "mse_hstar_file",
            "Path to file with h;sample for MSE.",
            "none");
    parser.add_option<string>(
            "mse_result_file",
            "Path to save MSE results.",
            "sampling_mse.csv");

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
