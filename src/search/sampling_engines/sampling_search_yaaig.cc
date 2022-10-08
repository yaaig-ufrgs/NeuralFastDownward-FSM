#include "sampling_search_yaaig.h"

#include "sampling_search_base.h"
#include "sampling_engine.h"

#include "../option_parser.h"
#include "../plugin.h"

#include "../task_utils/task_properties.h"
#include "../task_utils/successor_generator.h"

#include "../evaluator.h"
#include "../evaluation_context.h"

#include <sstream>
#include <fstream>
#include <string>
#include <chrono>
#include <queue>
#include <iterator>

using namespace std;

namespace sampling_engine {

string SamplingSearchYaaig::construct_header() const {
    ostringstream oss;

    if (store_plan_cost){
        oss << "#<PlanCost>=single integer value" << endl;
    }
    if (store_state) {
        oss << "#<State>=";
        for (unsigned i = 0; i < relevant_facts.size(); i++)
            oss << task->get_fact_name(relevant_facts[i]) << state_separator;
        oss.seekp(-1,oss.cur);
    }

    return oss.str();
}

string SamplingSearchYaaig::sample_file_header() const {
    return header;
}

vector<string> SamplingSearchYaaig::format_output(vector<shared_ptr<PartialAssignment>>& samples) {
    utils::g_log << "[Sampling Engine] Formatting the output..." << endl;

    vector<string> lines;
    for (shared_ptr<PartialAssignment>& s: samples) {
        ostringstream line;
        if (store_plan_cost)
            line << s->estimated_heuristic << field_separator;
        if (store_state)
            line << s->to_binary();
        lines.push_back(line.str());
    }
    return lines;
}

vector<int> SamplingSearchYaaig::binary_to_values(string bin) {
    vector<int> values = {};
    assert(bin.size() == relevant_facts.size());
    for (unsigned i = 0; i < bin.size(); i++) {
        if (bin[i] == '1')
            values.push_back(relevant_facts[i].value);
        else if ((values.size() == (unsigned)relevant_facts[i].var) &&
                (i == bin.size()-1 || relevant_facts[i].var != relevant_facts[i+1].var))
            values.push_back(relevant_facts[i].value+1);
        else
            continue;
        assert(values.size()-1 == (unsigned)relevant_facts[i].var);
    }
    return values;
}

void SamplingSearchYaaig::create_trie_statespace() {
    // Does nothing if there is no input file or it was already created in a previous call
    if (statespace_file == "none" || !trie_statespace.empty())
        return;

    auto t_sstrie = std::chrono::high_resolution_clock::now();
    utils::g_log << "[State space] Creating the state space trie..." << endl;
    string h_sample;
    ifstream f(statespace_file);
    if (f.is_open()) {
        while (getline(f, h_sample)) {
            if (h_sample[0] == '#')
                continue;
            int h = stoi(h_sample.substr(0, h_sample.find(';')));
            vector<int> key;
            string bin = "";
            for (char &b : h_sample.substr(h_sample.find(';') + 1, h_sample.size())) {
                key.push_back((int)b - '0');
                bin += b;
            }
            trie_statespace.insert(key, make_pair(h, bin));
        }
        f.close();
        utils::g_log << "[State space] Time creating trie: " << fixed << (std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_sstrie).count() / 1000.0) << "s" << endl;
    } else {
        utils::g_log << "[State space] *** COULD NOT OPEN STATE SPACE FILE (" << statespace_file << ")! ***" << endl;
    }
}

void SamplingSearchYaaig::successor_improvement(vector<shared_ptr<PartialAssignment>>& samples) {
    if (sui_k <= 0)
        return;
    if (use_evaluator)
        return;
    if (statespace_file != "none")
        create_trie_statespace();

    // Trie
    utils::g_log << "[SUI] Creating the SUI trie..." << endl;
    auto t_sui = std::chrono::high_resolution_clock::now();
    auto t = t_sui;
    trie::trie<shared_ptr<PartialAssignment>> trie;
    for (shared_ptr<PartialAssignment>& s: samples) {
        trie.insert(s->get_values(), s);
    }
    utils::g_log << "[SUI] Time creating trie: " << fixed << (std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t).count() / 1000.0) << "s" << endl;

    // Check for hash conflicts
    unordered_set<size_t> hashset;
    unordered_set<string> sampleset;
    for (shared_ptr<PartialAssignment> &s : samples) {
        size_t h_key = hash<string>{}(s->values_to_string());
        if (sampleset.count(s->values_to_string()) == 0) {
            sampleset.insert(s->values_to_string());
            assert(hashset.count(h_key) == 0);
            hashset.insert(h_key);
        }
    }
    hashset.clear();
    sampleset.clear();

    // Mapping
    utils::g_log << "[SUI] Computing the mapping..." << endl;
    t = std::chrono::high_resolution_clock::now();
    const std::unique_ptr<successor_generator::SuccessorGenerator> succ_generator =
        utils::make_unique_ptr<successor_generator::SuccessorGenerator>(task_proxy);
    const OperatorsProxy operators = task_proxy.get_operators();
    unordered_map<size_t,SuiNode> sui_mapping;
    for (shared_ptr<PartialAssignment>& s : samples) {
        size_t s_key = hash<string>{}(s->values_to_string());
        if (sui_mapping[s_key].samples.size() == 0) {
            vector<OperatorID> applicable_operators;
            succ_generator->generate_applicable_ops(*s, applicable_operators, true);
            for (OperatorID& op_id : applicable_operators) {
                OperatorProxy op_proxy = operators[op_id];
                PartialAssignment t = s->get_partial_successor(op_proxy);
                if (state_representation == "complete_no_mutex" || !t.violates_mutexes()) {
                    std::vector<shared_ptr<PartialAssignment>> compatible_states;
                    trie.find_all_compatible(t.get_values(), sui_rule, compatible_states);
                    for (shared_ptr<PartialAssignment>& t_: compatible_states) {
                        size_t t_key = hash<string>{}(t_->values_to_string());
                        pair<size_t,int> pair = make_pair(t_key, op_proxy.get_cost());
                        if (find(sui_mapping[s_key].successors.begin(), sui_mapping[s_key].successors.end(), pair)
                                == sui_mapping[s_key].successors.end()) {
                            sui_mapping[s_key].successors.push_back(pair);
                        }
                    }
                }
            }
        }
        sui_mapping[s_key].samples.push_back(s);
        if (s->estimated_heuristic < sui_mapping[s_key].best_h)
            sui_mapping[s_key].best_h = s->estimated_heuristic;
    }
    if (sai_partial) {
        for (pair<size_t,SuiNode> p : sui_mapping) {
            for (shared_ptr<PartialAssignment>& s : p.second.samples)
                s->estimated_heuristic = p.second.best_h;
        }
    }
    utils::g_log << "[SUI] Time creating mapping: " << fixed << (std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t).count() / 1000.0) << "s" << endl;

    // SUI loop
    utils::g_log << "[SUI] Updating the h-values..." << endl;
    t = std::chrono::high_resolution_clock::now();
    bool relaxed, any_relaxed;
    do {
        any_relaxed = false;
        for (pair<size_t,SuiNode> s: sui_mapping) {
            relaxed = false;
            for (pair<size_t,int> s_ : s.second.successors) { // pair<state,op_cost>
                int candidate_heuristic = sui_mapping[s_.first].best_h + (unit_cost ? 1 : s_.second);
                if (candidate_heuristic < sui_mapping[s.first].best_h) {
                    sui_mapping[s.first].best_h = candidate_heuristic;
                    relaxed = true;
                    for (shared_ptr<PartialAssignment>& s : s.second.samples)
                        s->estimated_heuristic = candidate_heuristic;
                }
            }
            if (relaxed)
                any_relaxed = true;
        }
    } while (any_relaxed);

    utils::g_log << "[SUI] Time updating h-values: " << fixed << (std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t).count() / 1000.0) << "s." << endl;

    utils::g_log << "[SUI] Total time: " << fixed << (std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_sui).count() / 1000.0) << "s." << endl;
}

void SamplingSearchYaaig::sample_improvement(vector<shared_ptr<PartialAssignment>>& samples) {
    if (use_evaluator)
        return;

    auto t_sai = std::chrono::high_resolution_clock::now();
    utils::g_log << "[SAI] Computing SAI..." << endl;

    // Mapping where each state will have a pair, where the first element is the
    // smallest h-value found for the state and the second is a list of pointers
    // to all h-values vars of all identical states.
    unordered_map<string,pair<int,vector<int*>>> pairs;

    for (shared_ptr<PartialAssignment>& s: samples) {
        string bin = s->to_binary(true);
        if (pairs.count(bin) == 0) {
            pairs[bin] = make_pair(
                s->estimated_heuristic,
                vector<int*>{&s->estimated_heuristic}
            );
        } else {
            pairs[bin].first = min(pairs[bin].first, s->estimated_heuristic);
            pairs[bin].second.push_back(&s->estimated_heuristic);
        }
    }
    int updates = 0;
    for (pair<string,pair<int,vector<int*>>> p : pairs) {
        for (int* h_ptr : p.second.second) {
            if (*h_ptr != p.second.first) {
                assert(*h_ptr > p.second.first);
                *h_ptr = p.second.first;
                updates++;
            }
        }
    }
    utils::g_log << "[SAI] Updated samples:" << updates << endl;
    utils::g_log << "[SAI] Done in " << fixed << (std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_sai).count() / 1000.0) << "s." << endl;
}

void SamplingSearchYaaig::replace_h_with_evaluator(vector<shared_ptr<PartialAssignment>>& samples) {
    auto t_eval = std::chrono::high_resolution_clock::now();
    utils::g_log << "[Evaluator] Replacing h-values according to the evaluator "
        << evaluator->get_description() << "..." << endl;
    utils::g_log << "[Evaluator] Eval equal to infinite will be replaced by 2*regression_depth_value ("
        << regression_depth_value*2 << ")." << endl;

    int total_inf = 0;
    for (shared_ptr<PartialAssignment>& s: samples) {
        vector<int> values = s->get_values();
        EvaluationContext eval_context(registry.insert_state(move(values)));
        EvaluationResult eval_results = evaluator->compute_result(eval_context);
        assert(!eval_results.is_uninitialized());
        // If the state is not found in the PDB then replace it with regression_depth_value*2!
        if (eval_results.is_infinite()) {
            s->estimated_heuristic = regression_depth_value*2;
            total_inf++;
        } else {
            s->estimated_heuristic = eval_results.get_evaluator_value();
        }
    }
    utils::g_log << "[Evaluator] Total infinite values: " << total_inf << "/" << samples.size() << endl;
    utils::g_log << "[Evaluator] Done in " << fixed << (std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_eval).count() / 1000.0) << "s." << endl;
}

void SamplingSearchYaaig::create_random_samples(
    vector<shared_ptr<PartialAssignment>>& samples, int num_random_samples
) {
    if (num_random_samples == 0) return;
    auto t_rs = std::chrono::high_resolution_clock::now();

    utils::g_log << "[Random Samples] Inserting " << num_random_samples << " random samples..." << endl;

    PartialAssignment pa_aux = *(samples[0]);
    // Hack: if 100% random then we sample 1 state to know the structure of the state.
    // At this point it is no longer important.
    if (samples.size() == 1)
        samples.clear();
    const size_t n_atoms = pa_aux.get_values().size();

    unordered_map<string,int> binary_hvalue;
    // If full state SAI is enabled this will be done implicitly later
    if (sai_complete) {
        for (shared_ptr<PartialAssignment>& s: samples) {
            string bin = s->to_binary(true);
            if (binary_hvalue.count(bin) == 0)
                binary_hvalue[bin] = s->estimated_heuristic;
        }
    }

     // Biggest h found in the sampling
    int max_h = -1;
    for (shared_ptr<PartialAssignment>& s: samples)
        max_h = max(max_h, s->estimated_heuristic);
    assert(max_h != -1);
    utils::g_log << "[Random Samples] Biggest h-value found in the samples: " << max_h << endl;
    utils::g_log << "[Random Samples] h-value " << max_h + 1 << " will be assigned for unknown samples." << endl;

    while (num_random_samples > 0) {
        PartialAssignment random_sample(pa_aux, vector<int>(n_atoms, PartialAssignment::UNASSIGNED));
        pair<bool,State> p = random_sample.get_full_state(state_representation != "complete_no_mutex", *rng);
        if (!p.first)
            continue;
        random_sample.assign(p.second.get_values());
        random_sample.estimated_heuristic = max_h + 1;
        if (sai_complete) {
            string bin = random_sample.to_binary(true);
            if (binary_hvalue.count(bin) != 0)
                random_sample.estimated_heuristic = binary_hvalue[bin];
        }
        samples.push_back(make_shared<PartialAssignment>(random_sample));
        num_random_samples--;
    }

    utils::g_log << "[Random Samples] Done in " << fixed << (std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_rs).count() / 1000.0) << "s." << endl;
}

vector<string> SamplingSearchYaaig::extract_samples() {
    utils::g_log << "[Sampling Engine] Extracting samples..." << endl;

    if (sui_k > 0) {
        if (sai_partial)
            utils::g_log << "[SAI] SAI in partial states will be done implicitly with the SUI." << endl;
        successor_improvement(sampling_technique::modified_tasks);
    } else if (sai_partial) {
        sample_improvement(sampling_technique::modified_tasks);
    }
    
    auto t_completion = std::chrono::high_resolution_clock::now();
    utils::g_log << "[Sample Completion] " << sampling_technique::modified_tasks.size() << " samples obtained in sampling." << endl;
    utils::g_log << "[Sample Completion] State representation: " << state_representation << endl;
    for (shared_ptr<PartialAssignment>& s: sampling_technique::modified_tasks) {
        if (state_representation == "complete" || state_representation == "complete_no_mutex") {
            pair<bool,State> fs = s->get_full_state(state_representation != "complete_no_mutex", *rng);
            if (!fs.first) {
                utils::g_log << "[Sample Completion] Could not cast " << s->to_binary(true)
                    << " to full state. Undefined values will be output as 0 in binary." << endl;
            }
            s->assign(fs.second.get_values());

        } else if (state_representation == "valid") {
            // Looking for valid state in forward state space
            if (trie_statespace.empty()) create_trie_statespace();
            vector<int> key;
            for (char &b : s->to_binary(true))
                key.push_back(b == '*' ? -1 : (int)b - '0');
            vector<pair<int,string>> compatibles;
            trie_statespace.find_all_compatible(key, SearchRule::subsets, compatibles);
            if (compatibles.empty()) {
                utils::g_log << "[ERROR] Sample " << s->to_binary(true)
                    << " not found in state space!" << endl;
                exit(0);
            }
            s->assign(binary_to_values(compatibles[(*rng)()*compatibles.size()].second));

        } else if (state_representation == "partial") {
            // nothing to do

        } else {
            utils::g_log << "[ERROR] State representation \"" << state_representation << "\" not implemented!";
            exit(0);
        }

        if (task_properties::is_goal_assignment(task_proxy, *s))
            s->estimated_heuristic = 0;
    }
    utils::g_log << "[Sample Completion] Done in " << fixed << (std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_completion).count() / 1000.0) << "s." << endl;

    if (sampling_technique::random_samples > 0)
        create_random_samples(sampling_technique::modified_tasks, sampling_technique::random_samples);

    if (sai_complete && state_representation != "partial")
        sample_improvement(sampling_technique::modified_tasks);

    if (use_evaluator)
        replace_h_with_evaluator(sampling_technique::modified_tasks);

    return format_output(sampling_technique::modified_tasks);
}

SamplingSearchYaaig::SamplingSearchYaaig(const options::Options &opts)
    : SamplingSearchBase(opts),
      store_plan_cost(opts.get<bool>("store_plan_cost")),
      store_state(opts.get<bool>("store_state")),
      state_representation(opts.get<string>("state_representation")),
      sai_partial(opts.get<string>("sai") == "partial" || opts.get<string>("sai") == "both"),
      sai_complete(opts.get<string>("sai") == "complete" || opts.get<string>("sai") == "both"),
      sui_k(opts.get<int>("sui_k")),
      sui_rule(getRule(opts.get<string>("sui_rule"))),
      statespace_file(opts.get<string>("statespace_file")),
      evaluator(opts.get<shared_ptr<Evaluator>>("evaluator")),
      use_evaluator(
          evaluator->get_description() != "evaluator = blind" &&
          evaluator->get_description() != "blind"),
      relevant_facts(task_properties::get_strips_fact_pairs(task.get())),
      registry(task_proxy),
      header(construct_header()),
      rng(utils::parse_rng_from_options(opts)) {
    assert(sui_k == 0 || sui_k == 1);
    if (statespace_file != "none")
        create_trie_statespace();
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
            "State facts representation format (complete, complete_no_mutex, partial, valid).",
            "complete");
    parser.add_option<string>(
            "sai",
            "Identical states receive the best heuristic value assigned between them (SAI in: none, partial, complete, both).",
            "none");
    parser.add_option<int>(
            "sui_k",
            "Correct h-values using SUI via K-step forward repeatedly",
            "0");
    parser.add_option<string>(
            "sui_rule",
            "Rule applied when checking subset states.",
            "vu_u");
    parser.add_option<string>(
            "statespace_file",
            "Path to file with h;sample for statespace trie.",
            "none");
    parser.add_option<shared_ptr<Evaluator>>(
            "evaluator",
            "Evaluator to use to estimate the h-values.",
            "blind()");

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
