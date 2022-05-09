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

void SamplingSearchYaaig::do_minimization(vector<shared_ptr<PartialAssignment>>& states) {
    // Mapping where each state will have a pair, where the first element is the
    // smallest h-value found for the state and the second is a list of pointers
    // to all h-values vars of all identical states.
    unordered_map<string,pair<int,vector<int*>>> pairs;

    for (shared_ptr<PartialAssignment>& s: states) {
        string bin = s->to_binary(true);
        if (pairs.count(bin) == 0) {
            pairs[bin] = make_pair(
                s->estimated_heuristic,
                vector<int*>{&s->estimated_heuristic}
            );
        } else {
            if (s->estimated_heuristic < pairs[bin].first)
                pairs[bin].first = s->estimated_heuristic;
            pairs[bin].second.push_back(&s->estimated_heuristic);
        }
    }
    for (pair<string,pair<int,vector<int*>>> p : pairs) {
        for (int* h_ptr : p.second.second)
            *h_ptr = p.second.first;
    }
}

void SamplingSearchYaaig::do_minimization(vector<pair<int,pair<vector<int>,string>>>& states) {
    unordered_map<string,pair<int,vector<int*>>> pairs;

    for (pair<int,pair<vector<int>,string>>& s: states) {
        if (pairs.count(s.second.second) == 0) {
            pairs[s.second.second] = make_pair(s.first, vector<int*>{&s.first});
        } else {
            if (s.first < pairs[s.second.second].first)
                pairs[s.second.second].first = s.first;
            pairs[s.second.second].second.push_back(&s.first);
        }
    }
    for (pair<string,pair<int,vector<int*>>> p : pairs) {
        for (int* h_ptr : p.second.second)
            *h_ptr = p.second.first;
    }
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
    vector<pair<int,pair<vector<int>,string>>>& values_set,
    int percentage
) {
    if (percentage == 0)
        return;
    // assert(percentage > 0 && percentage <= 100);
    if (!(percentage > 0 && percentage <= 100)) exit(10);

    const size_t n_atoms = sampling_technique::modified_tasks[0]->get_values().size();
    PartialAssignment pa(
        *sampling_technique::modified_tasks[0],
        vector<int>(n_atoms, PartialAssignment::UNASSIGNED)
    );

    // Biggest h found in the search
    int max_h = 0;
    for (pair<int,pair<vector<int>,string>>& p : values_set)
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
    if (minimization != "none") {
        for (pair<int,pair<vector<int>,string>>& p : values_set) {
            if (state_value.count(p.second.second) == 0)
                state_value[p.second.second] = p.first;
        }
    }

    while (samples_to_be_created > 0) {
        pair<bool,State> fs = pa.get_full_state(true, *rng);
        if (!fs.first)
            continue;
        State s = fs.second;
        s.unpack();

        int h = max_h;
        if (minimization != "none") {
            string bin = s.to_binary();
            if (state_value.count(bin) != 0)
                h = state_value[bin];
        }
        
        values_set.push_back(make_pair(h, make_pair(s.get_values(), s.to_binary())));
        samples_to_be_created--;
    }
}

vector<string> SamplingSearchYaaig::values_to_samples(
    vector<pair<int,pair<vector<int>,string>>> values_set
) {
    vector<string> samples;
    for (pair<int,pair<vector<int>,string>>& p : values_set) {
        ostringstream oss;
        if (store_plan_cost)
            oss << p.first << field_separator;
        if (store_state) {
            if (state_representation == "facts_partial" || state_representation == "facts_complete") {
                for (unsigned i = 0; i < relevant_facts.size(); i++) {
                    if (p.second.first[relevant_facts[i].var] == relevant_facts[i].value) {
                        oss << task->get_fact_name(relevant_facts[i]);
                        if (i < relevant_facts.size() - 1)
                            oss << ' ';
                    }
                }
            } else if (state_representation == "values_partial" || state_representation == "values_complete") {
                for (unsigned i = 0; i < p.second.first.size(); i++) {
                    oss << p.second.first[i];
                    if (i < p.second.first.size() - 1)
                        oss << ' ';
                }
            } else if (
                    state_representation == "complete" ||
                    state_representation == "complete_no_mutex" ||
                    state_representation == "partial" ||
                    state_representation == "undefined_char" ||
                    state_representation == "valid") {
                oss << p.second.second;
            } else if (state_representation == "undefined" || state_representation == "assign_undefined") {
                for (unsigned i = 0; i < relevant_facts.size(); i++) {
                    if ((state_representation == "undefined") && (i == 0 || relevant_facts[i].var != relevant_facts[i-1].var))
                        oss << (p.second.first[relevant_facts[i].var] == PartialAssignment::UNASSIGNED);
                    oss << (p.second.first[relevant_facts[i].var] == relevant_facts[i].value ? 1 : 0);
                }
            } else {
                exit(10); // not implemented
            }
        }
        samples.push_back(oss.str());
    }
    return samples;
}

double SamplingSearchYaaig::mse(vector<shared_ptr<PartialAssignment>>& samples, bool root) {
    double sum = 0.0;
    for (shared_ptr<PartialAssignment>& pa: samples) {
        int best_h = INT_MAX;
        vector<int> key;
        for (char& b : pa->to_binary(true))
            key.push_back(b == '*' ? -1 : (int)b - '0');
        for (pair<int,string> p: trie_statespace.find_all_compatible(key, "v_vu"))
            best_h = min(best_h, p.first);
        // assert(best_h != INT_MAX);
        if (best_h != INT_MAX) {
            int err = best_h - pa->estimated_heuristic;
            sum += (err * err);
        }
    }
    double e = sum / samples.size();
    return root ? sqrt(e) : e;
}

void SamplingSearchYaaig::log_mse(int updates) {
    ostringstream file_str, log_str;
    static chrono::_V2::system_clock::time_point start;
    if (updates == 0) {
        start = std::chrono::high_resolution_clock::now();
        file_str << "updates,mse,rmse,time (sec)" << endl;
    }
    double e = -1, re = -1;
    if (!trie_statespace.empty()) {
        chrono::_V2::system_clock::time_point start_mse;
        e = mse(sampling_technique::modified_tasks);
        re = sqrt(e);
        file_str << updates << "," << e << "," << re << ",," << endl;
    }
    log_str << "[AVI] Updates: " << updates;
    if (re != -1)
        log_str << " | RMSE: " << re;
    log_str << " | Time: " << fixed << (chrono::duration<double, milli>(
            chrono::high_resolution_clock::now() - start).count() / 1000.0) << "s";
    if (mse_result_file != "none" && !trie_statespace.empty()) {
        ofstream mse_result(mse_result_file);
        mse_result << file_str.str();
        mse_result.close();
    }
    utils::g_log << log_str.str() << "\n";
}

void SamplingSearchYaaig::create_trie_statespace() {
    if (mse_hstar_file == "none") {
        // utils::g_log << "Could not create trie_statespace: missing state space file." << endl;
        return;
    }
    if (!trie_statespace.empty()) // was already created in a previous call
        return;
    auto t_start_avi = std::chrono::high_resolution_clock::now();
    string h_sample;
    ifstream f(mse_hstar_file);
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
        utils::g_log << "Time creating trie_statespace: " << (std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_start_avi).count() / 1000.0) << "s" << endl;
    } else {
        utils::g_log << "*** Could not open state space file! ***" << endl;
    }
}

void SamplingSearchYaaig::approximate_value_iteration() {
    if (avi_k <= 0 || avi_its <= 0)
        return;
    if (mse_hstar_file != "none")
        create_trie_statespace();

    // Trie
    auto t_avi = std::chrono::high_resolution_clock::now();
    auto t = t_avi;
    unordered_map<string,int> min_pairs_pre;
    trie::trie<shared_ptr<PartialAssignment>> trie;
    for (shared_ptr<PartialAssignment>& partialAssignment: sampling_technique::modified_tasks) {
        string bin = partialAssignment->to_binary(true);
        int h = partialAssignment->estimated_heuristic;
        // Maintain a mapping to keep the smallest h-value
        // in the trie in case there are duplicate samples
        if (min_pairs_pre.count(bin) == 0 || h < min_pairs_pre[bin]) {
            trie.insert(partialAssignment->get_values(), partialAssignment);
            min_pairs_pre[bin] = h;
        }
    }
    utils::g_log << "[AVI] Time creating trie: " << (std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t).count() / 1000.0) << "s" << endl;

    // Mapping
    t = std::chrono::high_resolution_clock::now();
    const std::unique_ptr<successor_generator::SuccessorGenerator> succ_generator =
        utils::make_unique_ptr<successor_generator::SuccessorGenerator>(task_proxy);
    const OperatorsProxy operators = task_proxy.get_operators();
    unordered_map<string,AviNode> avi_mapping;
    for (shared_ptr<PartialAssignment>& s : sampling_technique::modified_tasks) {
        string s_key = s->to_binary(true);
        if (avi_mapping[s_key].samples.size() == 0) {
            vector<OperatorID> applicable_operators;
            succ_generator->generate_applicable_ops(*s, applicable_operators, true);
            for (OperatorID& op_id : applicable_operators) {
                OperatorProxy op_proxy = operators[op_id];
                PartialAssignment t = s->get_partial_successor(op_proxy);
                if (!t.violates_mutexes()) {
                    for (shared_ptr<PartialAssignment>& t_: trie.find_all_compatible(t.get_values(), avi_rule)) {
                        string t_key = t_->to_binary(true);
                        pair<string,int> pair = make_pair(t_key, op_proxy.get_cost());
                        if (find(avi_mapping[s_key].successors.begin(), avi_mapping[s_key].successors.end(), pair)
                                == avi_mapping[s_key].successors.end()) {
                            avi_mapping[s_key].successors.push_back(pair);
                            avi_mapping[t_key].predecessors.push_back(make_pair(s_key, op_proxy.get_cost()));
                        }
                    }
                }
            }
        }
        avi_mapping[s_key].samples.push_back(s);
        if (s->estimated_heuristic < avi_mapping[s_key].best_h)
            avi_mapping[s_key].best_h = s->estimated_heuristic;
    }
    if (minimization == "partial" || minimization == "both") {
        for (pair<string,AviNode> p: avi_mapping)
            for (shared_ptr<PartialAssignment>& s : p.second.samples)
                s->estimated_heuristic = p.second.best_h;
    }
    utils::g_log << "[AVI] Time creating AVI mapping: " << (std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t).count() / 1000.0) << "s" << endl;

    // AVI loop
    utils::g_log << "Starting AVI updates...\n";
    t = std::chrono::high_resolution_clock::now();
    int updates = 0, updates_between_rmse = sampling_technique::modified_tasks.size() * 0.25;
    log_mse(updates);
    set<AviNode*> queue;
    for (pair<string,AviNode> p: avi_mapping)
        queue.insert(&avi_mapping[p.first]);
    while (!queue.empty()) {
        set<AviNode*>::iterator it = queue.begin();
        bool relaxed = false;
        for (pair<string,int> s_ : (*it)->successors) { // pair<binary,op_cost>
            int candidate_heuristic = avi_mapping[s_.first].best_h + (avi_unit_cost ? 1 : s_.second);
            if (candidate_heuristic < (*it)->best_h) {
                (*it)->best_h = candidate_heuristic;
                relaxed = true;
                for (shared_ptr<PartialAssignment>& s : (*it)->samples)
                    s->estimated_heuristic = candidate_heuristic;
            }
        }
        if (relaxed) {
            for (pair<string,int> s_ : (*it)->predecessors)
                queue.insert(&avi_mapping[s_.first]);
            if (++updates % updates_between_rmse == 0)
                log_mse(updates);
        }
        queue.erase(it);
    }
    log_mse(updates);
    utils::g_log << "[AVI] Time updating h-values: " << (std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t).count() / 1000.0) << "s" << endl;

    utils::g_log << "[AVI] Total time: " << (std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_avi).count() / 1000.0) << "s" << endl;
}

vector<string> SamplingSearchYaaig::extract_samples() {
    utils::g_log << endl;

    if (sort_h) {
        sort(
            sampling_technique::modified_tasks.begin(),
            sampling_technique::modified_tasks.end(),
            [](shared_ptr<PartialAssignment>& object1, shared_ptr<PartialAssignment>& object2) {
                return (object1->estimated_heuristic < object2->estimated_heuristic);
            }
        );
    }

    if (avi_k > 0 && avi_its > 0) {
        approximate_value_iteration();
    } else if (minimization == "partial" || minimization == "both") {
        do_minimization(sampling_technique::modified_tasks);
    }
    
    vector<pair<int,pair<vector<int>,string>>> values_set;
    for (shared_ptr<PartialAssignment>& partialAssignment: sampling_technique::modified_tasks) {
        int h = -1;
        if (store_plan_cost)
            h = partialAssignment->estimated_heuristic;

        if (state_representation == "complete" || state_representation == "complete_no_mutex" || state_representation == "values_complete" || state_representation == "facts_complete") {
            State s = partialAssignment->get_full_state(
                state_representation != "complete_no_mutex", *rng).second;
            if (task_properties::is_goal_state(task_proxy, s)) h = 0;
            s.unpack();
            values_set.push_back(
                make_pair(h, make_pair(s.get_values(), s.to_binary()))
            );
        } else if (state_representation == "valid") {
            State s = partialAssignment->get_full_state(true, *rng).second;
            bool is_valid = false;

            if (trie_statespace.empty()) create_trie_statespace();
            vector<int> key;
            for (char &b : partialAssignment->to_binary(true)) {
                key.push_back(b == '*' ? -1 : (int)b - '0');
            }
            vector<pair<int,string>> compatibles = trie_statespace.find_all_compatible(key, "v_vu"); 
            if (!compatibles.empty()) {
                //vector<int> values = binary_to_values(compatibles[(*rng)()*compatibles.size()].second);
                //s = State(*task, move(values));
                is_valid = true;
            }

            for (int i = 0; i < 10000 && !is_valid; i++) { // MAX_TRIES = 10000
                pair<bool,State> p = partialAssignment->get_full_state(true, *rng);
                if (p.first) {
                    s = p.second;
                    vector<int> v = s.get_values();
                    EvaluationContext eval_context(registry.insert_state(move(v)));
                    EvaluationResult eval_results = evaluator->compute_result(eval_context);
                    if (eval_results.is_uninitialized() || eval_results.is_infinite())
                        continue;
                    is_valid = true;
                }
            }
            if (is_valid) {
                if (task_properties::is_goal_assignment(task_proxy, *partialAssignment)) h = 0;
                s.unpack();
                values_set.push_back(
                    make_pair(h, make_pair(s.get_values(), s.to_binary()))
                );
            } else {
                utils::g_log << "Sample " << partialAssignment->to_binary(true)
                    << " not found in state space or PDB + mutex!" << endl;
                continue;
            }
        } else if (state_representation == "partial" || state_representation == "undefined" || state_representation == "undefined_char" || state_representation == "values_partial" || state_representation == "facts_partial") {
            if (task_properties::is_goal_assignment(task_proxy, *partialAssignment))
                h = 0;
            values_set.push_back(
                make_pair(h, make_pair(partialAssignment->get_values(), partialAssignment->to_binary(state_representation == "undefined_char")))
            );
        } else if (state_representation == "assign_undefined") {
            for (State &s : assign_undefined_state(partialAssignment, 5 * assignments_by_undefined_state)) {
                if (task_properties::is_goal_state(task_proxy, s)) h = 0;
                s.unpack();
                values_set.push_back(
                    make_pair(h, make_pair(s.get_values(), s.to_binary()))
                );
            }
        }
    }

    if (contrasting_samples > 0)
        create_contrasting_samples(values_set, contrasting_samples);

    if ((minimization == "complete" || minimization == "both")
            && !(state_representation == "partial" || state_representation == "undefined" || state_representation == "undefined_char")) {
        do_minimization(values_set);
    }

    // blind is the default, sampling keeps the regression value
    if (evaluator->get_description() != "blind")
        replace_h_with_evaluator(values_set);

    if (state_representation == "complete" || state_representation == "complete_no_mutex" || state_representation == "partial" || state_representation == "valid")
        compute_sampling_statistics(values_set);
    return values_to_samples(values_set);
}

void SamplingSearchYaaig::replace_h_with_evaluator(
    vector<pair<int,pair<vector<int>,string>>>& samples
) {
    for (pair<int,pair<vector<int>,string>>& p : samples) {
        vector<int> v = task_proxy.create_state(move(p.second.first)).get_values();
        EvaluationContext eval_context(registry.insert_state(move(v)));
        EvaluationResult eval_results = evaluator->compute_result(eval_context);
        // assert(!eval_results.is_uninitialized());
        if (eval_results.is_uninitialized()) exit(10);
        p.first = eval_results.is_infinite() ? bound_value*2 : eval_results.get_evaluator_value();
    }
}

void SamplingSearchYaaig::compute_sampling_statistics(
    vector<pair<int,pair<vector<int>,string>>> samples
) {
    create_trie_statespace();
    utils::g_log << "Generating statistics on samples...\n";
    if (!trie_statespace.empty()) {
        int not_in_statespace = 0, underestimates = 0, with_hstar = 0, sum_h = 0;
        for (auto& s : samples) {
            int h = s.first;
            sum_h += h;
            vector<int> key;
            for (char& b : s.second.second)
                key.push_back((int)b - '0');
            vector<pair<int, string>> hs = trie_statespace.find_all_compatible(key, "v_v");
            if (hs.size() == 0) {
                not_in_statespace++;
            } else {
                // assert(hs.size() == 1);
                if (!(hs.size() == 1)) exit(10);
                int hstar = hs[0].first;
                if (h == hstar)
                    with_hstar++;
                else if (h < hstar)
                    underestimates++;
            }
        }
        if (evaluator->get_description() == "evaluator = pdb(hstar_pattern(list))") {
            // assert((unsigned)(with_hstar + not_in_statespace) == samples.size());
            if (!((unsigned)(with_hstar + not_in_statespace) == samples.size())) exit(10);
        }

        utils::g_log << "[STATS] Samples: " << samples.size() << "\n";
        utils::g_log << "[STATS] Samples with h*: " << with_hstar << "\n";
        utils::g_log << "[STATS] Samples underestimating h*: " << underestimates << "\n";
        utils::g_log << "[STATS] Samples not in state space: " << not_in_statespace << "\n";
        utils::g_log << "[STATS] Average h value: " << ((float)sum_h / samples.size()) << "\n";
    } else {
        // utils::g_log << "[STATS] trie_statespace was not created." << endl;
    }
}

SamplingSearchYaaig::SamplingSearchYaaig(const options::Options &opts)
    : SamplingSearchBase(opts),
      store_plan_cost(opts.get<bool>("store_plan_cost")),
      store_state(opts.get<bool>("store_state")),
      state_representation(opts.get<string>("state_representation")),
      minimization(opts.get<string>("minimization")),
      assignments_by_undefined_state(opts.get<int>("assignments_by_undefined_state")),
      contrasting_samples(opts.get<int>("contrasting_samples")),
      avi_k(opts.get<int>("avi_k")),
      avi_its(opts.get<int>("avi_its")),
      avi_rule(opts.get<string>("avi_rule")),
      avi_epsilon(stod(opts.get<string>("avi_epsilon"))),
      avi_unit_cost(opts.get<bool>("avi_unit_cost")),
      sort_h(opts.get<bool>("sort_h")),
      mse_hstar_file(opts.get<string>("mse_hstar_file")),
      mse_result_file(opts.get<string>("mse_result_file")),
      evaluator(opts.get<shared_ptr<Evaluator>>("evaluator")),
      relevant_facts(task_properties::get_strips_fact_pairs(task.get())),
      registry(task_proxy),
      header(construct_header()),
      rng(utils::parse_rng_from_options(opts)) {
    // assert(contrasting_samples >= 0 && contrasting_samples <= 100);
    if (!(contrasting_samples >= 0 && contrasting_samples <= 100)) exit(10);
    // assert(assignments_by_undefined_state > 0);
    if (!(assignments_by_undefined_state > 0)) exit(10);
    // assert(avi_k == 0 || avi_k == 1);
    if (!(avi_k == 0 || avi_k == 1)) exit(10);
    // assert(avi_its > 0);
    if (!(avi_its >= 0)) exit(10);
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
            "State facts representation format (complete, complete_no_mutex, partial, valid, undefined, assign_undefined, undefined_char, values_partial, values_complete, facts_partial, or facts_complete).",
            "complete");
    parser.add_option<string>(
            "minimization",
            "Identical states receive the best heuristic value assigned between them (minimization in : none, partial, complete, both).",
            "none");
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
    parser.add_option<string>(
            "avi_rule",
            "Rule applied when checking subset states.",
            "vu_u");
    parser.add_option<string>(
            "avi_epsilon",
            "RMSE no-improvement threshold for AVI early stop.",
            "-1");
    parser.add_option<bool>(
            "avi_unit_cost",
            "Increments h by unit cost instead of operator cost.",
            "false");
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
