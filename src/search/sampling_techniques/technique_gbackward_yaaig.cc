#include <stack>
#include <algorithm>

#include "technique_gbackward_yaaig.h"

#include "../evaluation_result.h"
#include "../heuristic.h"
#include "../plugin.h"

#include "../tasks/modified_init_goals_task.h"
#include "../tasks/partial_state_wrapper_task.h"

#include "../task_utils/sampling.h"

#include "../task_utils/task_properties.h"

#include "../sampling_engines/sampling_engine.h"

using namespace std;

namespace sampling_technique {
const string TechniqueGBackwardYaaig::name = "gbackward_yaaig";

static int compute_heuristic(
        const TaskProxy &task_proxy, Heuristic *bias,
        utils::RandomNumberGenerator &rng, const PartialAssignment &assignment) {
    auto pair_success_state = assignment.get_full_state(
            true, rng);
    if (!pair_success_state.first) {
        return EvaluationResult::INFTY;
    }
    StateRegistry state_registry(task_proxy);
    vector<int> initial_facts= pair_success_state.second.get_values();
    State state = state_registry.insert_state(move(initial_facts));

    return bias->compute_heuristic(state);
}

bool is_number(const string& s) {
    return !s.empty() && find_if(s.begin(),
        s.end(), [](unsigned char c) { return !isdigit(c); }) == s.end();
}

const string &TechniqueGBackwardYaaig::get_name() const {
    return name;
}

TechniqueGBackwardYaaig::TechniqueGBackwardYaaig(const options::Options &opts)
        : SamplingTechnique(opts),
          technique(opts.get<string>("technique")),
          subtechnique(opts.get<string>("subtechnique")),
          bound(opts.get<string>("bound")),
          depth_k(opts.get<int>("depth_k")),
          unit_cost(opts.get<bool>("unit_cost")),
          allow_duplicates_interrollout(
              opts.get<string>("allow_duplicates") == "all" || opts.get<string>("allow_duplicates") == "interrollout"
          ),
          allow_duplicates_intrarollout(
              opts.get<string>("allow_duplicates") == "all"
          ),
          wrap_partial_assignment(opts.get<bool>("wrap_partial_assignment")),
          deprioritize_undoing_steps(opts.get<bool>("deprioritize_undoing_steps")),
          is_valid_walk(opts.get<bool>("is_valid_walk")),
          restart_h_when_goal_state(opts.get<bool>("restart_h_when_goal_state")),
          state_filtering(opts.get<string>("state_filtering")),
          bfs_percentage(opts.get<int>("bfs_percentage") / 100.0),
          bias_evaluator_tree(opts.get_parse_tree("bias", options::ParseTree())),
          bias_probabilistic(opts.get<bool>("bias_probabilistic")),
          bias_adapt(opts.get<double>("bias_adapt")),
          bias_reload_frequency(opts.get<int>("bias_reload_frequency")),
          bias_reload_counter(0) {
    if (technique == "bfs_rw") {
        // assert(subtechnique == "round_robin" || subtechnique == "random_leaf" || subtechnique == "percentage");
        if (!(subtechnique == "round_robin" || subtechnique == "random_leaf" || subtechnique == "percentage")) { utils::g_log << "Error: technique_gbackward_yaaig.cc:72" << endl; exit(0); }
    }
    if (technique == "dfs_rw") {
        // assert(subtechnique == "round_robin" || subtechnique == "random_leaf");
        if (!(subtechnique == "round_robin" || subtechnique == "random_leaf")) { utils::g_log << "Error: technique_gbackward_yaaig.cc:76" << endl; exit(0); }
    }
    // assert(bfs_percentage >= 0 && bfs_percentage <= 100);
    if (!(bfs_percentage >= 0 && bfs_percentage <= 100)) { utils::g_log << "Error: technique_gbackward_yaaig.cc:79" << endl; exit(0); }
}

vector<shared_ptr<PartialAssignment>> TechniqueGBackwardYaaig::sample_with_random_walk(
    PartialAssignment initial_state,
    const unsigned steps,
    const ValidStateDetector &is_valid_state,
    const PartialAssignmentBias *bias,
    const TaskProxy &task_proxy,
    const bool sample_initial_state,
    const bool global_hash_table,
    const utils::HashSet<PartialAssignment> states_to_avoid
) {
    OperatorsProxy ops = task_proxy.get_operators();
    PartialAssignment pa = initial_state;
    vector<shared_ptr<PartialAssignment>> samples;
    if (sample_initial_state) {
        samples.push_back(make_shared<PartialAssignment>(pa));
    }
    utils::HashSet<PartialAssignment> local_hash_table;
    utils::HashSet<PartialAssignment> *ht_pointer = global_hash_table ? &hash_table : &local_hash_table;
    bool renegerate_applicable_ops = true;
    while ((samples.size() < steps) && !stopped) {
        OperatorID applied_op = OperatorID::no_operator;
        PartialAssignment pa_ = rrws->sample_state_length(
            pa,
            1,
            applied_op,
            renegerate_applicable_ops,
            deprioritize_undoing_steps,
            is_valid_state,
            bias,
            bias_probabilistic,
            bias_adapt
        );
        // TODO: if op.index() != regression_op.index()
        // assert(
        //     (pa_ == pa && applied_op == OperatorID::no_operator) ||
        //     (pa_ != pa && applied_op != OperatorID::no_operator)
        // );
        if (!(
            (pa_ == pa && applied_op == OperatorID::no_operator) ||
            (pa_ != pa && applied_op != OperatorID::no_operator)
        )) { utils::g_log << "Error: technique_gbackward_yaaig.cc:120" << endl; exit(0); }
        if (pa_ == pa) // there is no applicable operator
            break;

        if ((allow_duplicates_intrarollout || ht_pointer->find(pa_) == ht_pointer->end())
            && (states_to_avoid.find(pa_) == states_to_avoid.end())) {
            // if it is goal state then set h to 0
            if (restart_h_when_goal_state && task_properties::is_goal_assignment(task_proxy, pa_))
                pa_.estimated_heuristic = 0;
            else
                pa_.estimated_heuristic = pa.estimated_heuristic + (unit_cost ? 1 : ops[applied_op].get_cost());

            ht_pointer->insert(pa_);
            samples.push_back(make_shared<PartialAssignment>(pa_));
            pa = pa_;
            renegerate_applicable_ops = true;
        } else {
            renegerate_applicable_ops = false;
        }
        stopped = stop_sampling();
    }
    // assert(samples.size() <= steps);
    if (!(samples.size() <= steps)) { utils::g_log << "Error: technique_gbackward_yaaig.cc:142" << endl; exit(0); }
    return samples;
}

vector<shared_ptr<PartialAssignment>> TechniqueGBackwardYaaig::sample_with_bfs_or_dfs(
    string technique,
    PartialAssignment initial_state,
    const unsigned steps,
    const ValidStateDetector &is_valid_state,
    const TaskProxy &task_proxy
) {
    OperatorsProxy ops = task_proxy.get_operators();
    PartialAssignment pa = initial_state;
    vector<shared_ptr<PartialAssignment>> samples;
    // Each element of the stack is (state, operator index used to achieve the state)
    stack<PartialAssignment> stack;
    queue<PartialAssignment> queue;

    if (technique == "dfs")
        stack.push(pa);
    else
        queue.push(pa);

    hash_table.insert(pa);
    while ((samples.size() < steps) && !stopped) {
        if ((technique == "dfs" && stack.empty()) || (technique == "bfs" && queue.empty()))
            break;
        if (technique == "dfs") {
            pa = stack.top();
            stack.pop();
        } else {
            pa = queue.front();
            queue.pop();
        }
        samples.push_back(make_shared<PartialAssignment>(pa));

        int idx_op = 0, rng_seed = (*rng)() * (INT32_MAX - 1);
        while (idx_op != -1) {
            OperatorID applied_op = OperatorID::no_operator;
            PartialAssignment pa_ = dfss->sample_state_length(
                pa,
                rng_seed,
                idx_op,
                applied_op,
                is_valid_state
            );
            // idx_op has the index of the operator that was used,
            // or -1 if all operators have already been tested
            // assert(
            //     (idx_op == -1 && applied_op == OperatorID::no_operator) ||
            //     (idx_op != -1 && applied_op != OperatorID::no_operator)
            // );
            if (!(
                (idx_op == -1 && applied_op == OperatorID::no_operator) ||
                (idx_op != -1 && applied_op != OperatorID::no_operator)
            )) { utils::g_log << "Error: technique_gbackward_yaaig.cc:197" << endl; exit(0); }
            if (idx_op == -1) {
                // assert(pa == pa_);
                if (!(pa == pa_)) { utils::g_log << "Error: technique_gbackward_yaaig.cc:200" << endl; exit(0); }
                break;
            }
            if (pa_ == pa)
                continue;

            if (allow_duplicates_intrarollout || hash_table.find(pa_) == hash_table.end()) {
                // if it is goal state then set h to 0
                if (restart_h_when_goal_state && task_properties::is_goal_assignment(task_proxy, pa_))
                    pa_.estimated_heuristic = 0;
                else
                    pa_.estimated_heuristic = pa.estimated_heuristic + (unit_cost ? 1 : ops[applied_op].get_cost());

                if (pa_.estimated_heuristic <= depth_k) {
                    hash_table.insert(pa_);
                    if (technique == "dfs")
                        stack.push(pa_);
                    else
                        queue.push(pa_);
                }
            }
            idx_op++;
            stopped = stop_sampling();
            if (stopped) break;
        }
    }
    // assert(samples.size() <= steps);
    if (!(samples.size() <= steps)) { utils::g_log << "Error: technique_gbackward_yaaig.cc:227" << endl; exit(0); }
    return samples;
}

vector<shared_ptr<PartialAssignment>> TechniqueGBackwardYaaig::sample_with_percentage_limited_bfs(
    float bfs_percentage,
    PartialAssignment initial_state,
    const ValidStateDetector &is_valid_state,
    vector<PartialAssignment> &leaves,
    const TaskProxy &task_proxy
) {
    // assert(bfs_percentage >= 0.0 && bfs_percentage <= 1.0);
    if (!(bfs_percentage >= 0.0 && bfs_percentage <= 1.0)) { utils::g_log << "Error: technique_gbackward_yaaig.cc:239" << endl; exit(0); }
    OperatorsProxy ops = task_proxy.get_operators();
    unsigned bfs_samples = (int)(bfs_percentage * max_samples);
    vector<PartialAssignment> vk = {initial_state}, vk1 = {}; // vector_k, vector_k+1
    vector<shared_ptr<PartialAssignment>> samples = {make_shared<PartialAssignment>(initial_state)};
    leaves.push_back(initial_state);

    while ((samples.size() < bfs_samples && !vk.empty()) && !stopped) {
        rng->shuffle(vk);
        for (PartialAssignment& s : vk) {
            vector<PartialAssignment> succ_s;
            int idx_op = 0, rng_seed = (*rng)() * (INT32_MAX - 1);
            while (!stopped) {
                OperatorID applied_op = OperatorID::no_operator;
                PartialAssignment s_ = dfss->sample_state_length(
                    s, rng_seed, idx_op, applied_op, is_valid_state
                );
                // assert(
                //     (idx_op == -1 && applied_op == OperatorID::no_operator) ||
                //     (idx_op != -1 && applied_op != OperatorID::no_operator)
                // );
                if (!(
                    (idx_op == -1 && applied_op == OperatorID::no_operator) ||
                    (idx_op != -1 && applied_op != OperatorID::no_operator)
                )) { utils::g_log << "Error: technique_gbackward_yaaig.cc:263" << endl; exit(0); }
                if (idx_op == -1)
                    break;
                if (find(succ_s.begin(), succ_s.end(), s_) == succ_s.end()
                    && (allow_duplicates_intrarollout || hash_table.find(s_) == hash_table.end())) {
                    // if it is goal state then set h to 0
                    if (restart_h_when_goal_state && task_properties::is_goal_assignment(task_proxy, s_))
                        s_.estimated_heuristic = 0;
                    else
                        s_.estimated_heuristic = s.estimated_heuristic + (unit_cost ? 1 : ops[applied_op].get_cost());

                    succ_s.push_back(s_);
                }
                idx_op++;
                stopped = stop_sampling(true, bfs_percentage);
            }
            if (samples.size() + succ_s.size() <= bfs_samples) {
                leaves.erase(find(leaves.begin(), leaves.end(), s));
                for (PartialAssignment& s_ : succ_s) {
                    samples.push_back(make_shared<PartialAssignment>(s_));
                    vk1.push_back(s_);
                    leaves.push_back(s_);
                    hash_table.insert(s_);
                }
            }
            if (stopped || samples.size() == bfs_samples)
                break;
            stopped = stop_sampling(true, bfs_percentage);
        }
        vk = vk1;
        vk1.clear();
    }
    stopped = false;
    return samples;
}

vector<shared_ptr<PartialAssignment>> TechniqueGBackwardYaaig::create_next_all(
        shared_ptr<AbstractTask> seed_task, const TaskProxy &task_proxy) {
    if (seed_task != last_task) {
        regression_task_proxy = make_shared<RegressionTaskProxy>(*seed_task);
        state_registry = make_shared<StateRegistry>(task_proxy);
        if (technique == "dfs" || technique == "bfs" || technique == "dfs_rw" || technique == "bfs_rw")
            dfss = make_shared<sampling::DFSSampler>(*regression_task_proxy, *rng);
        if (technique == "rw" || technique == "dfs_rw" || technique == "bfs_rw")
            rrws = make_shared<sampling::RandomRegressionWalkSampler>(*regression_task_proxy, *rng);
    }
    bias_reload_counter++;
    if (!bias_evaluator_tree.empty() &&
        (seed_task != last_task ||
         (bias_reload_frequency != -1 &&
          bias_reload_counter > bias_reload_frequency))) {
        options::OptionParser bias_parser(bias_evaluator_tree, *registry, *predefinitions, false);
        bias = bias_parser.start_parsing<shared_ptr<Heuristic>>();
        bias_reload_counter = 0;
        cache.clear();
    }

    PartialAssignmentBias *func_bias = nullptr;
    PartialAssignmentBias pab = [&](PartialAssignment &partial_assignment) {
        auto iter = cache.find(partial_assignment);
        if (iter != cache.end()) {
            return iter->second;
        } else {
            int h = compute_heuristic(task_proxy, bias.get(), *rng, partial_assignment);
            cache[partial_assignment] = h;
            return h;
        }
    };

    // assert(state_filtering != "statespace" || !sampling_engine::trie_statespace.empty());
    if (!(state_filtering != "statespace" || !sampling_engine::trie_statespace.empty())) { utils::g_log << "Error: technique_gbackward_yaaig.cc:337" << endl; exit(0); }

    auto is_valid_state = [&](PartialAssignment &partial_assignment) {
        if (state_filtering == "none") {
            return true;
        } else if (state_filtering == "mutex") {
            return !(is_valid_walk) || regression_task_proxy->convert_to_full_state(
                    partial_assignment, true, *rng).first;
        } else if (state_filtering == "statespace") {
            vector<int> key;
            for (char& b : partial_assignment.to_binary(true))
                key.push_back(b == '*' ? -1 : (int)b - '0');
            return sampling_engine::trie_statespace.find_all_compatible(key, "v_vu").size() > 0;
        }
        return false;
    };

    if (bias != nullptr) {
        func_bias = &pab;
    }

    // Hash table does not work for cases like:
    //   Atom on(b, a);Atom on(c, b);Atom on(d, c) and
    //   Atom on(b, a);Atom on(c, b);Atom on(d, c);(handempty)
    if (allow_duplicates_interrollout)
        hash_table.clear();

    PartialAssignment pa = regression_task_proxy->get_goal_assignment();
    pa.estimated_heuristic = 0;
    vector<shared_ptr<PartialAssignment>> samples;
    vector<PartialAssignment> leaves;

    if (samples_per_search == -1)
        samples_per_search = max_samples;

    float bound_n = -1;
    if (is_number(bound)) {
        bound_n = stoi(bound);
    } else {
        if (bound == "default") {
            if (technique == "dfs" || technique == "bfs")
                bound_n = depth_k;
            else // rw, bfs_rw, dfs_rw
                bound_n = samples_per_search;
        } else if (bound == "propositions") {
            bound_n = pa.to_binary().length();
        } else if (bound == "propositions_per_mean_effects") {
            int num_props = pa.to_binary().length();
            int num_effects = 0, num_ops = 0;
            for (OperatorProxy op : task_proxy.get_operators()) {
                num_ops++;
                for (EffectProxy eff : op.get_effects()) {
                    num_effects++;
                }
            }
            float mean_num_effects = (float)num_effects / num_ops;
            bound_n = (float)num_props / mean_num_effects;
        }
    }
    // assert(bound_n > 0);
    if (!(bound_n > 0)) { utils::g_log << "Error: technique_gbackward_yaaig.cc:380" << endl; exit(0); }
    bound_value = bound_n;

    if (technique == "rw" || technique == "bfs_rw" || technique == "dfs_rw")
        samples_per_search = ceil(bound_multiplier * bound_n);
    else if (technique == "dfs" || technique == "bfs")
        depth_k = ceil(bound_multiplier * bound_n);


    if (technique == "rw") {
        samples = sample_with_random_walk(pa, samples_per_search, is_valid_state, func_bias, task_proxy);

    } else if (technique == "bfs_rw" && subtechnique == "percentage") {
        samples = sample_with_percentage_limited_bfs(bfs_percentage, pa, is_valid_state, leaves, task_proxy);

    } else if (technique == "dfs" || technique == "bfs") {
        do {
            vector<shared_ptr<PartialAssignment>> samples_ = sample_with_bfs_or_dfs(
                technique.substr(0, 3), pa, max_samples-samples.size(), is_valid_state, task_proxy
            );
            samples.insert(samples.end(), samples_.begin(), samples_.end());
            hash_table.clear();
        } while ((samples.size() < (unsigned)max_samples) && !stopped);

    } else if (technique == "dfs_rw" || technique == "bfs_rw") {
        utils::g_log << technique << " (" << subtechnique << ") not implemented!" << endl;
        exit(0);
    }

    // pos-dfs/bfs random walk step
    if (technique == "dfs_rw" || technique == "bfs_rw") {
        if (leaves.size() <= 0) {
            utils::g_log << "The whole statespace was sampled. Skipping Random Walk." << endl;
            stopped = true;
            return samples;
        }

        utils::HashSet<PartialAssignment> bfs_core; // or dfs_core
        bool avoid_bfs_core = true;
        if (avoid_bfs_core) {
            for (shared_ptr<PartialAssignment> &s : samples) {
                bfs_core.insert(*s);
            }
        }

        utils::g_log << "Starting random walk search (" << subtechnique << ") from " << leaves.size() << " leaves" << endl;
        if (max_samples != numeric_limits<int>::max())
            utils::g_log << "Looking for " << (max_samples - samples.size()) << " more samples..." << endl;
        else
            utils::g_log << "Looking for more samples until mem/time budget runs out." << endl;

        int lid = 0;
        vector<bool> leaves_used(leaves.size(), false);
        while ((samples.size() < (unsigned)max_samples) && !stopped) {
            do {
                lid = (subtechnique == "round_robin") ? // round_robin or random_leaf (random_leaf if percentage)
                      (lid + 1) % leaves.size() : (int)((*rng)() * (INT32_MAX - 1)) % leaves.size();
            } while (leaves_used[lid]);
            leaves_used[lid] = true;
            if (all_of(leaves_used.begin(), leaves_used.end(), [](bool v) {return v;}))
                fill(leaves_used.begin(), leaves_used.end(), false);

            vector<shared_ptr<PartialAssignment>> samples_ = sample_with_random_walk(
                leaves[lid],
                min(samples_per_search - leaves[lid].estimated_heuristic, (int)(max_samples-samples.size())),
                is_valid_state,
                func_bias,
                task_proxy,
                false,
                !allow_duplicates_interrollout,
                bfs_core
            );
            samples.insert(samples.end(), samples_.begin(), samples_.end());
        }
    }
    return samples;
}

/* PARSING TECHNIQUE_GBACKWARD_YAAIG*/
static shared_ptr<TechniqueGBackwardYaaig> _parse_technique_gbackward_yaaig(
        options::OptionParser &parser) {
    SamplingTechnique::add_options_to_parser(parser);
    parser.add_option<string>(
            "technique",
            "Search technique (rw, dfs, bfs, dfs_rw, bfs_rw). "
            "If dfs_rw or bfs_rw and subtechnique != percentage, set depth_k.",
            "rw"
    );
    parser.add_option<string>(
            "subtechnique",
            "If dfs_rw or bfs_rw: round_robin, random_leaf, percentage",
            "random_leaf"
    );
    parser.add_option<string>(
            "bound",
            "How to bound each rollout: default, propositions, propositions_per_mean_effects, digit",
            "default"
    );
    parser.add_option<int>(
            "depth_k",
            "Maximum depth using the dfs/bfs algorithm. "
            "If it doesn't reach max_samples, complete with random walks of each leaf state.",
            "99999"
    );
    parser.add_option<bool>(
            "unit_cost",
            "Increments h by unit cost instead of operator cost.",
            "false"
    );
    parser.add_option<string>(
            "allow_duplicates",
            "Allow sample duplicated states in [all, interrollout, none]",
            "all"
    );
    parser.add_option<bool>(
            "wrap_partial_assignment",
            "If set, wraps a partial assignment obtained by the regression for the "
            "initial state into a task which has additional values for undefined "
            "variables. By default, the undefined variables are random uniformly "
            "set (satisfying the mutexes).",
            "false"
    );
    parser.add_option<bool>(
            "deprioritize_undoing_steps",
            "Deprioritizes actions which undo the previous action",
            "false"
    );
    parser.add_option<bool>(
            "is_valid_walk",
            "enforces states during random walk are avalid states w.r.t. "
            "the KNOWN mutexes",
            "true"
    );
    parser.add_option<bool>(
            "restart_h_when_goal_state",
            "Restart h value when goal state is sampled (only random walk)",
            "true"
    );
    parser.add_option<string>(
            "state_filtering",
            "Filtering of applicable operators (none, mutex, statespace)",
            "mutex"
    );
    parser.add_option<int>(
            "bfs_percentage",
            "Percentage of samples per BFS when technique=bfs_rw",
            "10"
    );
    parser.add_option<shared_ptr<Heuristic>>(
            "bias",
            "bias heuristic",
            "<none>"
    );
    parser.add_option<int>(
            "bias_reload_frequency",
            "the bias is reloaded everytime the tasks for which state are"
            "generated changes or if it has not been reloaded for "
            "bias_reload_frequency steps. Use -1 to prevent reloading.",
            "-1"
    );
    parser.add_option<bool>(
            "bias_probabilistic",
            "uses the bias values as weights for selecting the next state"
            "on the walk. Otherwise selects a random state among those with "
            "maximum bias",
            "true"
    );
    parser.add_option<double>(
            "bias_adapt",
            "if using the probabilistic bias, then the bias values calculated"
            "for the successors s1,..., sn of the state s are adapted as "
            "bias_adapt^(b(s1) - b(s)). This gets right of the issue that for"
            "large bias values, there was close to no difference between the "
            "states probabilities and focuses more on states increasing the bias.",
            "-1"
    );
    parser.add_option<int>(
            "max_upgrades",
            "Maximum number of times this sampling technique can upgrade its"
            "parameters. Use -1 for infinite times.",
            "0"
    );
    options::Options opts = parser.parse();

    shared_ptr<TechniqueGBackwardYaaig> technique;
    if (!parser.dry_run()) {
        technique = make_shared<TechniqueGBackwardYaaig>(opts);
    }
    return technique;
}

static Plugin<SamplingTechnique> _plugin_technique_gbackward_yaaig(
        TechniqueGBackwardYaaig::name, _parse_technique_gbackward_yaaig);
}
