#include <stack>

#include "technique_gbackward_yaaig.h"

#include "../evaluation_result.h"
#include "../heuristic.h"
#include "../plugin.h"

#include "../tasks/modified_init_goals_task.h"
#include "../tasks/partial_state_wrapper_task.h"

#include "../task_utils/sampling.h"

#include "../task_utils/task_properties.h"

#define RW_MAX_ATTEMPTS 100

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
          bias_evaluator_tree(opts.get_parse_tree("bias", options::ParseTree())),
          bias_probabilistic(opts.get<bool>("bias_probabilistic")),
          bias_adapt(opts.get<double>("bias_adapt")),
          bias_reload_frequency(opts.get<int>("bias_reload_frequency")),
          bias_reload_counter(0) {
    if (technique == "bfs_rw" || technique == "dfs_rw")
        assert(subtechnique == "round_robin" || subtechnique == "round_robin_fashion" || subtechnique == "random_leaf");
}

vector<shared_ptr<PartialAssignment>> TechniqueGBackwardYaaig::sample_with_random_walk(
    PartialAssignment initial_state,
    const ValidStateDetector &is_valid_state,
    const PartialAssignmentBias *bias,
    const TaskProxy &task_proxy
) {
    PartialAssignment pa = initial_state;
    vector<shared_ptr<PartialAssignment>> samples = {
        make_shared<PartialAssignment>(pa)
    };
    // Attempts to find a new state when performing each step
    int attempts = 0;
    while (samples.size() < (unsigned)samples_per_search) {
        PartialAssignment pa_ = rrws->sample_state_length(
            pa,
            1,
            deprioritize_undoing_steps,
            is_valid_state,
            bias,
            bias_probabilistic,
            bias_adapt
        );
        if (pa_ == pa) // there is no applicable operator
            break;

        if (allow_duplicates_intrarollout || hash_table.find(pa_) == hash_table.end()) {
            // if it is goal state then set h to 0
            pa_.estimated_heuristic = (
                restart_h_when_goal_state &&
                task_properties::is_goal_assignment(task_proxy, pa_)
            ) ? 0 : pa.estimated_heuristic + 1;

            hash_table.insert(pa_);
            samples.push_back(make_shared<PartialAssignment>(pa_));
            pa = pa_;
            attempts = 0;
        } else if (++attempts >= RW_MAX_ATTEMPTS) {
            break;
        }
    }
    return samples;
}

vector<shared_ptr<PartialAssignment>> TechniqueGBackwardYaaig::sample_with_bfs_or_dfs(
    string technique,
    PartialAssignment initial_state,
    const ValidStateDetector &is_valid_state
) {
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
    while (samples.size() < (unsigned)samples_per_search) {
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

        int idx_op = 0;
        int rng_seed = (*rng)(INT32_MAX - 1);
        while (idx_op != -1) {
            PartialAssignment pa_ = dfss->sample_state_length(
                pa,
                rng_seed,
                idx_op,
                is_valid_state
            );
            // idx_op has the index of the operator that was used,
            // or -1 if all operators have already been tested
            if (idx_op == -1)
                break;

            if ((allow_duplicates_intrarollout && pa_ != pa && technique == "dfs") || hash_table.find(pa_) == hash_table.end()) {
                pa_.estimated_heuristic = pa.estimated_heuristic + 1; // TODO: non-unitary operator

                if (pa_.estimated_heuristic <= depth_k) {
                    hash_table.insert(pa_);
                    if (technique == "dfs")
                        stack.push(pa_);
                    else
                        queue.push(pa_);
                }
            }
            idx_op++;
        }
    }
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
    
    auto is_valid_state = [&](PartialAssignment &partial_assignment) {
        return !(is_valid_walk) || regression_task_proxy->convert_to_full_state(
                partial_assignment, true, *rng).first;
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

    if (samples_per_search == -1)
        samples_per_search = max_samples;

    int bound_n = -1;

    if (is_number(bound)) {
        bound_n = stoi(bound);
    } else {
        if (bound == "default") {
            if (technique == "dfs" || technique == "bfs")
                bound_n = depth_k;
            else
                bound_n = samples_per_search;
        } else if (bound == "propositions") {
            // TODO
        } else if (bound == "propositions_per_mean_effects") {
            // TODO
        }
    }

    assert(bound_n != -1);

    // TODO Recheck this.
    if (technique == "rw" || technique == "bfs_rw")
        samples_per_search = ceil(bound_multiplier * bound_n);
    else if (technique == "dfs" || technique == "bfs")
        depth_k = ceil(bound_multiplier * bound_n);

    if (technique == "rw") {
        samples = sample_with_random_walk(pa, is_valid_state, func_bias, task_proxy);

    } else if (technique == "dfs" || technique == "bfs" || technique == "dfs_rw" || technique == "bfs_rw") {
        samples = sample_with_bfs_or_dfs(technique.substr(0, 3), pa, is_valid_state);
        if (technique.substr(technique.size() - 2) == "rw") { // dfs_rw or bfs_rw
            vector<PartialAssignment> leaves, original_leaves;
            int leaf_h = 0;
            for (shared_ptr<PartialAssignment> pa : samples) {
                if (pa->estimated_heuristic > leaf_h) {
                    leaf_h = pa->estimated_heuristic;
                    original_leaves.clear();
                    leaves.clear();
                }
                if (pa->estimated_heuristic == leaf_h) {
                    original_leaves.push_back(*pa);
                    leaves.push_back(*pa);
                }
            }
            if (subtechnique == "random_leaf") {
                leaves.clear();
                int lid = (*rng)(INT32_MAX - 1) % original_leaves.size();
                leaves.push_back(original_leaves[lid]);
            }

            vector<bool> dead_leaf = vector<bool>(leaves.size(), false);
            vector<utils::HashSet<PartialAssignment>> hash_table_leaf(leaves.size());

            cout << "Starting random walk search from " << leaves.size() << " leaves (depth = " << leaf_h << ")" << endl;
            cout << "Looking for " << (samples_per_search - samples.size()) << " more samples..." << endl;

            while (samples.size() < (unsigned)samples_per_search) {
                bool all_leaves_dead = true;
                for (unsigned i = 0; i < leaves.size(); i++) {
                    if (dead_leaf[i])
                        continue;
                    // Adapted from RW code
                    PartialAssignment pa = leaves[i];
                    unsigned attempts;
                    for (attempts = 0; attempts < RW_MAX_ATTEMPTS; attempts++) {
                        PartialAssignment pa_ = rrws->sample_state_length(
                            pa,
                            1,
                            deprioritize_undoing_steps,
                            is_valid_state,
                            func_bias,
                            bias_probabilistic,
                            bias_adapt
                        );
                        if (pa_ == pa) {
                            attempts = RW_MAX_ATTEMPTS;
                            break;
                        }

                        if (allow_duplicates_intrarollout || hash_table_leaf[i].find(pa_) == hash_table_leaf[i].end()) {
                            // if it is goal state then set h to 0
                            pa_.estimated_heuristic = (
                                restart_h_when_goal_state &&
                                task_properties::is_goal_assignment(task_proxy, pa_)
                            ) ? 0 : pa.estimated_heuristic + 1;

                            hash_table_leaf[i].insert(pa_);
                            samples.push_back(make_shared<PartialAssignment>(pa_));
                            leaves[i] = pa_;
                            break;
                        }
                    }
                    if (attempts == RW_MAX_ATTEMPTS)
                        dead_leaf[i] = true;
                    if (!dead_leaf[i])
                        all_leaves_dead = false;
                    if (samples.size() >= (unsigned)samples_per_search)
                        break;
                }

                if (all_leaves_dead) {
                    if (subtechnique == "round_robin") {
                        break;
                    } else if (subtechnique == "round_robin_fashion") {
                        for (unsigned i = 0; i < leaves.size(); i++) {
                            leaves[i] = original_leaves[i];
                            dead_leaf[i] = false;
                            hash_table_leaf[i].clear();
                        }
                    } else if (subtechnique == "random_leaf") {
                        int lid = (*rng)(INT32_MAX - 1) % original_leaves.size();
                        leaves[0] = original_leaves[lid];
                        dead_leaf[0] = false;
                        hash_table_leaf[0].clear();
                    }
                }
            }
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
            "Search technique (rw, dfs, bfs, dfs_rw, bfs_rw). If dfs_rw or bfs_rw, set depth_k.",
            "rw"
    );
    parser.add_option<string>(
            "subtechnique",
            "If dfs_rw or bfs_rw: round_robin, round_robin_fashion, random_leaf",
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
