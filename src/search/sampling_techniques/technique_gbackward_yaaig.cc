#include <stack>

#include "technique_gbackward_yaaig.h"

#include "../evaluation_result.h"
#include "../heuristic.h"
#include "../plugin.h"

#include "../tasks/modified_init_goals_task.h"
#include "../tasks/partial_state_wrapper_task.h"

#include "../task_utils/sampling.h"

#include "../task_utils/task_properties.h"

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

const string &TechniqueGBackwardYaaig::get_name() const {
    return name;
}

TechniqueGBackwardYaaig::TechniqueGBackwardYaaig(const options::Options &opts)
        : SamplingTechnique(opts),
          technique(opts.get<string>("technique")),
          depth_k(opts.get<int>("depth_k")),
          allow_duplicates(opts.get<bool>("allow_duplicates")),
          wrap_partial_assignment(opts.get<bool>("wrap_partial_assignment")),
          deprioritize_undoing_steps(opts.get<bool>("deprioritize_undoing_steps")),
          is_valid_walk(opts.get<bool>("is_valid_walk")),
          restart_h_when_goal_state(opts.get<bool>("restart_h_when_goal_state")),
          allow_internal_rollout_duplicates(opts.get<bool>("allow_internal_rollout_duplicates")),
          bias_evaluator_tree(opts.get_parse_tree("bias", options::ParseTree())),
          bias_probabilistic(opts.get<bool>("bias_probabilistic")),
          bias_adapt(opts.get<double>("bias_adapt")),
          bias_reload_frequency(opts.get<int>("bias_reload_frequency")),
          bias_reload_counter(0) {
}

vector<shared_ptr<PartialAssignment>> TechniqueGBackwardYaaig::create_next_all(
        shared_ptr<AbstractTask> seed_task, const TaskProxy &task_proxy) {
    if (seed_task != last_task) {
        regression_task_proxy = make_shared<RegressionTaskProxy>(*seed_task);
        state_registry = make_shared<StateRegistry>(task_proxy);
        if (technique == "dfs" || technique == "bfs")
            dfss = make_shared<sampling::DFSSampler>(*regression_task_proxy, *rng);
        else if (technique == "rw")
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
    hash_table.clear();
    
    PartialAssignment partial_assignment = regression_task_proxy->get_goal_assignment();
    partial_assignment.estimated_heuristic = 0;
    vector<shared_ptr<PartialAssignment>> samples;

    if (technique == "dfs" || technique == "bfs") {
        // Each element of the stack is (state, operator index used to achieve the state)
        PartialAssignment pa = partial_assignment;
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
            while (idx_op != -1) {
                PartialAssignment pa_ = dfss->sample_state_length(
                    pa,
                    (*rng)(INT16_MAX),
                    idx_op,
                    is_valid_state
                );
                // idx_op has the index of the operator that was used,
                // or -1 if all operators have already been tested

                if (idx_op != -1 &&
                   (allow_internal_rollout_duplicates || hash_table.find(pa_) == hash_table.end())) {
                    pa_.estimated_heuristic = pa.estimated_heuristic + 1; // TODO: non-unitary operator

                    hash_table.insert(pa_);
                    if (technique == "dfs")
                        stack.push(pa_);
                    else
                        queue.push(pa_);
                }
            }
        }
    } else if (technique == "rw") {
        samples.push_back(make_shared<PartialAssignment>(partial_assignment));
        // Attempts to find a new state when performing each step
        int MAX_ATTEMPTS = 100, attempts = 0;
        while (samples.size() < (unsigned)samples_per_search) {
            PartialAssignment new_partial_assignment = rrws->sample_state_length(
                partial_assignment,
                1,
                deprioritize_undoing_steps,
                is_valid_state,
                func_bias,
                bias_probabilistic,
                bias_adapt
            );

            if (allow_internal_rollout_duplicates || hash_table.find(new_partial_assignment) == hash_table.end()) {
                hash_table.insert(new_partial_assignment);

                // if it is goal state then set h to 0
                new_partial_assignment.estimated_heuristic = (
                    restart_h_when_goal_state && task_properties::is_goal_state(
                        task_proxy,
                        new_partial_assignment.get_full_state(true, *rng).second)
                    ) ? 0 : partial_assignment.estimated_heuristic + 1;

                samples.push_back(make_shared<PartialAssignment>(new_partial_assignment));
                partial_assignment = new_partial_assignment;
                attempts = 0;
            } else if (++attempts >= MAX_ATTEMPTS) {
                break;
            }
        }
    }

    if (!allow_duplicates) {
        for (int i = samples.size()-1; i >= 0; i--) {
            if (unique_samples.find(*samples[i]) == unique_samples.end()) {
                unique_samples.insert(*samples[i]);
            } else {
                samples.erase(samples.begin()+i);
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
    parser.add_option<int>(
            "depth_k",
            "Maximum depth using the dfs/bfs algorithm. "
            "If it doesn't reach max_samples, complete with random walks of each leaf state.",
            "99999"
    );
    parser.add_option<bool>(
            "allow_duplicates",
            "Allow sample duplicated states in two different rollouts. "
            "If false, duplicate states are removed after each rollout.",
            "true"
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
    parser.add_option<bool>(
            "allow_internal_rollout_duplicates",
            "If false, states already seen within the same rollout are ignored.",
            "false"
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
