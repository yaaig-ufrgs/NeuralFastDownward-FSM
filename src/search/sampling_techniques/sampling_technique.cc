#include "sampling_technique.h"

#include "../evaluator.h"
#include "../evaluation_context.h"
#include "../option_parser.h"
#include "../plugin.h"
#include "../state_registry.h"
#include "../utils/system.h"
#include <limits>

#include <fstream>
#include <sstream>
#include <unordered_map>

using namespace std;


namespace sampling_technique {
/*
 * Use modified_task as transformed task in heuristic (and now also sampling).
 * We use this to instantiate search engines with our desired modified tasks.
 */
std::shared_ptr<AbstractTask> modified_task = nullptr;
std::vector<std::shared_ptr<PartialAssignment>> modified_tasks = std::vector<std::shared_ptr<PartialAssignment>>();

int random_samples = 0;
string random_estimates = "default";

static shared_ptr<AbstractTask> _parse_sampling_transform(
        options::OptionParser &parser) {
    if (parser.dry_run()) {
        return nullptr;
    } else {
        if (modified_task == nullptr) {
            return tasks::g_root_task;
        } else {
            return modified_task;
        }
    }
}

static Plugin<AbstractTask> _plugin_sampling_transform(
        "sampling_transform", _parse_sampling_transform);


const string SamplingTechnique::no_dump_directory = "false";
int SamplingTechnique::next_id = 0;

//REDUNDANCY
void check_magic(istream &in, const string &magic) {
    string word;
    in >> word;
    if (word != magic) {
        cerr << "Failed to match magic word '" << magic << "'." << endl
             << "Got '" << word << "'." << endl;
        if (magic == "begin_version") {
            cerr << "Possible cause: you are running the planner "
                 << "on a translator output file from " << endl
                 << "an older version." << endl;
        }
        utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
    }
}

//REDUNDANCY
bool are_facts_mutex(const vector<vector<set<FactPair>>> &mutexes,
                     const FactPair &fact1,
                     const FactPair &fact2) {
    if (fact1.var == fact2.var) {
        // Same variable: mutex iff different value.
        return fact1.value != fact2.value;
    }
    assert(utils::in_bounds(fact1.var, mutexes));
    assert(utils::in_bounds(fact1.value, mutexes[fact1.var]));
    return bool(mutexes[fact1.var][fact1.value].count(fact2));
}

static vector<vector<string>> load_mutexes(const string &path) {
    if (path == "none") {
        return vector<vector<string>>();
    }

    ifstream in;
    in.open(path);
    if (!in) {
        cerr << "Unable to open external mutex file: " << path << endl;
        utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
    }

    //Compare mutex loading to root_task.cc
    int num_mutex_groups;
    in >> num_mutex_groups;
    vector<vector<string>> mutexes(num_mutex_groups);
    for (int i = 0; i < num_mutex_groups; ++i) {
        check_magic(in, "begin_mutex_group");

        int num_facts;
        in >> num_facts;
        in >> ws;
        mutexes[i].reserve(num_facts);

        for (int j = 0; j < num_facts; ++j) {
            string atom;
            getline(in, atom);
            mutexes[i].push_back(atom);
        }
        check_magic(in, "end_mutex_group");
    }
    in.close();
    return mutexes;
}

/* START DEFINITION SAMPLING_TECHNIQUE */
SamplingTechnique::SamplingTechnique(const options::Options &opts)
        : id(next_id++),
          registry(opts.get_registry()),
          predefinitions(opts.get_predefinitions()),
          searches(opts.get<int>("searches")),
          samples_per_search(opts.get<int>("samples_per_search")),
          max_samples(opts.get<int>("max_samples")),
          limit_reached(false),
          unit_cost(opts.get<bool>("unit_cost")),
          random_percentage(opts.get<double>("random_percentage")),
          random_estimates(opts.get<string>("random_estimates")),
          regression_depth_multiplier(opts.get<double>("regression_depth_multiplier")),
          max_time(opts.get<double>("max_time")),
          mem_limit_mb(opts.get<int>("mem_limit_mb")),
          remove_duplicates(opts.get<bool>("remove_duplicates")),
          // dump_directory(opts.get<string>("dump")),
          check_mutexes(opts.get<bool>("check_mutexes")),
          check_solvable(opts.get<bool>("check_solvable")),
          use_alternative_mutexes(opts.get<string>("mutexes") != "none"),
          alternative_mutexes(load_mutexes(opts.get<string>("mutexes"))),
          eval_parse_tree(opts.get_parse_tree("evals")),
          option_parser(utils::make_unique_ptr<OptionParser>(
                  eval_parse_tree,
                  *opts.get_registry(), *opts.get_predefinitions(),
                  false, false)),
          remaining_upgrades(opts.get<int>("max_upgrades", 0)),
          rng(utils::parse_rng_from_options(opts)) {

    for (const shared_ptr<Evaluator> &e:
            option_parser->start_parsing<vector<shared_ptr<Evaluator>>>()) {
        if (!e->dead_ends_are_reliable()) {
            cout << "Warning: A given dead end detection evaluator is not safe."
                 << endl;
        }
    }

    if (max_samples == -1)
        max_samples = numeric_limits<int>::max();
    if (searches == -1)
        searches = numeric_limits<int>::max();

    assert(random_percentage >= 0.0 && random_percentage <= 1.0);

    if (random_percentage == 1.0) {
        sampling_technique::random_samples = max_samples;
        max_samples = 1;
    } else if (random_percentage > 0.0) {
        sampling_technique::random_samples = int(max_samples * random_percentage);
        max_samples -= sampling_technique::random_samples;
    }

    sampling_technique::random_estimates = random_estimates;

    if (mem_limit_mb != -1)
        mem_limit = mem_limit_mb * 1024; // KB
    else
        mem_limit = numeric_limits<int>::max();

    if (max_time == -1.0)
        max_time = numeric_limits<double>::max();
    sampling_timer = utils::make_unique_ptr<utils::CountdownTimer>(max_time);

}

SamplingTechnique::SamplingTechnique(
        int searches,
        // string dump_directory,
        bool check_mutexes,
        bool check_solvable, mt19937 &mt)
        : id(next_id++),
            registry(nullptr),
            predefinitions(nullptr),
            searches(searches),
            samples_per_search(-1),
            max_samples(-1),
            unit_cost(false),
            regression_depth_multiplier(1.0),
            max_time(-1.0),
            mem_limit_mb(-1),
            remove_duplicates(false),
            // dump_directory(move(dump_directory)),
            check_mutexes(check_mutexes),
            check_solvable(check_solvable),
            use_alternative_mutexes(false),
            alternative_mutexes(),
            eval_parse_tree(options::generate_parse_tree("[]")),
            option_parser(nullptr),
            remaining_upgrades(0),
            rng(make_shared<utils::RandomNumberGenerator>(mt)) {}

SamplingTechnique::~SamplingTechnique() {}

int SamplingTechnique::get_count() const {
    return searches;
}

int SamplingTechnique::get_counter() const {
    return counter;
}

bool SamplingTechnique::empty() const {
    return counter >= searches || limit_reached;
}

bool SamplingTechnique::stop_sampling(bool is_bfs, float bfs_pct) const {
    bool stop = false;
    if (!is_bfs) {
        stop = (sampling_timer->is_expired() ||
                utils::get_curr_memory_in_kb() >= mem_limit);
    } else {
        double time_reserved_bfs = max_time * bfs_pct;
        int mem_reserved_bfs = (mem_limit - mem_presampling) * bfs_pct + mem_presampling;
        stop = (sampling_timer->get_elapsed_time() >= time_reserved_bfs ||
                utils::get_curr_memory_in_kb() >= mem_reserved_bfs);
    }
    if (stop) {
        cout << "[STOP] Time/memory limit breached. Stopping " << (is_bfs ? "BFS" : "") << "sampling..." << endl
             << "[STOP] Current time:   " << sampling_timer->get_elapsed_time() << "/" << max_time << "s" << endl
             << "[STOP] Current memory: " << utils::get_curr_memory_in_kb() << "/" << mem_limit << endl;
    }
    return stop;
}

int SamplingTechnique::mem_usage_mb() const {
    return (utils::get_curr_memory_in_kb() / 1024);
}

shared_ptr<AbstractTask> SamplingTechnique::next(
        const shared_ptr<AbstractTask> &seed_task) {
    return next(seed_task, TaskProxy(*seed_task));
}

shared_ptr<AbstractTask> SamplingTechnique::next(
        const shared_ptr<AbstractTask> &seed_task,
        const TaskProxy &task_proxy) {
    if (empty()) {
        return nullptr;
    } else {
        update_alternative_task_mutexes(seed_task);
        counter++;
        while (true) {
            shared_ptr<AbstractTask> next_task = create_next(
                    seed_task, task_proxy);
            modified_task = next_task;
            if ((check_mutexes && !test_mutexes(next_task)) ||
                (check_solvable && !test_solvable(
                        TaskProxy(*next_task)))) {
                cout << "Generated task invalid, try anew." << endl;
                continue;
            }
            last_task = seed_task;
            return next_task;
        }
    }
}

vector<shared_ptr<PartialAssignment>> SamplingTechnique::next_all(
        const shared_ptr<AbstractTask> &seed_task) {
    vector<shared_ptr<PartialAssignment>> samples;
    utils::HashSet<PartialAssignment> hash_table;
    while (!empty()) {
        update_alternative_task_mutexes(seed_task);
        vector<shared_ptr<PartialAssignment>> next_samples = create_next_all(seed_task, TaskProxy(*seed_task));
        for (shared_ptr<PartialAssignment>& sample : next_samples) {
            if (remove_duplicates) {
                if (hash_table.count(*sample))
                    continue;
                hash_table.insert(*sample);
            }

            samples.push_back(sample);
            if (max_samples != numeric_limits<int>::max() && samples.size() >= (unsigned)max_samples) {
                limit_reached = true;
                break;
            }
        }
        counter++;
    }
    return samples;
}

void SamplingTechnique::update_alternative_task_mutexes(
        const std::shared_ptr<AbstractTask> &task) {
    if (!use_alternative_mutexes || task == last_task) {
        return;
    }

    vector<FactPair> facts;
    unordered_map<string, int> names;
    for (int var = 0; var < task->get_num_variables(); ++var) {
        for (int val = 0; val < task->get_variable_domain_size(var); ++val) {
            facts.emplace_back(var, val);
            names[task->get_fact_name(facts.back())] = facts.size() - 1;
        }
    }

    vector<vector<set<FactPair>>> inconsistent_facts(task->get_num_variables());
    for (int i = 0; i < task->get_num_variables(); ++i)
        inconsistent_facts[i].resize(task->get_variable_domain_size(i));

    for (const vector<string> &mutex: alternative_mutexes) {
        for (const string &sfact1: mutex) {
            auto ifact1 = names.find(sfact1);
            if (ifact1 == names.end()) {
                continue;
            }
            int fact1 = ifact1->second;
            for (const string &sfact2: mutex) {
                auto ifact2 = names.find(sfact2);
                if (ifact2 == names.end()) {
                    continue;
                }
                int fact2 = ifact2->second;
                inconsistent_facts[facts[fact1].var][facts[fact1].value].insert(
                        FactPair(facts[fact2].var, facts[fact2].value));
            }
        }
    }
    alternative_task_mutexes = inconsistent_facts;
}

bool SamplingTechnique::has_upgradeable_parameters() const {
    return remaining_upgrades > 0 || remaining_upgrades == -1;
}

void SamplingTechnique::do_upgrade_parameters() {
    cerr << "Either this sampling technique has not upgradable parameters"
            "or the upgrade method has to be implemented." << endl;
    utils::exit_with(utils::ExitCode::SEARCH_CRITICAL_ERROR);
}

void SamplingTechnique::upgrade_parameters() {
    if (remaining_upgrades == -1) {
        do_upgrade_parameters();
    } else if (remaining_upgrades > 0) {
        do_upgrade_parameters();
        remaining_upgrades--;
    } else {
        cerr << "Either this sampling technique has not upgradable parameters"
                "or no upgrades are remaining." << endl;
        utils::exit_with(utils::ExitCode::SEARCH_CRITICAL_ERROR);
    }
}

void SamplingTechnique::dump_upgradable_parameters(ostream &/*stream*/) const {
    cerr << "Either this sampling technique has not upgradable parameters"
            "or the dump parameter method has to be implemented." << endl;
    utils::exit_with(utils::ExitCode::SEARCH_CRITICAL_ERROR);
}

bool SamplingTechnique::test_mutexes(const shared_ptr<AbstractTask> &task) const {
    //Check initial state
    vector<int> init = task->get_initial_state_values();
    for (size_t i = 0; i < init.size(); ++i) {
        FactPair fi(i, init[i]);
        for (size_t j = i + 1; j < init.size(); ++j) {
            FactPair fj(j, init[j]);
            if ((use_alternative_mutexes &&
                 are_facts_mutex(alternative_task_mutexes, fi, fj)) ||
                (!use_alternative_mutexes && task->are_facts_mutex(fi, fj))) {
                return false;
            }
        }
    }
    //Check goal facts
    for (int i = 0; i < task->get_num_goals(); ++i) {
        FactPair fi = task->get_goal_fact(i);
        for (int j = i + 1; j < task->get_num_goals(); ++j) {
            FactPair fj = task->get_goal_fact(j);
            if ((use_alternative_mutexes &&
                 are_facts_mutex(alternative_task_mutexes, fi, fj)) ||
                (!use_alternative_mutexes && task->are_facts_mutex(fi, fj))) {
                return false;
            }
        }
    }
    return true;
}

bool SamplingTechnique::test_solvable(const TaskProxy &task_proxy) const {
    if (option_parser == nullptr) {
        return true;
    }
    StateRegistry state_registry(task_proxy);
    const State &state = state_registry.get_initial_state();
    EvaluationContext eval_context(state);
    int round = 0;
    for (const shared_ptr<Evaluator> &e:
            option_parser->start_parsing<vector<shared_ptr<Evaluator>>>()) {
        EvaluationResult eval_result = e->compute_result(eval_context);
        if (eval_result.is_infinite()) {
            cout << "Task unsolvable said by evaluator: " << round << endl;
            return false;
        }
        round++;
    }
    return true;
}

//void SamplingTechnique::dump_modifications(
//        std::shared_ptr<AbstractTask> /*task*/) const {
//    if (dump_directory.empty() || dump_directory == "false") {
//        return;
//    }
//    //ostringstream path;
//    //path << dump_directory << "/" << counter << "-" << get_name();
//    //ostringstream content;
//    cout << "SamplingTechnique problem dump currently not supported" << endl;
//    utils::exit_with(utils::ExitCode::SEARCH_UNSUPPORTED);
//}

void SamplingTechnique::add_options_to_parser(options::OptionParser &parser) {
    parser.add_option<int>(
            "searches",
            "Number of times this sampling technique shall be used.",
            "-1"
    );
    parser.add_option<int>(
            "samples_per_search",
            "Maximum number of steps in each run.",
            "-1"
    );
    parser.add_option<int>(
            "max_samples",
            "Force searches until reaching max_samples samples."
            "If -1, max_samples is qual to numeric_limits<int>.",
            "-1"
    );
    parser.add_option<bool>(
            "unit_cost",
            "Increments h by unit cost instead of operator cost.",
            "false"
    );
    parser.add_option<double>(
            "random_percentage",
            "Percentage [0.0, 1.0] of randomly generated samples (h = L+1).",
            "0.0"
    );
    parser.add_option<string>(
            "random_estimates",
            "Estimate values for random samples:"
            "default, random, number",
            "default"
    );
    parser.add_option<double>(
            "regression_depth_multiplier",
            "Multiplies the bound of each rollout by the given value.",
            "1.0"
    );
    parser.add_option<double>(
            "max_time",
            "Max time to consider when doing sampling.",
            "-1.0"
    );
    parser.add_option<int>(
            "mem_limit_mb",
            "Memory limit to consider when sampling."
            "If -1, no limit is given.",
            "-1"
    );
    parser.add_option<bool>(
            "remove_duplicates",
            "Remove duplicated samples.",
            "false"
    );
    parser.add_list_option<shared_ptr<Evaluator>>(
            "evals",
            "evaluators for dead-end detection (use only save ones to not "
            "reject a non dead end). If any of the evaluators detects a dead "
            "end, the assignment is rejected. \nATTENTON: The evaluators are "
            "initialize for each task to test anew. To be initialized "
            "correctly, they need to have the attribute "
            "'transform=sampling_transform'", "[]");
    parser.add_option<string>(
            "mutexes",
            "Path to a file describing mutexes. Uses those mutexes instead of "
            "the task mutexes if given.",
            "none");
    parser.add_option<string>(
            "dump",
            "Path to a directory where to dump the modifications or \"false\" "
            "to not dump.",
            SamplingTechnique::no_dump_directory);
    parser.add_option<bool>("check_mutexes", "Boolean flag to set whether to "
                                             "to check that a generated task does not violate "
                                             "mutexes.", "true");
    parser.add_option<bool>("check_solvable", "Boolean flag to set whether "
                                              "the generated task is solvable.",
                            "true");
    utils::add_rng_options(parser);
}

vector<int> SamplingTechnique::extractInitialState(const State &state) {
    vector<int> values;
    values.reserve(state.size());
    for (int val : state.get_values()) {
        values.push_back(val);
    }
    return values;
}

vector<FactPair> SamplingTechnique::extractGoalFacts(
        const GoalsProxy &goals_proxy) {
    vector<FactPair> goals;
    goals.reserve(goals_proxy.size());
    for (size_t i = 0; i < goals_proxy.size(); i++) {
        goals.push_back(goals_proxy[i].get_pair());
    }
    return goals;
}

vector<FactPair> SamplingTechnique::extractGoalFacts(const State &state) {
   vector<FactPair> goals;
   goals.reserve(state.size());
   int var = 0;
   for (int val : state.get_values()) {
       goals.emplace_back(var, val);
       var++;
   }
   return goals;
}


static PluginTypePlugin<SamplingTechnique> _type_plugin(
        "SamplingTechnique",
        "Generates from a given task a new one.");
}

