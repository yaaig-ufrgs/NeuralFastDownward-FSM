#include "sampling_search_ferber.h"

#include "sampling_search_base.h"
#include "sampling_engine.h"

#include "../option_parser.h"
#include "../plugin.h"

#include "../task_utils/task_properties.h"

#include <sstream>
#include <string>

using namespace std;

namespace sampling_engine {

string SamplingSearchFerber::construct_header() const {
    ostringstream oss;
    oss << "# <Cost>;<State>";
    return oss.str();
}

string SamplingSearchFerber::sample_file_header() const {
    return header;
}

string SamplingSearchFerber::extract_single_sample(Trajectory trajectory, size_t idx_t, Plan plan, OperatorsProxy ops, int *cost) {
    ostringstream oss;

    if (store_plan_cost) {
        if (select_state_method == SelectStateMethod::ENTIRE_PLAN) {
            if (idx_t != trajectory.size() - 1)
                *cost += ops[plan[idx_t]].get_cost();
            oss << *cost << field_separator;
        }
        else {
            for (size_t i = idx_t; i-- > 0;) {
                if (i != trajectory.size() - 1)
                    *cost += ops[plan[i]].get_cost();
            } 
            oss << *cost << field_separator;
        }
    }

    if (store_state) {
        State state = engine->get_state_registry().lookup_state(trajectory[idx_t]);
        state.unpack();
        vector<int> values = state.get_values();
        for (const FactPair &fp: relevant_facts) {
            oss << (values[fp.var] == fp.value ? 1 : 0) << state_separator;
        }
        oss.seekp(-1, oss.cur);
        oss << field_separator;
    }

    if (store_operator) {
        // Select no operator for the goal state
        int next_op = idx_t >= plan.size() ?
                OperatorID::no_operator.get_index() :
                plan[idx_t].get_index();

        for (int idx = 0; idx < task->get_num_operators(); ++idx) {
            oss << (next_op == idx ? 1 : 0) << state_separator;
        }
        oss.seekp(-1, oss.cur);
        oss << field_separator;
    }

    string s = oss.str();
    s.pop_back();

    return s;
}

vector<string> SamplingSearchFerber::extract_samples() {
    vector<string> samples;

    OperatorsProxy ops = task_proxy.get_operators();
    Trajectory trajectory;
    engine->get_search_space().trace_path(engine->get_goal_state(), trajectory);
    Plan plan = engine->get_plan();
    assert(plan.size() == trajectory.size() - 1);
    // Sequence:    s0 a0 s1 a1 s2
    // Plan:        a0 a1
    // Trajectory:  s0 s1 s2

    int cost = 0;
    // RANDOM_STATE
    if (select_state_method == SelectStateMethod::RANDOM_STATE) {
        size_t idx_t = (*rng)(trajectory.size());
        string sample = extract_single_sample(trajectory, idx_t, plan, ops, &cost);
        samples.push_back(sample);

    }
    // ENTIRE_PLAN
    else if (select_state_method == SelectStateMethod::ENTIRE_PLAN) {
        for (size_t idx_t = trajectory.size(); idx_t-- > 0;) {
            string sample = extract_single_sample(trajectory, idx_t, plan, ops, &cost);
            samples.push_back(sample);
        }
    }
    // INIT_STATE
    else if (select_state_method == SelectStateMethod::INIT_STATE) {
        size_t idx_t = trajectory.size() - 1;
        string sample = extract_single_sample(trajectory, idx_t, plan, ops, &cost);
        samples.push_back(sample);
    }

    return samples;
}

SelectStateMethod select_state_method_format(const string &sel_state_method) {
    if (sel_state_method == "random_state") {
        return SelectStateMethod::RANDOM_STATE;
    } else if (sel_state_method == "entire_plan") {
        return SelectStateMethod::ENTIRE_PLAN;
    } else if (sel_state_method == "init_state") {
        return SelectStateMethod::INIT_STATE;
    }
    cerr << "Invalid select state format:" << sel_state_method << endl;
    utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
}

SamplingSearchFerber::SamplingSearchFerber(const options::Options &opts)
    : SamplingSearchBase(opts),
      select_state_method(select_state_method_format(
          opts.get<string>("select_state_method"))),
      store_plan_cost(opts.get<bool>("store_plan_cost")),
      store_state(opts.get<bool>("store_state")),
      store_operator(opts.get<bool>("store_operator")),
      relevant_facts(task_properties::get_strips_fact_pairs(task.get())),
      header(construct_header()){
}


static shared_ptr<SearchEngine> _parse_sampling_search_ferber(OptionParser &parser) {
    parser.document_synopsis("Sampling Search Manager", "");

    sampling_engine::SamplingSearchBase::add_sampling_search_base_options(parser);
    sampling_engine::SamplingEngine::add_sampling_options(parser);
    sampling_engine::SamplingStateEngine::add_sampling_state_options(
            parser, "fields", "pddl", ";", ";");

    parser.add_option<string>(
            "select_state_method",
            "Method to select states along the plans. Choose from:\n\
             * random_state - select a random state from the plans\n\
             * entire_plan - select all the state from the plans\n\
             * init_state - select the initial state from the plans",
            "random_state");
    parser.add_option<bool>(
            "store_plan_cost",
            "Store for every state its cost along the plan to the goal",
            "true");

    parser.add_option<bool>(
            "store_state",
            "Store every state along the plan",
            "true");

    parser.add_option<bool>(
            "store_operator",
            "Store for every state along the plan the next chosen operator",
            "false");

    SearchEngine::add_options_to_parser(parser);
    Options opts = parser.parse();
    shared_ptr<sampling_engine::SamplingSearchFerber> engine;
    if (!parser.dry_run()) {
        engine = make_shared<sampling_engine::SamplingSearchFerber>(opts);
    }

    return engine;
}

static Plugin<SearchEngine> _plugin_search("sampling_search_ferber", _parse_sampling_search_ferber);

}
