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
    oss << "# <Cost>;<State>";
    return oss.str();
}

string SamplingSearchFukunaga::sample_file_header() const {
    return header;
}

vector<string> SamplingSearchFukunaga::extract_samples() {
    vector<string> samples;
    for (std::shared_ptr<AbstractTask>& task: sampling_technique::modified_tasks) {
        ostringstream oss;

        if (store_plan_cost) {
            oss << task->estimated_heuristic << field_separator;
        }

        if (store_state) {
            vector<int> values = task->get_initial_state_values();
            for (const FactPair &fp: relevant_facts) {
                oss << (values[fp.var] == fp.value ? 1 : 0) << state_separator;
            }
            oss.seekp(-1, oss.cur);
            oss << field_separator;
        }

        string s = oss.str();
        s.pop_back();

        samples.push_back(s);
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

SamplingSearchFukunaga::SamplingSearchFukunaga(const options::Options &opts)
    : SamplingSearchBase(opts),
      select_state_method(select_state_method_format(
          opts.get<string>("select_state_method"))),
      store_plan_cost(opts.get<bool>("store_plan_cost")),
      store_state(opts.get<bool>("store_state")),
      store_operator(opts.get<bool>("store_operator")),
      relevant_facts(task_properties::get_strips_fact_pairs(task.get())),
      header(construct_header()){
}


static shared_ptr<SearchEngine> _parse_sampling_search_fukunaga(OptionParser &parser) {
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
    shared_ptr<sampling_engine::SamplingSearchFukunaga> engine;
    if (!parser.dry_run()) {
        engine = make_shared<sampling_engine::SamplingSearchFukunaga>(opts);
    }

    return engine;
}

static Plugin<SearchEngine> _plugin_search("sampling_search_fukunaga", _parse_sampling_search_fukunaga);

}
