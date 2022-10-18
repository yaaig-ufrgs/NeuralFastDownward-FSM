#include "blind_print_search_heuristic.h"

#include "../option_parser.h"
#include "../plugin.h"

#include "../task_utils/task_properties.h"
#include "../utils/logging.h"

#include <cstddef>
#include <limits>
#include <utility>

using namespace std;

namespace blind_print_search_heuristic {
BlindPrintSearchHeuristic::BlindPrintSearchHeuristic(const Options &opts)
    : Heuristic(opts),
      relevant_facts(task_properties::get_strips_fact_pairs(task.get())),
      min_operator_cost(task_properties::get_min_operator_cost(task_proxy)) {
    utils::g_log << "Initializing blind print search heuristic..." << endl;

    int curVar = 0;
    cout << prefix << "#";
    for (unsigned i = 0; i < relevant_facts.size(); i++) {
        if (curVar != relevant_facts[i].var) {
            cout << ";";
            curVar = relevant_facts[i].var;
        }
        cout << task->get_fact_name(relevant_facts[i]);
        cout << (i < relevant_facts.size()-1 ? ";" : "\n");
    }
}

BlindPrintSearchHeuristic::~BlindPrintSearchHeuristic() {
}

int BlindPrintSearchHeuristic::compute_heuristic(const State &ancestor_state) {
    State state = convert_ancestor_state(ancestor_state);

    cout << prefix << state.to_binary() << endl;

    if (task_properties::is_goal_state(task_proxy, state))
        return 0;
    else
        return min_operator_cost;
}

static shared_ptr<Heuristic> _parse(OptionParser &parser) {
    parser.document_synopsis("Blind Print heuristic",
                             "Returns cost of cheapest action for "
                             "non-goal states, "
                             "0 for goal states");
    parser.document_language_support("action costs", "supported");
    parser.document_language_support("conditional effects", "supported");
    parser.document_language_support("axioms", "supported");
    parser.document_property("admissible", "yes");
    parser.document_property("consistent", "yes");
    parser.document_property("safe", "yes");
    parser.document_property("preferred operators", "no");

    Heuristic::add_options_to_parser(parser);
    Options opts = parser.parse();
    if (parser.dry_run())
        return nullptr;
    else
        return make_shared<BlindPrintSearchHeuristic>(opts);
}

static Plugin<Evaluator> _plugin("blind_print", _parse);
}
