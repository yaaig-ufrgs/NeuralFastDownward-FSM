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

    cout << prefix << " ";
    for (unsigned i = 0; i < relevant_facts.size(); i++) {
        cout << task->get_fact_name(relevant_facts[i]);
        cout << (i < relevant_facts.size()-1 ? ";" : "\n");
    }
}

BlindPrintSearchHeuristic::~BlindPrintSearchHeuristic() {
}

int BlindPrintSearchHeuristic::compute_heuristic(const State &ancestor_state) {
    /* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
     * Comment lines 172 and 173 of
     * src/search/search_egnines/eager_search.cc
     * to print the entire state space
     * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

    State state = convert_ancestor_state(ancestor_state);

    vector<int> values = state.get_values();
    vector<int> bin;
    for (unsigned i = 0; i < relevant_facts.size(); i++)
        bin.push_back(values[relevant_facts[i].var] == relevant_facts[i].value ? 1 : 0);

    // binary to vector<long long> decimals
    // bin: 111111111110101100101111111010101001101010101010101101010110101001
    // split, max 63 bits: 11 1111111110101100101111111010101001101010101010101101010110101001
    // vector decimals: 3 18423310914320782761
    vector<unsigned long long> decimals;
	unsigned long long decimal = 0, base = 1;
    int bits = 0;
    for (int i = bin.size()-1; i >= 0; i--) {
        decimal += (bin[i] == 1) ? base : 0;
        base *= 2;
        if (++bits >= 64 || i == 0) {
            decimals.insert(decimals.begin(), decimal);
            bits = 0;
            base = 1;
            decimal = 0;
        }
    }
    cout << prefix;
    for (size_t i = 0; i < decimals.size(); i++)
        cout << " " << decimals[i];
    cout << endl;

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
