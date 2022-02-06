#include "cached_search_heuristic.h"

#include "../option_parser.h"
#include "../plugin.h"

#include "../task_utils/task_properties.h"
#include "../utils/logging.h"

#include <cstddef>
#include <limits>
#include <utility>
#include <fstream>

using namespace std;

namespace cached_search_heuristic {
CachedSearchHeuristic::CachedSearchHeuristic(const Options &opts)
    : Heuristic(opts) {
    utils::g_log << "Initializing cached search heuristic..." << endl;

    max_value = 0;
    ifstream f(opts.get<string>("cache_file"));
    assert(f.is_open());
    string line;
    while (getline(f, line)) {
        int cut = line.find(";");
        string state = line.substr(cut+1);
        int value = stoi(line.substr(0, cut));
        if (value > max_value)
            max_value = value;
        if (h.find(state) == h.end())
            h[state] = value;
        else
            assert(value == h[state]);
    }
    f.close();
}

CachedSearchHeuristic::~CachedSearchHeuristic() {
}

int CachedSearchHeuristic::compute_heuristic(const State &ancestor_state) {
    State state = convert_ancestor_state(ancestor_state);
    string state_bin = state.to_binary();
    if (h.find(state_bin) != h.end())
        return h[state_bin];
    return max_value + 1;
}

static shared_ptr<Heuristic> _parse(OptionParser &parser) {
    parser.document_synopsis("Cached heuristic",
                             "Returns cost of state on sampling. "
                             "If not sampled, returns MAX_H+1.");

    Heuristic::add_options_to_parser(parser);
    parser.add_option<string>(
            "cache_file",
            "File for state value pairs. (each state on a line in \"h;binary\" format."
    );

    Options opts = parser.parse();
    if (parser.dry_run())
        return nullptr;
    else
        return make_shared<CachedSearchHeuristic>(opts);
}

static Plugin<Evaluator> _plugin("cached", _parse);
}
