#include "pattern_generator_hstar.h"

#include "pattern_information.h"

#include "../option_parser.h"
#include "../plugin.h"
#include "../task_proxy.h"

#include "../utils/logging.h"

#include <iostream>

// usage: ./fast-downward.py problem.pddl --search "astar(pdb(hstar_pattern([])))"

using namespace std;

namespace pdbs {
PatternGeneratorHstar::PatternGeneratorHstar(const Options &opts)
    : pattern(opts.get_list<int>("pattern")) {
}

PatternInformation PatternGeneratorHstar::generate(
    const shared_ptr<AbstractTask> &task) {
    pattern.clear();
    TaskProxy task_proxy = TaskProxy(*task);
    for (VariableProxy var : task_proxy.get_variables()) {
        pattern.push_back(var.get_id());
    }

    PatternInformation pattern_info(TaskProxy(*task), move(pattern));
    utils::g_log << "h* pattern: " << pattern_info.get_pattern() << endl;
    return pattern_info;
}

static shared_ptr<PatternGenerator> _parse(OptionParser &parser) {
    parser.add_list_option<int>(
        "pattern",
        "list of variable numbers of the planning task that should be used as "
        "pattern.");

    Options opts = parser.parse();
    if (parser.dry_run())
        return nullptr;

    return make_shared<PatternGeneratorHstar>(opts);
}

static Plugin<PatternGenerator> _plugin("hstar_pattern", _parse);
}
