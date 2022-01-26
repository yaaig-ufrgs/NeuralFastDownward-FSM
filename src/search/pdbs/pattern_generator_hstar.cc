#include "pattern_generator_hstar.h"

#include "pattern_information.h"

#include "../option_parser.h"
#include "../plugin.h"
#include "../task_proxy.h"

#include "../utils/logging.h"

#include <iostream>

using namespace std;

namespace pdbs {
PatternGeneratorHstar::PatternGeneratorHstar(const Options &/*opts*/) {}

PatternInformation PatternGeneratorHstar::generate(
    const shared_ptr<AbstractTask> &task) {
    pattern.clear();
    TaskProxy task_proxy = TaskProxy(*task);
    for (VariableProxy var : task_proxy.get_variables()) {
        pattern.push_back(var.get_id());
    }

    PatternInformation pattern_info(TaskProxy(*task), move(pattern));
    utils::g_log << "Manual pattern: " << pattern_info.get_pattern() << endl;



    return pattern_info;
}

/* Add in pdb_heuristic.cc:

#include <fstream>
#include <unordered_map>
int PDBHeuristic::compute_heuristic(const State &ancestor_state) {
    State state = convert_ancestor_state(ancestor_state);
    vector<int> sample_values(relevant_facts.size(), 0);
    unordered_map<string, int> map;
    ifstream in_file("samples/yaaig_blocks_probBLOCKS-7-0_dfs_fs_500x200_ss1");
    ofstream out_file("hstar_from_yaaig_blocks_probBLOCKS-7-0_dfs_fs_500x200_ss1.csv");
    string line;
    while (getline(in_file, line)) {
        if (line[0] == '#')
            continue;
        string sample = line.substr(line.find_first_of(';')+1);
        assert(sample.size() == relevant_facts.size());
        for (unsigned i = 0; i < sample.size(); i++)
            if (sample[i] == '1')
                sample_values[relevant_facts[i].var] = relevant_facts[i].value;
        if (map.count(sample) == 0)
            map[sample] = pdb->get_value(sample_values);
    }
    for (auto &x : map)
        out_file << x.first << "," << x.second << endl;
    cout << "Done!" << endl;
    exit(0);
    return -1;
}
*/

static shared_ptr<PatternGenerator> _parse(OptionParser &parser) {
    // parser.add_list_option<int>(
    //     "pattern",
    //     "list of variable numbers of the planning task that should be used as "
    //     "pattern.");

    Options opts = parser.parse();
    if (parser.dry_run())
        return nullptr;

    return make_shared<PatternGeneratorHstar>(opts);
}

static Plugin<PatternGenerator> _plugin("hstar_pattern", _parse);
}
