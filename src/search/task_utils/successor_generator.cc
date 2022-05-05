#include "successor_generator.h"

#include "operator_generator_internals.h"
#include "successor_generator_factory.h"

using namespace std;

namespace successor_generator {
SuccessorGenerator::SuccessorGenerator(const TaskProxy &task_proxy)
    : root(SuccessorGeneratorFactory(task_proxy).create()) {
}

SuccessorGenerator::~SuccessorGenerator() = default;

void SuccessorGenerator::generate_applicable_ops(
    const State &state, vector<OperatorID> &applicable_ops, bool reject_unassigned) const {
    state.unpack();
    root->generate_applicable_ops(state.get_unpacked_values(), applicable_ops, reject_unassigned);
}

void SuccessorGenerator::generate_applicable_ops(
    const PartialAssignment &partialAssignment, vector<OperatorID> &applicable_ops, bool reject_unassigned) const {
    assert(partialAssignment.get_unpacked_values().size() > 0);
    root->generate_applicable_ops(partialAssignment.get_unpacked_values(), applicable_ops, reject_unassigned);
}

PerTaskInformation<SuccessorGenerator> g_successor_generators;

SuccessorGenerator &get_successor_generator(const TaskProxy &task_proxy) {
    utils::g_log << "Building successor generator..." << flush;
    int peak_memory_before = utils::get_peak_memory_in_kb();
    utils::Timer successor_generator_timer;
    SuccessorGenerator &successor_generator =
            g_successor_generators[task_proxy];
    successor_generator_timer.stop();
    utils::g_log << "done!" << endl;
    int peak_memory_after = utils::get_peak_memory_in_kb();
    int memory_diff = peak_memory_after - peak_memory_before;
    utils::g_log << "peak memory difference for successor generator creation: "
                 << memory_diff << " KB" << endl
                 << "time for successor generation creation: "
                 << successor_generator_timer << endl;
    return successor_generator;
}
}
