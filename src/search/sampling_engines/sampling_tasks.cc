#include "sampling_tasks.h"

#include "../task_utils/successor_generator.h"

#include <memory>
#include <string>

using namespace std;

namespace sampling_engine {
SamplingTasks::SamplingTasks(const options::Options &opts)
    : SamplingEngine(opts) {}


string SamplingTasks::sample_file_header() const {
    return "";
}
vector<string> SamplingTasks::sample(vector<shared_ptr<AbstractTask>> tasks) {
    return vector<string> {tasks[0]->get_sas()};
}
}
