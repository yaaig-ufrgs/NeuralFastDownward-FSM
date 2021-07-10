#include "torch_sampling_network.h"

#include "abstract_network.h"

#include "../option_parser.h"
#include "../plugin.h"

#include "../task_utils/task_properties.h"

#include <algorithm>
#include <fstream>
#include <memory>
#include <string>

using namespace std;
namespace neural_networks {

TorchSamplingNetwork::TorchSamplingNetwork(const Options &opts)
    : TorchNetwork(opts),
      relevant_facts(task_properties::get_strips_fact_pairs(task.get())),
      // Check _parse() to set these values:
      heuristic_shift(opts.get<int>("shift")),
      heuristic_multiplier(opts.get<int>("multiplier")),
      blind(opts.get<bool>("blind")) {}

TorchSamplingNetwork::~TorchSamplingNetwork() {}


bool TorchSamplingNetwork::is_heuristic() {
    return true;
}

int TorchSamplingNetwork::get_heuristic() {
    return last_h;
}

const vector<int> &TorchSamplingNetwork::get_heuristics() {
    return last_h_batch;
}

vector<at::Tensor> TorchSamplingNetwork::get_input_tensors(const State &state) {
    state.unpack();
    const vector<int> &values = state.get_values();

    at::Tensor tensor = torch::ones({1, static_cast<long>(relevant_facts.size())});
    auto accessor = tensor.accessor<float, 2>();
    size_t idx = 0;
    for (const FactPair &fp: relevant_facts) {
        accessor[0][idx++] = values[fp.var] == fp.value;
    }
    return {tensor};
}

void TorchSamplingNetwork::parse_output(const torch::jit::IValue &output) {
    at::Tensor tensor = output.toTensor();
    if (!blind) {
        auto accessor = tensor.accessor<float, 1>();
        for (int64_t i = 0; i < tensor.size(0); ++i) {
          // last_h = (accessor[i][0]+heuristic_shift) * heuristic_multiplier; // OLD originally the accessor had 2 dims
          last_h = (accessor[i] + heuristic_shift) * heuristic_multiplier;
          last_h_batch.push_back(last_h);
        }
    }
    else {
        for (int64_t i = 0; i < tensor.size(0); ++i) {
            last_h = 0;
            last_h_batch.push_back(last_h);
        }
    }
}

void TorchSamplingNetwork::clear_output() {
    last_h = Heuristic::NO_VALUE;
    last_h_batch.clear();
}
}

static shared_ptr<neural_networks::AbstractNetwork> _parse(OptionParser &parser) {
    parser.document_synopsis(
        "Torch State Network",
        "Takes a trained PyTorch model and evaluates it on a given state."
        "The output is read as and provided as a heuristic.");
    neural_networks::TorchNetwork::add_options_to_parser(parser);
    parser.add_option<int>(
            "shift",
            "shift the predicted heuristic value (useful, if the model"
            "output is expected to be negative up to a certain bound.", "0");
    parser.add_option<int>(
            "multiplier",
            "Multiply the predicted (and shifted) heuristic value (useful, if "
            "the model predicts small float values, but heuristics have to be "
            "integers", "1");
    parser.add_option<bool>(
            "blind",
            "Use heuristic = 0 to simulate a blind search.", "false");

    Options opts = parser.parse();

    shared_ptr<neural_networks::TorchSamplingNetwork> network;
    if (!parser.dry_run()) {
        network = make_shared<neural_networks::TorchSamplingNetwork>(opts);
    }

    return network;
}

static Plugin<neural_networks::AbstractNetwork> _plugin("torch_sampling_network", _parse);
