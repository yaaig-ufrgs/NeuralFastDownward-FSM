#ifndef NEURAL_NETWORKS_TORCH_SAMPLING_NETWORK_H
#define NEURAL_NETWORKS_TORCH_SAMPLING_NETWORK_H

#include "torch_network.h"

#include "../heuristic.h"

namespace neural_networks {

class TorchSamplingNetwork : public TorchNetwork {
protected:
    const std::vector<FactPair> relevant_facts;
    const int heuristic_shift;
    const int heuristic_multiplier;
    const bool heuristic_null;

    int last_h = Heuristic::NO_VALUE;
    std::vector<int> last_h_batch;


    virtual std::vector<at::Tensor> get_input_tensors(const State &state) override;
    virtual void parse_output(const torch::jit::IValue &output) override;
    virtual void clear_output() override;
public:
    explicit TorchSamplingNetwork(const Options &opts);
    TorchSamplingNetwork(const TorchSamplingNetwork &orig) = delete;
    virtual ~TorchSamplingNetwork() override;

    virtual bool is_heuristic() override;
    virtual int get_heuristic() override;
    virtual const std::vector<int> &get_heuristics() override;
};
}
#endif /* NEURAL_NETWORKS_TORCH_SAMPLING_NETWORK_H */
