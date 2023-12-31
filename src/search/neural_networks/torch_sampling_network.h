#ifndef NEURAL_NETWORKS_TORCH_SAMPLING_NETWORK_H
#define NEURAL_NETWORKS_TORCH_SAMPLING_NETWORK_H

#include "torch_network.h"

#include "../heuristic.h"

namespace neural_networks {

class TorchSamplingNetwork : public TorchNetwork {
protected:
    //const std::string path_cmp; // Path to secondary network model used for comparison.
    const int heuristic_shift;
    const int heuristic_multiplier;
    const std::vector<FactPair> relevant_facts;
    const std::vector<int> default_input_values;
    const bool blind;
    const double unary_threshold;
    const bool undefined_input;

    bool normalize_output;
    int max_h;

    int last_h = Heuristic::NO_VALUE;
    std::vector<int> last_h_batch;


    virtual std::vector<at::Tensor> get_input_tensors(const State &state) override;
    virtual void parse_output(const torch::jit::IValue &output) override;
    virtual void parse_output_both(const torch::jit::IValue &output, const torch::jit::IValue &output_cmp) override;
    virtual void clear_output() override;
public:
    explicit TorchSamplingNetwork(const Options &opts);
    TorchSamplingNetwork(const TorchSamplingNetwork &orig) = delete;
    virtual ~TorchSamplingNetwork() override;

    virtual bool is_heuristic() override;
    virtual int get_heuristic() override;
    virtual const std::vector<int> &get_heuristics() override;

    int unary_to_value(const std::vector<double>& unary_h);
    int h_adjustment(int h);
};
}
#endif /* NEURAL_NETWORKS_TORCH_SAMPLING_NETWORK_H */
