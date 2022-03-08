#ifndef SAMPLING_TECHNIQUES_TECHNIQUE_GBACKWARD_YAAIG_H
#define SAMPLING_TECHNIQUES_TECHNIQUE_GBACKWARD_YAAIG_H

#include "sampling_technique.h"

#include "../task_proxy.h"

#include "../utils/distribution.h"

class StateRegistry;
class RegressionTaskProxy;
namespace sampling {
class RandomRegressionWalkSampler;
class DFSSampler;
}

namespace sampling_technique {
class TechniqueGBackwardYaaig : public SamplingTechnique {
protected:
    const std::string technique;
    const std::string subtechnique;
    const int depth_k;
    const bool allow_duplicates_interrollout;
    const bool allow_duplicates_intrarollout;
    const bool wrap_partial_assignment;
    const bool deprioritize_undoing_steps;
    const bool is_valid_walk;
    const bool restart_h_when_goal_state;
    const options::ParseTree bias_evaluator_tree;
    const bool bias_probabilistic;
    const double bias_adapt;
    utils::HashMap<PartialAssignment, int> cache;
    std::shared_ptr<Heuristic> bias = nullptr;
    const int bias_reload_frequency;
    int bias_reload_counter;
    std::shared_ptr<StateRegistry> state_registry = nullptr;
    std::shared_ptr<AbstractTask> last_partial_wrap_task = nullptr;
    std::shared_ptr<RegressionTaskProxy> regression_task_proxy = nullptr;
    std::shared_ptr<sampling::RandomRegressionWalkSampler> rrws = nullptr;
    std::shared_ptr<sampling::DFSSampler> dfss = nullptr;
    utils::HashSet<PartialAssignment> hash_table;

    virtual std::vector<std::shared_ptr<PartialAssignment>> create_next_all(
            std::shared_ptr<AbstractTask> seed_task,
            const TaskProxy &task_proxy) override;

    // virtual void do_upgrade_parameters() override ;

public:
    explicit TechniqueGBackwardYaaig(const options::Options &opts);
    virtual ~TechniqueGBackwardYaaig() override = default;

    // virtual void dump_upgradable_parameters(std::ostream &/*stream*/) const override;

    virtual const std::string &get_name() const override;
    const static std::string name;
};
}
#endif
