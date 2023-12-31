#include "regression_task_proxy.h"

#include "../task_utils/task_properties.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

using namespace std;

RegressionCondition::RegressionCondition(int var, int value)
    : data(FactPair(var, value)) { }

bool RegressionCondition::is_satisfied(const PartialAssignment &assignment) const {
    return !assignment.assigned(data.var) || data.value == assignment[data.var].get_value();
    //int current_val = assignment[data.var].get_value();
    //return current_val == PartialAssignment::UNASSIGNED || current_val == data.value;
}

RegressionConditionProxy::RegressionConditionProxy(const AbstractTask &task, const RegressionCondition &condition)
    : task(&task), condition(condition) { }

RegressionConditionProxy::RegressionConditionProxy(const AbstractTask &task, int var_id, int value)
    : RegressionConditionProxy(task, RegressionCondition(var_id, value)) { }

RegressionEffect::RegressionEffect(int var, int value)
    : data(FactPair(var, value)) { }

RegressionEffectProxy::RegressionEffectProxy(const AbstractTask &task, const RegressionEffect &effect)
    : task(&task), effect(effect) { }

RegressionEffectProxy::RegressionEffectProxy(const AbstractTask &task, int var_id, int value)
    : RegressionEffectProxy(task, RegressionEffect(var_id, value)) { }

RegressionOperator::RegressionOperator(OperatorProxy &op, const int op_idx, const int undefined_value)
    : original_index(op_idx == -1 ? op.get_id() : op_idx),
      cost(op.get_cost()),
      name(op.get_name()),
      is_an_axiom(op.is_axiom()) {
    /*
      1) pre(v) = x, eff(v) = y  ==>  rpre(v) = y, reff(v) = x
      2) pre(v) = x, eff(v) = -  ==>  rpre(v) = x, reff(v) = x
                                <=/=>  rpre(v) = x, reff(v) = -, because x could
                                be satisfied by unknown, and should afterwards
                                be x
      3) pre(v) = -, eff(v) = y  ==>  rpre(v) = y, reff(v) = u
      4) (do per partial assignment)
          exists v s.t. eff(v) inter partial assignment is not empty
     */
    unordered_set<int> precondition_vars;
    unordered_map<int, int> vars_to_effect_values;

    for (EffectProxy effect : op.get_effects()) {
        FactProxy fact = effect.get_fact();
        int var_id = fact.get_variable().get_id();
        vars_to_effect_values[var_id] = fact.get_value();
        original_effect_vars.insert(var_id);
    }

    // Handle cases 1 and 2 where preconditions are defined.
    for (FactProxy precondition : op.get_preconditions()) {
        int var_id = precondition.get_variable().get_id();
        precondition_vars.insert(var_id);
        effects.emplace_back(var_id, precondition.get_value());
        if (vars_to_effect_values.count(var_id)) {
            // Case 1, effect defined.
            preconditions.emplace_back(var_id, vars_to_effect_values[var_id]);
        } else {
            // Case 2, effect undefined.
            preconditions.emplace_back(var_id, precondition.get_value());
        }
    }

    bool VISITALL_REMOVE_RPRE_VISITED_HACK = false;

    // Handle case 3 where preconditions are undefined.
    for (EffectProxy effect : op.get_effects()) {
        FactProxy fact = effect.get_fact();
        int var_id = fact.get_variable().get_id();
        if (precondition_vars.count(var_id) == 0) {
            if (VISITALL_REMOVE_RPRE_VISITED_HACK) {
                assert(var_id != 0);           // is a 'visited' tile (not robot tile)
                assert(fact.get_value() == 0); // value is visited
            } else {
                preconditions.emplace_back(var_id, fact.get_value());
            }
            effects.emplace_back(var_id, undefined_value);
        }
    }
}

bool RegressionOperator::achieves_subgoal(const PartialAssignment &assignment) const {
    return any_of(original_effect_vars.begin(), original_effect_vars.end(),
                  [&] (int var_id) {
                      return assignment.assigned(var_id);
                  });
}

bool RegressionOperator::is_applicable(const PartialAssignment &assignment) const {
    return achieves_subgoal(assignment) &&
        all_of(preconditions.begin(), preconditions.end(),
            [&](const RegressionCondition &condition) {
                return condition.is_satisfied(assignment);
        });
}

inline shared_ptr<vector<RegressionOperator>> extract_regression_operators(const AbstractTask &task, TaskProxy &tp) {
    task_properties::verify_no_axioms(tp);
    task_properties::verify_no_conditional_effects(tp);

    bool VISITALL_WITHOUT_UNDEFINED_HACK = false;
    int op_idx = 0;

    auto rops = make_shared<vector<RegressionOperator>>();
    for (OperatorProxy op : OperatorsProxy(task)) {
        if (VISITALL_WITHOUT_UNDEFINED_HACK) {
            // For each operator whose effect is undefined, it generates
            // n operators with the n possibilities of values.
            // - Specific implementation for visitall!!!!
            // (undefined can become 2 values: 0=visited or 1=not visited)
            // - No need to worry about applying this rule wrongly to robot
            // atoms because only tile atoms come into case 3.
            OperatorProxy op2 = op;
            RegressionOperator o(op, op_idx, 0);
            rops->emplace_back(op, op_idx++, 0);
            RegressionOperator o2(op2, op_idx, 1);
            rops->emplace_back(op2, op_idx++, 1);
        } else {
            RegressionOperator o(op);
            rops->emplace_back(op);
        }
    }
    return rops;
}

RegressionTaskProxy::RegressionTaskProxy(const AbstractTask &task)
    : TaskProxy(task),
      operators(extract_regression_operators(task, *this)) { }

