#ifndef TASK_UTILS_OPERATOR_GENERATOR_INTERNALS_H
#define TASK_UTILS_OPERATOR_GENERATOR_INTERNALS_H

#include "../operator_id.h"

#include <memory>
#include <unordered_map>
#include <vector>

class State;

namespace operator_generator {
class GeneratorBase {
public:
    virtual ~GeneratorBase() {}

    virtual OperatorID generate_min_applicable_op(const std::vector<int> &state, bool reject_unassigned = false) const = 0;
    virtual void generate_applicable_ops(
        const std::vector<int> &state, std::vector<OperatorID> &applicable_ops, bool reject_unassigned = false) const = 0;
};

class GeneratorForkBinary : public GeneratorBase {
    std::unique_ptr<GeneratorBase> generator1;
    std::unique_ptr<GeneratorBase> generator2;
public:
    GeneratorForkBinary(
        std::unique_ptr<GeneratorBase> generator1,
        std::unique_ptr<GeneratorBase> generator2);

    virtual OperatorID generate_min_applicable_op(const std::vector<int> &state, bool reject_unassigned = false) const override;
    virtual void generate_applicable_ops(
        const std::vector<int> &state, std::vector<OperatorID> &applicable_ops, bool reject_unassigned = false) const override;
};

class GeneratorForkMulti : public GeneratorBase {
    std::vector<std::unique_ptr<GeneratorBase>> children;
public:
    explicit GeneratorForkMulti(std::vector<std::unique_ptr<GeneratorBase>> children);

    virtual OperatorID generate_min_applicable_op(const std::vector<int> &state, bool reject_unassigned = false) const override; 
    virtual void generate_applicable_ops(
        const std::vector<int> &state, std::vector<OperatorID> &applicable_ops, bool reject_unassigned = false) const override;
};

class GeneratorSwitchBase : public GeneratorBase {
protected:
    int switch_var_id;
    const bool covers_all_values;
public:
    explicit GeneratorSwitchBase(int switch_var_id, bool covers_all_values);
};

class GeneratorSwitchVector : public GeneratorSwitchBase {
    std::vector<std::unique_ptr<GeneratorBase>> generator_for_value;
public:
    explicit GeneratorSwitchVector(
        int switch_var_id,
        std::vector<std::unique_ptr<GeneratorBase>> &&generator_for_value,
        bool covers_all_values = false);

    virtual OperatorID generate_min_applicable_op(const std::vector<int> &state, bool reject_unassigned = false) const override;
    virtual void generate_applicable_ops(
        const std::vector<int> &state, std::vector<OperatorID> &applicable_ops, bool reject_unassigned = false) const override;
};

class GeneratorSwitchHash : public GeneratorSwitchBase {
    std::unordered_map<int, std::unique_ptr<GeneratorBase>> generator_for_value;
public:
    explicit GeneratorSwitchHash(
        int switch_var_id,
        std::unordered_map<int, std::unique_ptr<GeneratorBase>> &&generator_for_value,
        bool covers_all_values = false);

    virtual OperatorID generate_min_applicable_op(const std::vector<int> &state, bool reject_unassigned = false) const override;
    virtual void generate_applicable_ops(
        const std::vector<int> &state, std::vector<OperatorID> &applicable_ops, bool reject_unassigned = false) const override;
};

class GeneratorSwitchSingle : public GeneratorSwitchBase {
    int value;
    std::unique_ptr<GeneratorBase> generator_for_value;
public:
    explicit GeneratorSwitchSingle(
        int switch_var_id, int value,
        std::unique_ptr<GeneratorBase> generator_for_value,
        bool covers_all_values = false);

    virtual OperatorID generate_min_applicable_op(const std::vector<int> &state, bool reject_unassigned = false) const override;
    virtual void generate_applicable_ops(
        const std::vector<int> &state, std::vector<OperatorID> &applicable_ops, bool reject_unassigned = false) const override;
};

class GeneratorLeafVector : public GeneratorBase {
    std::vector<OperatorID> applicable_operators;
public:
    explicit GeneratorLeafVector(std::vector<OperatorID> &&applicable_operators);

    virtual OperatorID generate_min_applicable_op(const std::vector<int> &state, bool reject_unassigned = false) const override;
    virtual void generate_applicable_ops(
        const std::vector<int> &state, std::vector<OperatorID> &applicable_ops, bool reject_unassigned = false) const override;
};

class GeneratorLeafSingle : public GeneratorBase {
    OperatorID applicable_operator;
public:
    explicit GeneratorLeafSingle(OperatorID applicable_operator);

    virtual OperatorID generate_min_applicable_op(const std::vector<int> &state, bool reject_unassigned = false) const override;
    virtual void generate_applicable_ops(
        const std::vector<int> &state, std::vector<OperatorID> &applicable_ops, bool reject_unassigned = false) const override;
};
}

#endif
