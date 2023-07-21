#ifndef PDBS_PATTERN_GENERATOR_HSTAR_H
#define PDBS_PATTERN_GENERATOR_HSTAR_H

#include "pattern_generator.h"
#include "types.h"

namespace options {
class Options;
}

namespace pdbs {
class PatternGeneratorHstar : public PatternGenerator {
    Pattern pattern;
public:
    explicit PatternGeneratorHstar(const options::Options &opts);
    virtual ~PatternGeneratorHstar() = default;

    virtual PatternInformation generate(const std::shared_ptr<AbstractTask> &task) override;
};
}

#endif
