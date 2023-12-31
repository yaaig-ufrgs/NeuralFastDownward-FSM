#include "rng_options.h"

#include "rng.h"

#include "../options/option_parser.h"

using namespace std;

namespace utils {
void add_rng_options(options::OptionParser &parser) {
    parser.add_option<int>(
        "random_seed",
        "Set to -1 (default) to use the global random number generator. "
        "Set to any other value to use a local random number generator with "
        "the given seed.",
        "-1",
        options::Bounds("-1", "infinity"));
}

std::mt19937 &get_global_mt19937() {
    static std::mt19937 twister(2011);
    return twister;
}

shared_ptr<RandomNumberGenerator> parse_rng_from_options(
    const options::Options &options) {
    int seed = options.get<int>("random_seed");
    if (seed == -1) {
        // Use an arbitrary default seed.
        static shared_ptr<utils::RandomNumberGenerator> rng =
            make_shared<utils::RandomNumberGenerator>();
        return rng;
    } else {
        return make_shared<RandomNumberGenerator>(seed);
    }
}
}
