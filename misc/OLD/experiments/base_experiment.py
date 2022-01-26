#!/usr/bin/env python

from collections import defaultdict
import json
import logging
import math
import os
import os.path
import platform
import re
import sys
sys.path.append("/home/ferber/bin/lab")

from lab.environments import LocalEnvironment, BaselSlurmEnvironment

from downward.dynamic_experiment import DynamicFastDownwardExperiment, SkipFlag
from downward.reports.absolute import AbsoluteReport
from downward.reports.scatter import ScatterPlotReport

## DEFAULTS
EMAIL_ADDRESS = None  # add your mail address
ATTRIBUTES = ['coverage', 'total_time', 'expansions', 'error', 'cost',
              'plan_length']
EPSILON = 10 ** (-5)


## DOMAINS
DOMAIN_DIR_BLOCKSWORLD = "blocksworld_ipc"
DOMAIN_DIR_DEPOTS = "depot_fix_goals"
DOMAIN_DIR_GRID = "grid_fix_goals"
DOMAIN_DIR_NPUZZLE = "npuzzle_ipc"
DOMAIN_DIR_PIPESWORLD_NOTANKAGE = "pipesworld-notankage_fix_goals"
DOMAIN_DIR_ROVERS = "rovers"
DOMAIN_DIR_SCANALYZER_08 = "scanalyzer-08-strips"
DOMAIN_DIR_SCANALYZER_11 = "scanalyzer-opt11-strips"
DOMAIN_DIR_STORAGE = "storage"
DOMAIN_DIR_TRANSPORT = "transport-opt14-strips"
DOMAIN_DIR_VISITALL = "visitall-opt14-strips"

DOMAIN_DIRS_AAAI20_MAIN = [
    DOMAIN_DIR_BLOCKSWORLD, DOMAIN_DIR_DEPOTS, DOMAIN_DIR_PIPESWORLD_NOTANKAGE,
    DOMAIN_DIR_SCANALYZER_08, DOMAIN_DIR_SCANALYZER_11, DOMAIN_DIR_TRANSPORT
]

DOMAIN_DIRS_AAAI20_OTHER = [
    DOMAIN_DIR_GRID, DOMAIN_DIR_NPUZZLE, DOMAIN_DIR_ROVERS,
    DOMAIN_DIR_STORAGE, DOMAIN_DIR_VISITALL
]

DOMAIN_DIRS_AAAI20_MAIN_EXTENDED_EXTENDED = [
    DOMAIN_DIR_BLOCKSWORLD, DOMAIN_DIR_DEPOTS, DOMAIN_DIR_PIPESWORLD_NOTANKAGE,
    DOMAIN_DIR_SCANALYZER_08, DOMAIN_DIR_SCANALYZER_11, DOMAIN_DIR_TRANSPORT,
    DOMAIN_DIR_GRID, DOMAIN_DIR_STORAGE,
]

DOMAIN_DIRS_AAAI20_REDUCED_REDUCED = [
    DOMAIN_DIR_NPUZZLE, DOMAIN_DIR_ROVERS, DOMAIN_DIR_VISITALL
]

TASK_DIRS_AAAI20_OPT = set([
    'probBLOCKS-10-1', 'probBLOCKS-10-2', 'probBLOCKS-11-0', 'probBLOCKS-11-1',
    'probBLOCKS-11-2',
    'depot_p02', 'depot_p03', 'depot_p04', 'depot_p07',
    'pipes_nt_p02-net1-b6-g4', 'pipes_nt_p09-net1-b14-g6',
    'pipes_nt_p12-net2-b10-g4', 'pipes_nt_p06-net1-b10-g6',
    'scanalyzer08_p01', 'scanalyzer08_p02', 'scanalyzer08_p03',
    'scanalyzer08_p25',
    'scanalyzer11_p03', 'scanalyzer11_p02', 'scanalyzer11_p11',
    'scanalyzer11_p14',
    'transport_p01', 'transport_p02', 'transport_p07', 'transport_p13',
    'transport_p14',

])

TASK_DIRS_AAAI20_SAT = set([
    'probBLOCKS-12-0', 'probBLOCKS-14-0', 'probBLOCKS-15-0', 'probBLOCKS-15-1',
    'probBLOCKS-16-1', 'probBLOCKS-17-0',
    'depot_p05', 'depot_p08', 'depot_p09', 'depot_p10', 'depot_p11',
    'depot_p15', 'depot_p16', 'depot_p19', 'depot_p21',
    'grid_prob03', 'grid_prob04', 'grid_prob05',
    'pipes_nt_p19', 'pipes_nt_p19-net2-b18-g6', 'pipes_nt_p21-net3-b12-g2',
    'pipes_nt_p24-net3-b14-g5', 'pipes_nt_p27-net3-b18-g6',
    'pipes_nt_p28-net3-b18-g7', 'pipes_nt_p30', 'pipes_nt_p30-net3-b20-g8',
    'pipes_nt_p31', 'pipes_nt_p31-net4-b14-g3', 'pipes_nt_p32-net4-b14-g5',
    'pipes_nt_p34', 'pipes_nt_p34-net4-b16-g6', 'pipes_nt_p41',
    'pipes_nt_p41-net5-b22-g2',
    'scanalyzer08_p11', 'scanalyzer08_p12', 'scanalyzer08_p15',
    'scanalyzer08_p16', 'scanalyzer08_p18', 'scanalyzer08_p19',
    'scanalyzer08_p21', 'scanalyzer08_p27', 'scanalyzer08_p28',
    'scanalyzer08_p29', 'scanalyzer08_p30', 'scanalyzer08_p8',
    'scanalyzer11_p07', 'scanalyzer11_p10', 'scanalyzer11_p13',
    'scanalyzer11_p15', 'scanalyzer11_p16', 'scanalyzer11_p20',
    'scanalyzer_p18', 'scanalyzer_p19',
    'transport_p05', 'transport_p10', 'transport_p11', 'transport_p12',
    'transport_p16', 'transport_p17', 'transport_p18', 'transport_p19',
    'transport_p20'
])






MAP_SUPER_DOMAINS_TO_TASK_DIRS = {
    "depot": ["depot_"],
    "grid": ["grid_prob"],
    "npuzzle": ["npuzzle_prob_"],
    "pipesworld-nt": ["pipes_nt_"],
    "blocksworld": ["probBLOCKS-"],
    "rovers": ["rovers_p"],
    "scanalyzer": ["scanalyzer_p", "scanalyzer08_", "scanalyzer11_"],
    "storage": ["storage_p"],
    "transport": ["transport_p"],
    "visitall": ["visitall_p"],
}

MAP_TASK_DIR_PREFIX_TO_SUPER_DOMAINS = {
    task_dir: super_domain
    for super_domain, task_dirs in MAP_SUPER_DOMAINS_TO_TASK_DIRS.items()
    for task_dir in task_dirs
}


def get_super_domain_from_task_dir(task_dir):
    for k, v in MAP_TASK_DIR_PREFIX_TO_SUPER_DOMAINS.items():
        if task_dir.startswith(k):
            return v
    assert False


def domain_as_category(run1, run2):
    assert run1["domain"] == run2["domain"]
    for k, v in CATEGORIES.items():
        if run1["domain"].startswith(k):
            return v
    assert False


## FETCHER FILTERS
def filter_universe_aaai20_main(arg):
    if arg["domain"].find("scanalyzer11_p20") > -1 or arg["domain"].find("scanalyzer08_p28") > -1:
        return False
    if arg["domain"].find("_op4_gs") > -1:
        return False

    return any(arg["domain"].find(name) > -1 for name in
               ["probB", "blocks", "depot", "scanaly", "transport", "pipes_nt"])

def filter_universe_aaai20_main_extended_extended(arg):
    if arg["domain"].find("scanalyzer11_p20") > -1 or arg["domain"].find("scanalyzer08_p28") > -1:
        return False
    if arg["domain"].find("_op4_gs") > -1:
        return False

    return any(arg["domain"].find(name) > -1 for name in
               ["probB", "blocks", "depot", "scanaly", "transport", "pipes_nt",
                "grid_prob", "storage_p"])

AAAAI20_OTHER_BENCHMARK_TASKS = (
    ["grid_prob03", "grid_prob04"] +
    ["npuzzle_prob_n%i_%i" % (a, b) for a in [6, 7] for b in range(1, 5)] +
    ["rovers_p%i" % a for a in [11, 18, 20, 21, 22, 23, 26, 28, 29, 33]] +
    ["storage_p18"] +
    ["visitall_p-1-%i" % a for a in range(12, 18)]
)


AAAAI20_OTHER_BENCHMARK_TASKS_REDUCED_REDUCED = (
    ["npuzzle_prob_n%i_%i" % (a, b) for a in [6, 7] for b in range(1, 5)] +
    ["rovers_p%i" % a for a in [11, 18, 20, 21, 22, 23, 26, 28, 29, 33]] +
    ["visitall_p-1-%i" % a for a in range(12, 18)]
)
def filter_universe_aaai20_other(arg):
    return arg["domain"] in AAAAI20_OTHER_BENCHMARK_TASKS

def filter_universe_aaai20_other_reduced_reduced(arg):
    return arg["domain"] in AAAAI20_OTHER_BENCHMARK_TASKS_REDUCED_REDUCED


KEY_UNEXPLAINED_ERRORS = "unexplained_errors"
PATTERN_TENSORFLOW_WARNING = re.compile(r"(run\.err:\s*)?\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d\.\d+: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX\n")
PATTERN_SOFT_LIMIT_LOG = re.compile(r"(driver\.err:\s*)?\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d,\d+ ERROR\s+planner finished and wrote \d+ KiB to run.log \(soft limit: \d+ KiB\)\n")


def filter_some_fetcher_errors(run):
    if KEY_UNEXPLAINED_ERRORS in run:
        run[KEY_UNEXPLAINED_ERRORS] = [
            ue for ue in run[KEY_UNEXPLAINED_ERRORS]
            if not (PATTERN_TENSORFLOW_WARNING.match(ue) or
                    PATTERN_SOFT_LIMIT_LOG.match(ue))]
        if len(run[KEY_UNEXPLAINED_ERRORS]) == 0:
            del run[KEY_UNEXPLAINED_ERRORS]
    return run


def get_filter_algorithms(nn_prefixes, allow_non_nn_algorithms=True):
    def _filter(arg):
        if any(arg["algorithm"].startswith(x) for x in ["ocls", "cls", "reg"]):
            return arg["algorithm"] in nn_prefixes
        else:
            return allow_non_nn_algorithms
    return _filter

def fetcher_expansions_per_second(run):
    if run["coverage"]:
        if run["search_time"] == 0:
            run["search_time"] = EPSILON
        run["expansions_per_second"] = run["expansions"] / run["search_time"]
    return run

# Paths
REPO = os.environ["DEEPDOWN"]
BENCHMARKS_ROOT = os.path.join(os.environ["DEEPDOWN"], "data", "FixedWorlds", "opt")
REVISION_CACHE = os.path.expanduser('~/lab/revision-cache')
NO_BASELINE_DOMAINS = ["nomystery_fix"]

# Detect Environment to use
NODE = platform.node()
IS_REMOTE = None
if NODE.endswith(".scicore.unibas.ch") or NODE.endswith(".cluster.bc2.ch"):
    IS_REMOTE = True
else:
    IS_REMOTE = False

PATTERN_PROBLEM_IDX = re.compile(".*?(\d+)\.pddl")
PATTERN_HIDDEN_LAYERS = re.compile("_h(fix)?(\d+)_")





class Param(object):
    PREFIX = "prefix"
    NETWORK_TYPE = "network_type"
    INPUT_STATE = "input_state"
    INPUT_GOAL = "input_goal"
    OUTPUT_LAYER = "output_layer"
    FORMAT = "format"
    ATOMS = "atoms"
    INITS = "inits"


def get_param(prefix, types, starts_with=False):
    param = None
    for k, v in types.items():
        if ((starts_with and prefix.startswith(k)) or
                (not starts_with and prefix.find(k) > -1)):
            assert param is None
            param = v
    return param


def get_output_layer(_prefix, _network_type, _output_hidden_modificator=1):
    return "{{MODEL_OUTPUT_LAYER,model.pb}}"
    hidden = PATTERN_HIDDEN_LAYERS.findall(prefix)
    assert len(hidden) == 1, "%s, %s" % (str(hidden), prefix)
    hidden = int(hidden[0][1])
    assert prefix.find("hlsfix") == -1 and prefix.find("hld") == -1
    if network_type == REGRESSION:
        activation = "Relu"
    elif network_type == CLASSIFICATION:
        activation = "Softmax"
    elif network_type == ORDINAL_CLASSIFICATION:
        activation = "Sigmoid"
    else:
        assert False
    return "[dense_%i_1/%s]" % (hidden + output_hidden_modificator, activation)


def get_atoms_inits(prefix, sformat):
    atoms, inits = "PDDL_ATOMS", "PDDL_INITS"
    if sformat == "full":
        return atoms, inits
    elif sformat == "non_statics":
        return atoms + "_FLEXIBLE", inits + "_FLEXIBLE"

## START NETWORK TYPE STUFF
REGRESSION = "regression"
CLASSIFICATION = "classification"
ORDINAL_CLASSIFICATION = "ordinal_classification"

NETWORK_TYPES = {
    "reg": REGRESSION,
    "cls": CLASSIFICATION,
    "ocls": ORDINAL_CLASSIFICATION
}

def get_fd_network_type(type):
    return REGRESSION if type is REGRESSION else CLASSIFICATION

def has_unary_threshold(type):
    return type == ORDINAL_CLASSIFICATION
## END NETWORK TYPE STUFF

FORMATS = {"ns": "non_statics", "full": "full"}


def get_network_param_from_prefix(x, output_hidden_modificator=1):
    params = {}
    params[Param.PREFIX] = x

    params[Param.NETWORK_TYPE] = get_param(x, NETWORK_TYPES, starts_with=True)
    params[Param.INPUT_STATE] = "input_1_1"
    params[Param.INPUT_GOAL] = "input_2_1"
    params[Param.OUTPUT_LAYER] = get_output_layer(
        x, params[Param.NETWORK_TYPE],
        _output_hidden_modificator=output_hidden_modificator)
    params[Param.FORMAT] = get_param(x, FORMATS)
    atoms, inits = get_atoms_inits(x, params[Param.FORMAT])
    params[Param.ATOMS] = atoms
    params[Param.INITS] = inits
    return params


def convert_network_param_to_h_evolution_config(params):
    prefix = params[Param.PREFIX]
    if prefix[-1] == "_":
        prefix = prefix[:-1]
    return tuple([
        prefix,
        params[Param.NETWORK_TYPE],
        params[Param.INPUT_STATE],
        params[Param.INPUT_GOAL],
        params[Param.OUTPUT_LAYER],
        params[Param.ATOMS],
        params[Param.INITS],
    ])


def convert_network_param_to_search_configuration(params):
    prefix = params[Param.PREFIX]
    if prefix[-1] == "_":
        prefix = prefix[:-1]
    return tuple([
        prefix,
        prefix + "_{FOLD}_fold_model.pb",
        params[Param.OUTPUT_LAYER],
        params[Param.ATOMS],
        params[Param.INITS]
    ])


def get_domains_from_property_files(*path_property):
    domains = set()
    for path in path_property:
        assert os.path.exists(path)
        with open(path, "r") as f:
            properties = json.load(f)
            for key, entry in properties.items():
                assert "domain" in entry
                domains.add(entry["domain"])
    return domains

PROPERTY_DOMAIN = "domain"
PROPERTY_PROBLEM = "problem"
def get_domains_problems_from_property_files(*path_property, **kwargs):
    algorithms = kwargs.pop("algorithms")
    domains_problems = defaultdict(set)
    for path in path_property:
        assert os.path.exists(path)
        with open(path, "r") as f:
            properties = json.load(f)
            for key, entry in properties.items():
                assert PROPERTY_DOMAIN in entry
                assert PROPERTY_PROBLEM
                if (algorithms is not None and
                        entry["algorithm"] not in algorithms):
                    continue

                domains_problems[entry[PROPERTY_DOMAIN]].add(
                    entry[PROPERTY_PROBLEM])
    return domains_problems


def find_benchmark_dirs(benchmarks, remote, baseline,
                        filter_task_regenerated=None):
    benchmark_dirs = defaultdict(list)
    if remote:
        for domain in os.listdir(benchmarks):
            path_domain = os.path.join(benchmarks, domain)
            if baseline and domain in NO_BASELINE_DOMAINS:
                continue
            if not os.path.isdir(path_domain):
                continue
            for universe in os.listdir(path_domain):
                path_universe = os.path.join(path_domain, universe)
                if not os.path.isdir(path_universe):
                    continue
                # Test if dir contains a universe (TODO: Improve)
                path_universe_domain_file = os.path.join(path_universe,
                                                         "domain.pddl")
                if not os.path.isfile(path_universe_domain_file):
                    continue

                if filter_task_regenerated is not None:
                    path_regenerated = os.path.join(path_universe, filter_task_regenerated)
                    if not os.path.isfile(path_regenerated):
                        continue
                    with open(path_regenerated, "r") as f:
                        if len(json.load(f)) == 0:
                            continue

                benchmark_dirs[path_domain].append(universe)
    else:
        benchmark_dirs[
            os.path.join(benchmarks, "depot_fix_goals")].append(
            "depot_p15")
    return benchmark_dirs

def get_base_experiment(baseline,
                        mail=EMAIL_ADDRESS, attributes=ATTRIBUTES, remote=IS_REMOTE,
                        benchmarks=BENCHMARKS_ROOT,
                        filter_benchmarks=None, skip_domains=None,
                        skip_problems=None,
                        rev_cache=REVISION_CACHE,
                        cores=1,
                        extra_options=None, partition="infai_1",
                        fetcher_filter=None,
                        report_filter=None,
                        add_suite=True,
                        experiment_class=None,
                        environment_setup=None,
                        parsers=None,
                        filter_task_regenerated=None):
    """

    :param baseline:
    :param mail:
    :param attributes:
    :param remote:
    :param benchmarks:
    :param filter_benchmarks:
    :param skip_domains:
    :param skip_problems: {domain: set(problems)} to skip
    :param rev_cache:
    :param add_suite: if True:adds benchmarks detected to experiment
                      else: return exp, suites
    :return:
    """
    if filter_benchmarks is not None and isinstance(filter_benchmarks, str):
        filter_benchmarks = re.compile(filter_benchmarks)
    skip_domains = set() if skip_domains is None else skip_domains
    skip_problems = {} if skip_problems is None else skip_problems

    extra_options = "" if extra_options is None else ("%s\n" % extra_options)
    if cores != 1:
        extra_options = "%s%s" % (extra_options,
                                  '#SBATCH --cpus-per-task=%i' % cores)
    if remote:
        ENV = BaselSlurmEnvironment(
            email=mail,
            extra_options=extra_options,
            memory_per_cpu=None if cores == 1 else ("%iM" % int(3872 / cores)),
            partition=partition,
            setup=environment_setup
        )
    else:
        ENV = LocalEnvironment(processes=2)

    exp_args = {"environment": ENV}
    if experiment_class is None:
        experiment_class = DynamicFastDownwardExperiment
        exp_args["revision_cache"] = rev_cache
    exp = experiment_class(**exp_args)

    # Add built-in parsers to the experiment.
    if parsers is None:
        exp.add_parser(exp.EXITCODE_PARSER)
        exp.add_parser(exp.TRANSLATOR_PARSER)
        exp.add_parser(exp.SINGLE_SEARCH_PARSER)
        exp.add_parser(exp.PLANNER_PARSER)
    else:
        for parser in parsers:
            exp.add_parser(parser)

    benchmark_dirs = find_benchmark_dirs(benchmarks, remote, baseline,
                                         filter_task_regenerated)

    all_suits = []
    for benchmark_dir in benchmark_dirs:
        if filter_benchmarks is None or filter_benchmarks.match(benchmark_dir):
            benchmark_domains = benchmark_dirs[benchmark_dir]
            for domain in list(benchmark_domains):
                if domain in skip_domains:
                    benchmark_domains.remove(domain)
            if len(benchmark_domains) == 0:
                continue
            print(benchmark_dir)
            benchmark_domains_added = []
            relevant_pddls = []
            for benchmark_domain in benchmark_domains:
                path_domain = os.path.join(benchmark_dir, benchmark_domain)

                if filter_task_regenerated is not None:
                    path_regenerated = os.path.join(benchmark_dir, benchmark_domain, filter_task_regenerated)
                    assert os.path.isfile(path_regenerated), path_regenerated
                    with open(path_regenerated, 'r') as f:
                        tasks_regenerated = [os.path.basename(x) for x in json.load(f)]
                new_relevant = [
                    "%s:%s" %(benchmark_domain, x)
                    for x in os.listdir(path_domain)
                    if x.endswith(".pddl") and
                       not x.endswith("source.pddl") and
                       not x.endswith("domain.pddl") and
                       (benchmark_domain not in skip_problems or
                        x not in skip_problems[benchmark_domain]) and
                       (filter_task_regenerated is None or
                        x in tasks_regenerated)

                ]
                if len(new_relevant) > 0:
                    relevant_pddls.extend(new_relevant)
                    benchmark_domains_added.append(benchmark_domain)
            if len(relevant_pddls) > 0:
                print(benchmark_domains_added, len(relevant_pddls))
            all_suits.append((benchmark_dir, relevant_pddls))
            if add_suite:
                exp.add_suite(benchmark_dir, relevant_pddls)

    # Add step that writes experiment files to disk.
    exp.add_step('build', exp.build)

    # Add step that executes all runs.
    exp.add_step('start', exp.start_runs)

    # Add step that collects properties from run directories and
    # writes them to *-eval/properties.

    fetcher_filter = [] if fetcher_filter is None else (fetcher_filter if isinstance(fetcher_filter, list) else [fetcher_filter])
    fetcher_filter.append(filter_some_fetcher_errors)
    exp.add_fetcher(name='fetch', filter=fetcher_filter)

    # Add report step (AbsoluteReport is the standard report).
    report = AbsoluteReport(attributes=attributes) if report_filter is None else AbsoluteReport(attributes=attributes, filter=report_filter)
    exp.add_report(
        report,
        outfile='report.html')

    if add_suite:
        return exp
    else:
        return exp, all_suits


_ALGORITHM_EAGER_GREEDY = ("eager_greedy([nh(blind={BLIND},network=sgnet("
                     "path=model.pb,type={TYPE},bin_size={BIN_SIZE},"
                     "unary_threshold={UNARY_THRESHOLD},state_layer={STATE_LAYER},"
                     "goal_layer={GOAL_LAYER},output_layers={OUTPUT_LAYER},"
                     "atoms={ATOMS},defaults={VALUES}))%s]"
                     ",%scost_type=one)")

ALGORITHM_EAGER_GREEDY = _ALGORITHM_EAGER_GREEDY % ("", "")

HEURISTIC_HFF_UNIFORM = "ff(transform=adapt_costs(one))"


def get_algorithm_eager_greedy(heuristic_queue=None, preferred_queue=None):
    heuristic_queue = [] if heuristic_queue is None else heuristic_queue
    heuristic_queue = ",".join(heuristic_queue)
    heuristic_queue = "" if heuristic_queue == "" else ("," + heuristic_queue)

    preferred_queue = [] if preferred_queue is None else preferred_queue
    preferred_queue = ",".join(preferred_queue)
    preferred_queue = "" if preferred_queue == "" else ("preferred=[%s]," % preferred_queue)

    return _ALGORITHM_EAGER_GREEDY % (heuristic_queue, preferred_queue)

ALGORITHM_WASTAR = ("eager_wastar([nh(blind={BLIND},network=sgnet("
                    "path=model.pb,type={TYPE},bin_size={BIN_SIZE},"
                    "unary_threshold={UNARY_THRESHOLD},state_layer={STATE_LAYER},"
                    "goal_layer={GOAL_LAYER},output_layers={OUTPUT_LAYER},"
                    "atoms={ATOMS},defaults={VALUES}))],w={W},cost_type=one)")

DEFAULT_ALGORITHM = ALGORITHM_EAGER_GREEDY

def add_nn_algorithm(experiment, type, state_layer, goal_layer,
                     configurations,
                     unary_threshold=0, bin_size=1,repo=REPO, blind=False, algorithm=None,
                     custom_callbacks=None, driver_options=[], **kwargs):
    algorithm = DEFAULT_ALGORITHM if algorithm is None else algorithm

    for name, prefix, output_layer, atoms, default_values in configurations:
        def cb(run):
            path_model = os.path.join(os.path.dirname(run.task.problem_file),
                                      cb.model)
            match = PATTERN_PROBLEM_IDX.match(run.task.problem_file)
            if match is None:
                logging.critical("Can only use {FOLD} with problem files of "
                                 "of the schema '%s': %s" % (
                    PATTERN_PROBLEM_IDX.pattern,
                    run.task.problem_file
                ))
            idx = int(match.groups()[0])
            if idx < 1:
                idx = 1
            if idx > 200:
                logging.critical("Fold detection only supported for "
                                 "problem indices threshold by 200 "
                                 "(inclusive).")
            fold = int(math.floor((idx - 1) / 20.0))

            path_model = path_model.replace("{FOLD}", str(fold))
            if not os.path.exists(path_model):
                raise SkipFlag()
            run.add_resource("model", path_model, "model.pb", symlink=True)

        cb.model = prefix

        experiment.add_algorithm(
            name, repo, 'default', [], ["release64dynamic"],
            driver_options + ["--build", "release64dynamic", "--pb-network",
             algorithm.format(
                 BLIND="true" if blind else "false",
                 TYPE=type,
                 STATE_LAYER=state_layer,
                 GOAL_LAYER=goal_layer, OUTPUT_LAYER=output_layer,
                 ATOMS="{" + atoms + "}",
                 VALUES="{" + default_values + "}",
                 UNARY_THRESHOLD=unary_threshold,
                 BIN_SIZE=bin_size,
                 **kwargs)],
            dynamic_resources=[("atoms", "atoms.json", True, "atoms.json")],
            callbacks=[cb] if custom_callbacks is None else custom_callbacks
        )


def get_model_regex_from_prefixes(prefixes):
    return re.compile(r"(.*/)?(%s)_\d+_fold_model\.pb$" % "|".join(prefixes))


PATTERN_MODEL_FILE = re.compile(r"(.*/)?(.+)_(\d+)_fold_model\.pb$")
def get_parts_from_model_file(model, prefix=True, fold=True):
    m = PATTERN_MODEL_FILE.match(model)
    assert m is not None
    return tuple(([m.group(2)] if prefix else []) +
                 ([int(m.group(3))] if fold else []))


PATTERN_PROBLEM_FOLD = re.compile(r"(.*/)?p(\d+)\.pddl$")
def get_problem_fold(problem, folds=10, N=200):
    fold_size = N / folds
    m = PATTERN_PROBLEM_FOLD.match(problem)
    assert m is not None
    return int((int(m.group(2)) - 1) / fold_size)


def add_h_comparing_algorithm(search_id, experiment, networks,
                                          driver_options=None, repo=REPO):
    """

    :param experiment:
    :param networks: [(prefix, type, state input layer, goal input layer,
                      output layer, atoms, default values), ...]
    :param repo:
    :return:
    """
    regex_networks = get_model_regex_from_prefixes([x[0] for x in networks])

    def cb(run):
        problem_fold = get_problem_fold(run.task.problem_file)
        path_dir = os.path.dirname(run.task.problem_file)
        if path_dir not in cb.model_cache:
            cb.model_cache[path_dir] = {
                get_parts_from_model_file(file): os.path.join(path_dir, file)
                for file in os.listdir(path_dir)
                if file.endswith("model.pb") and regex_networks.match(file)
            }

        heuristics = []
        for prefix, network_type, state_input, goal_input, output, atoms, default_values in networks:
            if (prefix, problem_fold) in cb.model_cache[path_dir]:
                path_model = cb.model_cache[path_dir][(prefix, problem_fold)]
                run.add_resource(prefix, path_model, "%s.pb" % prefix,
                                 symlink=True)
                heuristics.append(
                    "nh(network=sgnet(path=%s.pb,type=%s,state_layer=%s,"
                    "goal_layer=%s,output_layers=%s,atoms={%s},defaults={%s}))"
                    % (prefix, network_type, state_input, goal_input, output,
                       atoms, default_values))
        if len(heuristics) == 0:
            raise SkipFlag("No heuristics to add")
        heuristics = ",".join(heuristics)
        driver_options = [option.replace("{HEURISTICS}", heuristics)
                          for option in run.properties["driver_options"]]
        run.set_property("driver_options", driver_options)
        run.commands['planner'] = (
                ['{' + run.algo.cached_revision.get_planner_resource_name() + '}']
                + driver_options
                + ['{domain}', '{problem}']
                + run.algo.component_options, run.commands['planner'][1])
    cb.model_cache = {}

    driver_options += ["--build", "release64dynamic", "--pb-network"]
    if search_id == "astar_lmcut":
        search_name = "A*-LMCut-CompareH"
        driver_options += ["astar(lmcut(transform=adapt_costs(ONE)),"
                           "cost_type=one,ref_evals=[{HEURISTICS}])"]
        component_options = []
    elif search_id == "eager_lama_first":
        search_name = "EagerLamaFirst-CompareH"
        driver_options += [
            "{SPLIT}".join(["", "--evaluator",
                           "hlm=lmcount(lm_factory=lm_rhw(reasonable_orders="
                           "true),transform=adapt_costs(one),pref=false)",
                           "--evaluator", "hff=ff(transform=adapt_costs(one))",
                           "--search",
                            "eager_greedy([hff,hlm],preferred=[hff,hlm],"
                           "cost_type=one,ref_evals=[{HEURISTICS}])"])]
        component_options = []
    else:
        assert False

    experiment.add_algorithm(
        search_name, repo, 'default', component_options, ["release64dynamic"],
        driver_options,

        dynamic_resources=[("atoms", "atoms.json", True, "atoms.json")],
        callbacks=[cb]
    )

    parser = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          "parser_h_evolution.py"))
    experiment.add_parser(parser)


def add_astar_lmcut_h_comparing_algorithm(experiment, networks,
                                          driver_options=None, repo=REPO):
    add_h_comparing_algorithm("astar_lmcut", experiment, networks,
                              driver_options, repo)


def add_eager_lama_first_h_comparing_algorithm(experiment, networks,
                                          driver_options=None, repo=REPO):
    add_h_comparing_algorithm("eager_lama_first", experiment, networks,
                              driver_options, repo)


if __name__ == "__main__":
    print("This script shall not be run as main!")
