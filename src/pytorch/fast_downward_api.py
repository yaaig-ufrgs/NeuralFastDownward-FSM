import logging
from os import path, makedirs
from subprocess import check_output, CalledProcessError
from re import compile, findall, match
import src.pytorch.utils.default_args as default_args

_log = logging.getLogger(__name__)

_FD = "./fast-downward.py"
_FD_EXIT_CODE = {
    0: "success",
    1: "search plan found and out of memory",
    2: "search plan found and out of time",
    3: "search plan found and out of time & memory",
    11: "search unsolvable",
    12: "search unsolvable incomplete",
    22: "search out of memory",
    23: "search out of time",
    24: "search out of memory and time",
    36: "domain file not found",
}


def parse_fd_output(output: str) -> dict:
    """
    Reads Fast Downward's output during problem solving, and retuns a dict
    with the relevant information.
    """

    # Remove \n to use in re.
    output = output.replace("\n", " ")
    re_initial_h = match(r".*Initial heuristic value for .*?: (\d+)", output)
    re_plan = findall(
        r".*Plan length: (\d+) step\(s\)..*" r".*Plan cost: (\d+).*", output
    )
    re_states = findall(
        r".*Expanded (\d+) state\(s\)..*"
        r".*Reopened (\d+) state\(s\)..*"
        r".*Evaluated (\d+) state\(s\)..*"
        r".*Generated (\d+) state\(s\)..*"
        r".*Dead ends: (\d+) state\(s\)..*",
        output,
    )
    re_time = findall(r".*Search time: (.+?)s.*" r".*Total time: (.+?)s.*", output)
    exit_code = int(findall(r".*search exit code: (\d+).*", output)[0])
    results = {
        "search_state": _FD_EXIT_CODE[exit_code]
        if exit_code in _FD_EXIT_CODE
        else f"unknown exit code {exit_code}"
    }
    # Possible exit codes 12
    if exit_code in [0, 12]:
        if exit_code == 12:
            if "Time limit reached." in output:
                results["search_state"] = "timeout"
            elif "Maximum number of expansions reached." in output:
                results["search_state"] = "maximum expansions reached"
        if exit_code == 0:
            results["plan_length"] = int(re_plan[0][0])
            results["plan_cost"] = int(re_plan[0][1])
        results["initial_h"] = int(re_initial_h.group(1))
        results["expanded"] = int(re_states[0][0])
        results["reopened"] = int(re_states[0][1])
        results["evaluated"] = int(re_states[0][2])
        results["generated"] = int(re_states[0][3])
        results["dead_ends"] = int(re_states[0][4])
        results["search_time"] = float(re_time[0][0])
        results["expansion_rate"] = round(
            float(results["expanded"]) / float(results["search_time"]), 4
        )
        results["total_time"] = float(re_time[0][1])

    return results


def save_downward_log(folder: str, instance_pddl: str, output: str):
    """
    Saves the full Fast Downward log printed during problem solving.
    """
    downward_logs = f"{folder}/downward_logs"
    if not path.exists(downward_logs):
        makedirs(downward_logs)
    instance_name = instance_pddl.split("/")[-1].split(".pddl")[0]
    filename = f"{downward_logs}/{instance_name}.log"
    with open(filename, "w") as f:
        f.write(output)
    _log.info(f"Downward log saved to {filename}")


def solve_instance_with_fd(
    domain_pddl: str,
    instance_pddl: str,
    opts: str = "astar(lmcut())",
    memory_limit: int = default_args.MAX_SEARCH_MEMORY,
    save_log_to=None,
    save_log_bool: bool = default_args.SAVE_DOWNWARD_LOGS,
) -> dict:
    """
    Tries to solve an instance using Fast Downward and one of its search algorithms.
    """
    try:
        cl = [
            _FD,
            "--search-memory-limit",
            str(memory_limit),
            instance_pddl,
            "--search",
            opts,
        ]
        if domain_pddl != default_args.DOMAIN_PDDL:
            cl.insert(3, domain_pddl)
        if save_log_to != None:
            # Set temp files to allow running multiple
            # downwards at the same time
            cl.insert(1, "--sas-file")
            cl.insert(2, f"{save_log_to}/output.sas")
            cl.insert(3, "--plan-file")
            cl.insert(4, f"{save_log_to}/sas_plan")
        _log.info(f"Command line string: {' '.join(cl)}")
        _log.info(f"Running FastDownward...")
        output = check_output(cl)
        _log.info("Solution found.")
    except CalledProcessError as e:
        if e.returncode != 12:
            if e.returncode == 36:
                _log.error("Could not find domain file using automatic naming rules.")
            return {
                "search_state": _FD_EXIT_CODE[e.returncode]
                if e.returncode in _FD_EXIT_CODE
                else f"unknown exit code {e.returncode}"
            }
        output = e.output
        _log.info("Solution not found.")
    output = output.decode("utf-8")
    if save_log_bool and save_log_to != None:
        save_downward_log(save_log_to, instance_pddl, output)
    return parse_fd_output(output)


def solve_instance_with_fd_nh(
    domain_pddl: str,
    problem_pddl: str,
    traced_model: str,
    traced_model_cmp: str,
    search_algorithm: str = default_args.SEARCH_ALGORITHM,
    heuristic: str = default_args.HEURISTIC,
    heuristic_multiplier: int = default_args.HEURISTIC_MULTIPLIER,
    unary_threshold: float = default_args.UNARY_THRESHOLD,
    time_limit: int = default_args.MAX_SEARCH_TIME,
    memory_limit: int = default_args.MAX_SEARCH_MEMORY,
    max_expansions: int = default_args.MAX_EXPANSIONS,
    facts_file: str = default_args.FACTS_FILE,
    defaults_file: str = default_args.DEF_VALUES_FILE,
    save_log_to = None,
    save_log_bool: bool = default_args.SAVE_DOWNWARD_LOGS,
) -> dict:
    """
    Tries to solve a PDDL instance with the torch_sampling_network.
    """

    if heuristic == "nn":
        opt_network = "torch_sampling_network("
        opt_network += f"path={traced_model}"
        if traced_model_cmp:
            opt_network += f", path_cmp={traced_model_cmp}"
        if heuristic_multiplier != default_args.HEURISTIC_MULTIPLIER:
            opt_network += f", multiplier={heuristic_multiplier}"
        if facts_file:
            opt_network += f", facts=[file {facts_file}]"
        if defaults_file:
            opt_network += f", defaults=[file {defaults_file}]"
        if unary_threshold != default_args.UNARY_THRESHOLD:
            opt_network += f", unary_threshold={unary_threshold}"
        if "_us_" in traced_model:
            opt_network += f", undefined_input=true"
        opt_network += ")"

        opt_heuristic = f"nh({opt_network})"
    else:
        opt_heuristic = f"{heuristic}()"

    if search_algorithm == "eager_greedy":
        opt_heuristic = f"[{opt_heuristic}]"

    opts = search_algorithm + "(" + opt_heuristic
    if time_limit != default_args.MAX_SEARCH_TIME:
        opts += f", max_time={time_limit}"
    if max_expansions != default_args.MAX_EXPANSIONS:
        opts += f", max_expansions={max_expansions}"
    opts += ")"

    return solve_instance_with_fd(
        domain_pddl, problem_pddl, (opts), memory_limit, save_log_to, save_log_bool
    )
