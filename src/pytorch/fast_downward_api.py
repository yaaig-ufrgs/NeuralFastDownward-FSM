import logging
from os import path, makedirs
from subprocess import check_output, CalledProcessError
from re import compile, findall, match
from src.pytorch.utils.default_args import (
    DEFAULT_DOMAIN_PDDL,
    DEFAULT_SEARCH_ALGORITHM,
    DEFAULT_HEURISTIC,
    DEFAULT_MAX_SEARCH_TIME,
    DEFAULT_MAX_SEARCH_MEMORY,
    DEFAULT_MAX_EXPANSIONS,
    DEFAULT_UNARY_THRESHOLD,
)

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
}


def parse_fd_output(output: str):
    # Remove \n to use in re.
    output = output.replace("\n", " ")
    re_initial_h = match(
        r".*Initial heuristic value for .*?: (\d+)", output
    )
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
            results["plan_length"] = re_plan[0][0]
            results["plan_cost"] = re_plan[0][1]
        results["initial_h"] = re_initial_h.group(1)
        results["expanded"] = re_states[0][0]
        results["reopened"] = re_states[0][1]
        results["evaluated"] = re_states[0][2]
        results["generated"] = re_states[0][3]
        results["dead_ends"] = re_states[0][4]
        results["search_time"] = re_time[0][0]
        results["expansion_rate"] = round(
            float(results["expanded"]) / float(results["search_time"]), 4
        )
        results["total_time"] = re_time[0][1]
    return results


def save_downward_log(folder, instance_pddl, output):
    downward_logs = f"{folder}/downward_logs"
    if not path.exists(downward_logs):
        makedirs(downward_logs)
    instance_name = instance_pddl.split("/")[-1].split(".pddl")[0]
    filename = f"{downward_logs}/{instance_name}.log"
    with open(filename, "w") as f:
        f.write(output)
    _log.info(f"Downward log saved to {filename}")


def solve_instance_with_fd(
    domain_pddl,
    instance_pddl,
    opts="astar(lmcut())",
    memory_limit=DEFAULT_MAX_SEARCH_MEMORY,
    save_log_to=None,
):
    try:
        cl = [
            _FD,
            "--search-memory-limit",
            str(memory_limit),
            instance_pddl,
            "--search",
            opts,
        ]
        if domain_pddl != DEFAULT_DOMAIN_PDDL:
            cl.insert(3, domain_pddl)
        if save_log_to != None:
            # Set temp files to allow running multiple
            # downwards at the same time
            cl.insert(1, "--sas-file")
            cl.insert(2, f"{save_log_to}/output.sas")
            cl.insert(3, "--plan-file")
            cl.insert(4, f"{save_log_to}/sas_plan")
        output = check_output(cl)
        _log.info("Solution found.")
    except CalledProcessError as e:
        if e.returncode != 12:
            if e.returncode == 36:
                _log.error("Could not find domain file using automatic naming rules.")
            return {"search_state": f"downward returned {e.returncode}"}
        output = e.output
        _log.info("Solution not found.")
    output = output.decode("utf-8")
    if save_log_to != None:
        save_downward_log(save_log_to, instance_pddl, output)
    return parse_fd_output(output)


def solve_instance_with_fd_nh(
    domain_pddl,
    problem_pddl,
    traced_model,
    search_algorithm=DEFAULT_SEARCH_ALGORITHM,
    heuristic=DEFAULT_HEURISTIC,
    unary_threshold=DEFAULT_UNARY_THRESHOLD,
    time_limit=DEFAULT_MAX_SEARCH_TIME,
    memory_limit=DEFAULT_MAX_SEARCH_MEMORY,
    max_expansions=DEFAULT_MAX_EXPANSIONS,
    save_log_to=None,
):
    """
    Tries to solve a PDDL instance with the torch_sampling_network.
    """

    if heuristic == "nn":
        opt_network = (
            f"torch_sampling_network(path={traced_model},"
            f"unary_threshold={unary_threshold})"
        )
        opt_heuristic = f"nh({opt_network})"
    else:
        opt_heuristic = f"{heuristic}()"

    if search_algorithm == "eager_greedy":
        opt_heuristic = f"[{opt_heuristic}]"

    opts = f"{search_algorithm}({opt_heuristic}"
    if time_limit != float("inf"):
        opts += f", max_time={time_limit}"
    if max_expansions != float("inf"):
        opts += f", max_expansions={max_expansions}"
    opts += ")"

    return solve_instance_with_fd(
        domain_pddl, problem_pddl, (opts), memory_limit, save_log_to
    )
