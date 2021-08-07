from subprocess import check_output, CalledProcessError
import re

_FD = "./fast-downward.py"
_SAS_PLAN_FILE = "sas_plan"


def parse_plan():
    PLAN_INFO_REGEX = re.compile(r"; cost = (\d+) \((unit cost|general cost)\)\n")
    last_line = ""
    try:
        with open(_SAS_PLAN_FILE) as sas_plan:
            for last_line in sas_plan:
                pass
    except:
        pass
    match = PLAN_INFO_REGEX.match(last_line)
    if match:
        return int(match.group(1)), match.group(2)
    else:
        return None, None

def parse_fd_output(output: str):
    assert "Solution found!" in output

    # Remove \n to use in re.compile.
    output = output.replace("\n", " ")

    data = re.findall(
        r".*Plan length: (\d+) step\(s\)..*"
        r".*Plan cost: (\d+).*"
        r".*Expanded (\d+) state\(s\)..*"
        r".*Reopened (\d+) state\(s\)..*"
        r".*Evaluated (\d+) state\(s\)..*"
        r".*Generated (\d+) state\(s\)..*"
        r".*Dead ends: (\d+) state\(s\)..*"
        r".*Search time: (.+?)s.*"
        r".*Total time: (.+?)s.*"
        , output
    )

    return {
        "search_state" : "success",
        "plan_length" : data[0][0],
        "plan_cost" : data[0][1],
        "expanded" : data[0][2],
        "reopened" : data[0][3],
        "evaluated" : data[0][4],
        "generated" : data[0][5],
        "dead_ends" : data[0][6],
        "search_time" : data[0][7],
        "total_time" : data[0][8]
    }

def solve_instance_with_fd(
    domain_pddl,
    instance_pddl,
    opts="astar(lmcut())",
    time_limit=604800,
    memory_limit=32000000,
):
    """
    Tries to solve a PDDL instance. Return the cost (or None if search fails).
    """

    try:
        output = check_output(
            [
                _FD,
                "--search-time-limit",
                str(time_limit),
                "--search-memory-limit",
                str(memory_limit),
                domain_pddl,
                instance_pddl,
                "--search",
                opts,
            ]
        )
        return parse_fd_output(output.decode("utf-8"))

    except CalledProcessError as e:
        exit_code = {
            1  : "search plan found and out of memory",
            2  : "seach plan found and out of time",
            3  : "seach plan found and out of time & memory",
            11 : "search unsolvable",
            # 12 : "search unsolvable incomplete",
            12 : "search out of time",
            22 : "search out of memory",
            23 : "search out of time",
            24 : "search out of memory and time"
        }
        return {"search_state" : exit_code[e.returncode] \
            if e.returncode in exit_code else f"Unknown return code: {e.returncode}"}


def solve_instance_with_fd_nh(
    domain_pddl,
    problem_pddl,
    traced_model,
    search_algorithm = "astar",
    unary_threshold=0.01,
    time_limit=604800,
    memory_limit=128000,
):
    """
    Tries to solve a PDDL instance with the torch_sampling_network. Return the cost (or None if search fails).
    """

    network = f"torch_sampling_network(path={traced_model}," \
        f"blind={str(search_algorithm == 'blind').lower()}," \
        f"unary_threshold={unary_threshold})"

    if search_algorithm == "astar" or search_algorithm == "blind":
        opts = (f"astar(nh({network}), max_time={time_limit})")
    elif search_algorithm == "eager_greedy":
        opts = (f"eager_greedy([nh({network})], max_time={time_limit})")

    return solve_instance_with_fd(
        domain_pddl, problem_pddl, opts, time_limit, memory_limit
    )
