import logging
import torch
from torch.utils.data import Dataset, DataLoader

from src.pytorch.utils.helpers import to_prefix, to_onehot
import src.pytorch.fast_downward_api as fd_api

_log = logging.getLogger(__name__)

class InstanceDataset(Dataset):
    def __init__(self, state_value_pairs, domain_max_value, output_layer):
        states, hvalues = [], []
        for pair in state_value_pairs:
            states.append(pair[0])
            hvalues.append(pair[1])

        self.domain_max_value = domain_max_value

        self.states = torch.tensor(states, dtype=torch.float32)
        if output_layer == "regression":
            self.hvalues = torch.tensor(
                [[n] for n in hvalues], dtype=torch.float32
            )
        elif output_layer == "prefix":
            self.hvalues = torch.tensor(
                [to_prefix(n, self.domain_max_value) for n in hvalues], dtype=torch.float32
            )
        elif output_layer == "one-hot":
            self.hvalues = torch.tensor(
                [to_onehot(n, self.domain_max_value) for n in hvalues], dtype=torch.float32
            )
        else:
            raise RuntimeError(f"Invalid output layer: {output_layer}")

    def __getitem__(self, idx):
        return self.states[idx], self.hvalues[idx]

    def __len__(self):
        return len(self.states)

    def x_shape(self):
        return self.states.shape

    def y_shape(self):
        return self.hvalues.shape


def load_training_state_value_pairs(samples_file: str) -> ([([int], int)], int):
    """
    Load state-value pairs from a sampling output,
    Returns a tuple containing a list of state-value pairs
    and the domain max value.
    """

    state_value_pairs = []
    domain_max_value = 0
    lines = samples_file.readlines()[1:]
    for line in lines:
        l = line.split("\n")[0].split(";")
        value = int(l[0])
        state = []
        for i in range(1, len(l)):
            state.append(int(l[i]))
        state_value_pairs.append((state, int(l[0])))
        if state_value_pairs[-1][1] > domain_max_value:
            domain_max_value = state_value_pairs[-1][1]

    return state_value_pairs, domain_max_value


def generate_optimal_state_value_pairs(domain, problems):
    """
    Generates the state value pair from a set of problems.
    Returns a list of tuple(state, value).
    """

    states = convert_pddl_to_boolean(domain, problems)
    state_value_pairs = []
    for i in range(len(problems)):
        value = fd_api.solve_instance_with_fd(domain, problems[i])
        if value != None:
            state_value_pairs.append((states[i], value))
        else:
            print(f"Solution plan not found for the problem {problems[i]}")

    return state_value_pairs


def convert_pddl_to_boolean(domain, problems):
    """
    From a pddl, it gets the boolean vector in the format for network input.
    """

    # TODO: Get atoms order from domain
    with open("atoms/probBLOCKS-12-0.txt") as f:
        # Read and convert (e.g. "Atom ontable(a)" -> "ontable a")
        atoms = [x[5:] for x in f.readlines()[0].split(";")]  # remove "Atom " prefix
        for i in range(len(atoms)):
            pred, objs = atoms[i].split("(")
            objs = objs[:-1].replace(",", "")
            atoms[i] = pred
            if objs != "":
                atoms[i] += " " + objs

    states = []
    for problem in problems:
        with open(problem) as f:
            initial_state = f.read().split(":init")[1].split(":goal")[0]
        state = []
        for a in atoms:
            state.append(int(a in initial_state))
        states.append(state)

    return states


def load_training_state_value_tuples(sas_plan: str) -> ([[int]], [int]):
    """
    Load state-value pairs from a sampling output, returning
    a list of states (each state is a bool list) and a list
    of hvalues. States and hvalues correspond by index.
    """

    states = []
    hvalues = []
    with open(
        sas_plan,
    ) as f:
        lines = f.readlines()
        states_len = len(lines[2].split(";"))

        for i in range(5, len(lines)):
            if lines[i][0] != "#":
                values = lines[i].split(";")

                state = []
                for i in range(1, states_len + 1):
                    state.append(int(values[i]))
                states.append(state)
                hvalues.append(int(values[0]))

    return states, hvalues
