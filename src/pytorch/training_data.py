import torch
from torch.utils.data import Dataset, DataLoader
from src.pytorch.utils.helpers import to_prefix, to_onehot
import src.pytorch.fast_downward_api as fd_api
import src.pytorch.utils.default_args as default_args
from itertools import chain, zip_longest


class InstanceDataset(Dataset):
    def __init__(
        self, state_value_pairs: list, domain_max_value: int, output_layer: str
    ):
        states, hvalues, weights = [], [], []
        for pair in state_value_pairs:
            states.append(pair[0])
            hvalues.append(pair[1])
            weights.append(pair[2])

        self.output_layer = output_layer
        self.domain_max_value = domain_max_value

        self.weights = torch.tensor(states, dtype=torch.float32)
        self.states = torch.tensor(states, dtype=torch.float32)
        if output_layer == "regression":
            self.hvalues = torch.tensor(hvalues, dtype=torch.float32).unsqueeze(1)
        elif output_layer == "prefix":
            self.hvalues = torch.tensor(
                [to_prefix(n, self.domain_max_value) for n in hvalues],
                dtype=torch.float32,
            )
        elif output_layer == "one-hot":
            self.hvalues = torch.tensor(
                [to_onehot(n, self.domain_max_value) for n in hvalues],
                dtype=torch.float32,
            )
        else:
            raise RuntimeError(f"Invalid output layer: {output_layer}")

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        return self.states[idx], self.hvalues[idx], self.weights[idx]

    def __len__(self):
        return len(self.states)

    def x_shape(self):
        return self.states.shape

    def y_shape(self):
        if self.output_layer == "regression":
            return torch.Size([len(self.hvalues), 1])
        return self.hvalues.shape


def load_training_state_value_pairs(
        samples_file: str, clamping: int, remove_goals: bool,  unique_samples: bool, unique_states: bool
) -> ([([int], int)], int):
    """
    Load state-value pairs from a sampling output, returning a tuple
    containing a list of state-value pairs and the domain max value.
    This is the training data.
    """
    state_value_pairs = []
    domain_max_value, max_h = 0, 0
    uniques_xy = []
    uniques_x = []
    sample_count = {}

    with open(samples_file) as f:
        lines = f.readlines()
    for line in lines:
        if line[0] != "#":
            # If specified, skip repeated lines (state and heuristic) if they already appeared.
            if unique_samples:
                if line in uniques_xy:
                    continue
                uniques_xy.append(line)

            h, state = line.split("\n")[0].split(";")
            h_int = int(h)

            # h = 0 means state x is a goal, so remove it if specified.
            if h_int == 0 and remove_goals:
                continue

            # Count how many times each state (x) appeared.
            if state in sample_count:
                sample_count[state] += 1
            else:
                sample_count[state] = 1

            # If specified, skip state (x) if it already appeared.
            if unique_states:
                if state in uniques_x:
                    continue
                uniques_x.append(state)

            state = [int(s) for s in state]
            state_value_pairs.append([state, h_int])

            # Gets the domain max h value.
            if h_int > max_h:
                max_h = h_int
                domain_max_value = max_h

    # Clamps the heuristic value.
    if clamping != default_args.CLAMPING:
        for i in range(len(state_value_pairs)):
            curr_h = state_value_pairs[i][1]
            if (curr_h >= max_h - clamping) and (curr_h != max_h):
                state_value_pairs[i][1] = max_h

    # Appends weights (counts) to state_value_pairs:
    for sv in state_value_pairs:
        st = "".join([str(s) for s in sv[0]])
        sv.append(sample_count[st])
        
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
