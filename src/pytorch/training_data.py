import torch
from torch.utils.data import Dataset, DataLoader
from src.pytorch.utils.helpers import to_prefix, to_onehot
import src.pytorch.fast_downward_api as fd_api
from src.pytorch.utils.default_args import DEFAULT_CLAMPING
from itertools import chain, zip_longest

class InstanceDataset(Dataset):
    def __init__(self, state_value_pairs, domain_max_value, output_layer):
        states, hvalues = [], []
        for pair in state_value_pairs:
            states.append(pair[0])
            hvalues.append(pair[1])

        self.output_layer = output_layer
        self.domain_max_value = domain_max_value

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

    def __getitem__(self, idx):
        return self.states[idx], self.hvalues[idx]

    def __len__(self):
        return len(self.states)

    def x_shape(self):
        return self.states.shape

    def y_shape(self):
        if self.output_layer == "regression":
            return torch.Size([len(self.hvalues), 1])
        return self.hvalues.shape

    def set_x(self, x):
        self.states = x

    def set_y(self, y):
        self.hvalues = y


def load_training_state_value_pairs(samples_file: str, clamping: int, remove_goals: bool,
                                    standard_first: bool, contrast_first: bool,
                                    intercalate_samples: int) -> ([([int], int)], int):
    """
    Load state-value pairs from a sampling output,
    Returns a tuple containing a list of state-value pairs
    and the domain max value.
    """
    state_value_pairs = []
    domain_max_value, max_h = 0, 0

    with open(samples_file) as f:
        lines = f.readlines()
    for line in lines:
        if line[0] != "#":
            h, state = line.split("\n")[0].split(";")
            h_int = int(h)
            if h_int == 0 and remove_goals:
                continue
            state = [int(s) for s in state]
            state_value_pairs.append([state, h_int])
            if h_int > max_h:
                max_h = h_int
                domain_max_value = max_h

    if clamping != DEFAULT_CLAMPING:
        for i in range(len(state_value_pairs)):
            curr_h = state_value_pairs[i][1]
            if (curr_h >= max_h - clamping) and (curr_h != max_h):
                state_value_pairs[i][1] = max_h

    if standard_first:
        state_value_pairs = change_sampling_order(state_value_pairs, max_h, True, False, intercalate_samples)
    elif contrast_first:
        state_value_pairs = change_sampling_order(state_value_pairs, max_h, False, True, intercalate_samples)
    elif intercalate_samples > 0:
        state_value_pairs = change_sampling_order(state_value_pairs, max_h, False, False, intercalate_samples)

    return state_value_pairs, domain_max_value


def change_sampling_order(state_value_pairs, max_h, std_first, cont_first, interc_n):
    standard_samples = []
    contrast_samples = []

    for sv in state_value_pairs:
        if sv[1] == max_h:
            contrast_samples.append(sv)
        else:
            standard_samples.append(sv)

    if std_first or cont_first:
        return standard_samples + contrast_samples if std_first else contrast_samples + standard_samples
    else:
        """
        chunked_l1 = zip_longest(*[iter(standard_samples)]*interc_n)
        chunked_l2 = zip_longest(*[iter(contrast_samples)]*interc_n)
        new_state_value_pairs = (chain(a, b) for a, b in zip(chunked_l1, chunked_l2))
        new_state_value_pairs = (chain.from_iterable(new_state_value_pairs))
        new_state_value_pairs = [x for x in new_state_value_pairs if x is not None]
        """
        min_len = min(len(standard_samples), len(contrast_samples))
        new_state_value_pairs = []
        if interc_n > 1:
            for i in range(0, min_len, interc_n):
                new_state_value_pairs += standard_samples[i:i+interc_n] + contrast_samples[i:i+interc_n]
            if min_len == len(standard_samples):
                new_state_value_pairs += contrast_samples[i+interc_n:]
            else:
                new_state_value_pairs += standard_samples[i+interc_n:]
        else:
            new_state_value_pairs = [x for x in chain.from_iterable(
                zip_longest(standard_samples, contrast_samples)) if x is not None]
        """
        # Test
        a = new_state_value_pairs[-300:-1]
        for i in a:
            print(i[1])
        print(len(new_state_value_pairs), len(state_value_pairs), len(standard_samples), len(contrast_samples))
        exit(1)
        """

        return new_state_value_pairs

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
