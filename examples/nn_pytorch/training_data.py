import random
import torch
from torch.utils.data import Dataset, DataLoader

from utils import to_unary
import fast_downward_api as fd_api

class InstanceDataset(Dataset):
    def __init__(self, state_value_pairs, domain_max_value):
        self.domain_max_value = domain_max_value

        states, hvalues = [], []
        for pair in state_value_pairs:
            states.append(pair[0])
            hvalues.append(pair[1])

        self.states = torch.tensor(states, dtype=torch.float32)
        self.hvalues = torch.tensor([to_unary(n, domain_max_value) for n in hvalues],
                                    dtype=torch.float32)

    def __getitem__(self, idx):
        return self.states[idx], self.hvalues[idx]

    def __len__(self):
        return len(self.states)

    def x_shape(self):
        return self.states.shape

    def y_shape(self):
        return self.hvalues.shape


def load_training_state_value_pairs(sas_plans: [str]) -> [([int], int)]:
    """
    Load state-value pairs from a sampling output, returning
    a list of states (each state is a bool list) and a list
    of hvalues. States and hvalues correspond by index.
    """

    state_value_pairs = []
    states = []
    hvalues = []
    for sp in sas_plans:
        print(sp)
        with open(sp) as f:
            lines = f.readlines()
            states_len = len(lines[2].split(';'))

            for i in range(5, len(lines)):
                if lines[i][0] != '#':
                    # Reading the current plan.
                    values = lines[i].split(';')

                    state = []
                    for i in range(1, states_len+1):
                        state.append(int(values[i]))
                    states.append(state)
                    hvalues.append(int(values[0]))
                else:
                    # Finished reading the current plan, so get a random state-value from it.
                    state_value_pairs.append(select_random_state(states, hvalues))
                    states.clear()
                    hvalues.clear()

    return state_value_pairs


def select_random_state(states: [[int]], hvalues: [int]) -> ([int], int):
    i = random.choice(range(len(hvalues)))
    return ((states[i], hvalues[i]))


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
        atoms = [x[5:] for x in f.readlines()[0].split(";")] # remove "Atom " prefix
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


def setup_dataloaders(dataset: InstanceDataset, train_split: float, batch_size: int,
                      shuffle: bool) -> (DataLoader, DataLoader):
    """
    Setup training and validation datasets using a random split.
    """

    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=1)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                shuffle=shuffle, num_workers=1)

    return train_dataloader, val_dataloader


def setup_train_dataloader(dataset: InstanceDataset, batch_size: int, shuffle: bool) -> (DataLoader, DataLoader):
    """
    Setup only a training dataset.
    """

    train_dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=1)
    return train_dataloader


def load_training_state_value_tuples(sas_plan: str) -> ([[int]], [int]):
    """
    Load state-value pairs from a sampling output, returning
    a list of states (each state is a bool list) and a list
    of hvalues. States and hvalues correspond by index.
    """

    states = []
    hvalues = []
    with open(sas_plan,) as f:
        lines = f.readlines()
        states_len = len(lines[2].split(';'))

        for i in range(5, len(lines)):
            if lines[i][0] != '#':
                values = lines[i].split(';')

                state = []
                for i in range(1, states_len+1):
                    state.append(int(values[i]))
                states.append(state)
                hvalues.append(int(values[0]))

    return states, hvalues
