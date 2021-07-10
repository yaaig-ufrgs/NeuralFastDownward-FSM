import torch
from torch.utils.data import Dataset, DataLoader

from json import load
from itertools import permutations

class InstanceDataset(Dataset):
    def __init__(self, training_data: str):
        states, hvalues = load_training_state_value_tuples(training_data)
        self.states = torch.tensor(states, dtype=torch.float32)
        self.hvalues = torch.tensor(hvalues, dtype=torch.float32)

    def __getitem__(self, idx):
        return self.states[idx], self.hvalues[idx]

    def __len__(self):
        return len(self.states)

    def x_shape(self):
        return self.states.shape

    def y_shape(self):
        return self.hvalues.shape


def load_training_state_value_tuples(sas_plan: str):
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
