import torch
from torch.utils.data import Dataset, DataLoader

from json import load
from itertools import permutations

class InstanceDataset(Dataset):
    def __init__(self, training_data: str, domain: str):
        self.data = load_training_state_value_tuples(training_data)
        self.domain = domain
        self.states = torch.tensor(states_to_boolean(self.data, self.domain), dtype=torch.float32)
        self.hvalues = torch.tensor([t[1] for t in self.data], dtype=torch.float32)

    def __getitem__(self, idx):
        return self.states[idx], self.hvalues[idx]

    def __len__(self):
        return len(self.states)

    def x_shape(self):
        return self.states.shape

    def y_shape(self):
        return self.hvalues.shape


def load_training_state_value_tuples(json_file: str):
    """
    Load state-value pairs from a JSON file (Shen's format),
    returning a list of tuples in the format [([state], value)..].
    """
    with open(json_file,) as f:
        data = load(f)
        state_value_pairs = []

        for domain, training_pairs in data.items():
            for pair in training_pairs:
                state = [e[1:-1] for e in pair['state']]
                state_value = (state, pair['value'])
                state_value_pairs.append(state_value)

        return state_value_pairs

"""
Blocksworld state:

handempty
clear b1
on-table b3
on b1 b2
on b2 b3

Boolean format example for max blocks = 5

| handempty |   clear   |   holding   |   on-table   |             on               |
-------------------------------------------------------------------------------------
| 1         | 1 0 0 0 0 |  0 0 0 0 0  |0  0  1  0  0 | 1  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0 |
  0           1 2 3 4 5    6 7 8 9 10  11 12 13 14 15  16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
"""

def states_to_boolean(data: [], domain: str):
    bool_states = []
    states = [t[0] for t in data]

    if domain == "blocksworld":
        permDict = {}
        index = 0
        for p in permutations(range(1, 6), 2):
            permDict[p] = index
            index += 1

        for state in states:
           bool_state = []
           handempty = [0]
           clear = [0]*5
           holding = [0]*5
           on_table = [0]*5
           on = [0]*20

           for atom in state:
               var_list = atom.split()
               if var_list[0] == "handempty":
                   handempty[0] = 1
               elif var_list[0] == "clear":
                   bx = int(var_list[1][1])-1
                   clear[bx] = 1;
               elif var_list[0] == "holding":
                   bx = int(var_list[1][1])-1
                   holding[bx] = 1
               elif var_list[0] == "on-table":
                   bx = int(var_list[1][1])-1
                   on_table[bx] = 1
               elif var_list[0] == "on":
                   bx = (int(var_list[1][1]), int(var_list[2][1]))
                   on[permDict[bx]] = 1

           bool_state = handempty + clear + holding + on_table + on
           bool_states.append(bool_state)

    return bool_states

# TODO - to avoid having to take the state-value pairs from Shen's data.
def generate_optimal_state_value_pairs(problem):
    pass

#data = load_training_state_value_tuples("domain_to_training_pairs_blocks.json")
#boolean_states = states_to_boolean(data, "blocksworld")
