import torch
from torch.utils.data import Dataset, DataLoader

from utils import to_unary

class InstanceDataset(Dataset):
    def __init__(self, training_data: str, domain_max_value: int):
        states, hvalues = load_training_state_value_tuples(training_data)

        self.domain_max_value = domain_max_value
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


def setup_train_dataloader(dataset: InstanceDataset, batch_size: int, shuffle: bool) ->(DataLoader, DataLoader):
    """
    Setup only a training dataset.
    """

    train_dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=1)
    return train_dataloader

