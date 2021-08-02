import torch
import torch.nn as nn

# TODO Create model (Ferber)
#     - Supervised learning - DONE
#     - Feed forward - DONE
#     - Input = vector representation of a state S (boolean) - DONE for blocksworld
#     - 1-3 hidden layers, Sigmoid activation - DONE
#     - Output = heuristic estimate of the distance from S to a goal DONE
#     - Adam optimizer, MSE, Batch size 100 - WAIT
#     - 10-fold cross-validation (9 as training, 1 as validation) --> TODO low-priority

# H(euristic) Neural Network
class HNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_units: int,
        output_size: int,
    ):
        super(HNN, self).__init__()
        self.hid1 = nn.Linear(input_size, hidden_units)
        # self.hid2 = nn.Linear(hidden_units, hidden_units)
        # self.hid3 = nn.Linear(hidden_units, hidden_units)
        self.opt = nn.Linear(hidden_units, output_size)

    def forward(self, x):
        z = torch.sigmoid(self.hid1(x))
        # z = torch.sigmoid(self.hid2(z))
        # z = torch.sigmoid(self.hid3(z))
        z = torch.sigmoid(self.opt(z))
        # z = torch.flatten(self.opt(z))
        return z
