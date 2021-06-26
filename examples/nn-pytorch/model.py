import torch
import torch.nn as nn

# TODO Create model (Ferber)
#     - Supervised learning
#     - Feed forward
#     - Input = vector representation of a state S (boolean)
#     - 3 hidden layers, Sigmoid activation
#     - Output = heuristic estimate of the distance from S to a goal
#     - Adam optimizer, MSE, Batch size 100
#     - 10-fold cross-validation (9 as training, 1 as validation) --> low-priority

# H(euristic) Neural Network
class HNN(nn.Module):
    def __init__(
        self,
        input_size: int = 3,
        hidden_units: int = 32,
        output_size: int = 1,
            
    ):
        super(HNN, self).__init__()
        self.hid1 = nn.Linear(input_size, hidden_units)
        self.hid2 = nn.Linear(hidden_units, hidden_units)
        self.hid3 = nn.Linear(hidden_units, hidden_units)
        self.opt = nn.Linear(hidden_units, output_size)

    def forward(self, x):
        z = torch.sigmoid(self.hid1(x))
        z = torch.sigmoid(self.hid2(z))
        z = torch.sigmoid(self.hid3(z))
        z = torch.flatten(self.opt(z))
        return z
