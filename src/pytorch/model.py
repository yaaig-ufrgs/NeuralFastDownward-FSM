import torch
import torch.nn as nn

# TODO parametrization

# H(euristic) Neural Network
class HNN(nn.Module):
    def __init__(
        self,
        input_units: int,
        nb_layers: int,
        output_units: int,
    ):
        super(HNN, self).__init__()
        self.input_units = input_units
        self.output_units = output_units
        self.nb_layers = nb_layers
        unit_diff = input_units - output_units
        step = int(unit_diff / (nb_layers+1))

        self.hid1 = nn.Linear(input_units-0*step, input_units-1*step)
        self.opt = nn.Linear(input_units-nb_layers*step, output_units)


    def forward(self, x):
        z = torch.sigmoid(self.hid1(x))
        # z = torch.sigmoid(self.hid2(z))
        z = torch.sigmoid(self.opt(z))
        # z = torch.flatten(self.opt(z))
        return z
