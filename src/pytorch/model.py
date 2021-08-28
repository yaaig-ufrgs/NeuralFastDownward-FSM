import torch
import torch.nn as nn

# H(euristic) Neural Network
class HNN(nn.Module):
    def __init__(
        self,
        input_units: int,
        hidden_units: [int],
        output_units: int,
        hidden_layers: int,
        activation: str,
        dropout_rate: float,
    ):
        super(HNN, self).__init__()
        self.input_units = input_units
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate

        hu = [input_units]
        if len(hidden_units) == 0: # scalable
            unit_diff = input_units - output_units
            step = int(unit_diff / (hidden_layers+1))
            for i in range(self.hidden_layers):
                hu.append(input_units-(i+1)*step)
        elif len(hidden_units) == 1: # all the same
            hu += hidden_units * self.hidden_layers
        else:
            hu += hidden_units

        self.hid = nn.ModuleList()
        for i in range(self.hidden_layers):
            self.hid.append(nn.Linear(hu[i], hu[i+1]))
        self.opt = nn.Linear(hu[-1], output_units)

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate, inplace=False)
        
        if activation == "sigmoid":
            self.activation = torch.sigmoid
        elif activation == "relu":
            self.activation = torch.relu
        else:
            raise NotImplementedError(f"{activation} function not implemented!")

    def forward(self, x):
        for h in self.hid:
            x = self.activation(h(x))
            if self.dropout_rate > 0:
                x = self.dropout(x)
        return self.activation(self.opt(x))
        # return torch.flatten(self.opt(x))
