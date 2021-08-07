import torch
import torch.nn as nn

# H(euristic) Neural Network
class HNN(nn.Module):
    def __init__(
        self,
        input_units: int,
        output_units: int,
        hidden_layers: int,
        activation: str,
        dropout_rate: float,
    ):
        super(HNN, self).__init__()
        self.input_units = input_units
        self.output_units = output_units
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout_rate = dropout_rate

        unit_diff = input_units - output_units
        step = int(unit_diff / (hidden_layers+1))

        self.hid = []
        for i in range(self.hidden_layers):
            self.hid.append(nn.Linear(input_units-i*step, input_units-(i+1)*step))
        self.opt = nn.Linear(input_units-hidden_layers*step, output_units)

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)


    def forward(self, x):
        for i, h in enumerate(self.hid):
            if self.activation == "sigmoid":
                x = torch.sigmoid(h(x))
            elif self.activation == "relu":
                x = torch.relu(h(x))

            if self.dropout_rate > 0:
                x = self.dropout(x)

        return torch.sigmoid(self.opt(x))


    def __str__(self):
        s = f"HNN(\n"
        for i, h in enumerate(self.hid):
            s += f"    (hid{i}): {h} (func={self.activation})"
            s += f" (dropout={self.dropout_rate})\n" if self.dropout_rate > 0 else "\n"
        s += f"    (opt): {self.opt} (func=sigmoid)\n"
        s += ")"
        return s
