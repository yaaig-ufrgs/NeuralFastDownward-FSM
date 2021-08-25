import torch
import torch.nn as nn

# H(euristic) Neural Network
class HNN(nn.Module):
    def __init__(
        self,
        input_units: int,
        hidden_units: int,
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
        self.activation = activation
        self.dropout_rate = dropout_rate

        unit_diff = input_units - output_units
        step = int(unit_diff / (hidden_layers+1))

        # self.hid = []
        # for i in range(self.hidden_layers):
        #     self.hid.append(nn.Linear(input_units-i*step, input_units-(i+1)*step))

        # TODO change how this is done later.
        if self.hidden_layers > 0:
            self.hid1 = nn.Linear(input_units-0*step, input_units-1*step) if hidden_units == -1 else nn.Linear(input_units, hidden_units)
        if self.hidden_layers > 1:
            self.hid2 = nn.Linear(input_units-1*step, input_units-2*step) if hidden_units == -1 else nn.Linear(hidden_units, hidden_units)
        if self.hidden_layers > 2:
            self.hid3 = nn.Linear(input_units-2*step, input_units-3*step) if hidden_units == -1 else nn.Linear(hidden_units, hidden_units)
        if self.hidden_layers > 3:
            self.hid4 = nn.Linear(input_units-3*step, input_units-4*step) if hidden_units == -1 else nn.Linear(hidden_units, hidden_units)
        if self.hidden_layers > 4:
            self.hid5 = nn.Linear(input_units-4*step, input_units-5*step) if hidden_units == -1 else nn.Linear(hidden_units, hidden_units)

        self.opt = nn.Linear(input_units-hidden_layers*step, output_units) if hidden_units == -1 else nn.Linear(hidden_units, output_units)

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)


    def forward(self, x):
        # for h in self.hid:
        #     if self.activation == "sigmoid":
        #         x = torch.sigmoid(h(x))
        #     elif self.activation == "relu":
        #         x = torch.relu(h(x))

        #     if self.dropout_rate > 0:
        #         x = self.dropout(x)

        if self.hidden_layers > 0:
            if self.activation == "sigmoid":
                x = torch.sigmoid(self.hid1(x))
            elif self.activation == "relu":
                x = torch.relu(self.hid1(x))
            if self.dropout_rate > 0:
                x = self.dropout(x)
        if self.hidden_layers > 1:
            if self.activation == "sigmoid":
                x = torch.sigmoid(self.hid2(x))
            elif self.activation == "relu":
                x = torch.relu(self.hid2(x))
            if self.dropout_rate > 0:
                x = self.dropout(x)
        if self.hidden_layers > 2:
            if self.activation == "sigmoid":
                x = torch.sigmoid(self.hid3(x))
            elif self.activation == "relu":
                x = torch.relu(self.hid3(x))
            if self.dropout_rate > 0:
                x = self.dropout(x)
        if self.hidden_layers > 3:
            if self.activation == "sigmoid":
                x = torch.sigmoid(self.hid4(x))
            elif self.activation == "relu":
                x = torch.relu(self.hid4(x))
            if self.dropout_rate > 0:
                x = self.dropout(x)
        if self.hidden_layers > 4:
            if self.activation == "sigmoid":
                x = torch.sigmoid(self.hid5(x))
            elif self.activation == "relu":
                x = torch.relu(self.hid5(x))
            if self.dropout_rate > 0:
                x = self.dropout(x)

        if self.activation == "sigmoid":
            return torch.sigmoid(self.opt(x))
            #return torch.flatten(self.opt(x))
        elif self.activation == "relu":
            return torch.relu(self.opt(x))
            #return torch.flatten(self.opt(x))


    # def __str__(self):
    #     s = f"HNN(\n"
    #     for i, h in enumerate(self.hid):
    #         s += f"    (hid{i}): {h} (func={self.activation})"
    #         s += f" (dropout={self.dropout_rate})\n" if self.dropout_rate > 0 else "\n"
    #     s += f"    (opt): {self.opt} (func=sigmoid)\n"
    #     s += ")"
    #     return s
