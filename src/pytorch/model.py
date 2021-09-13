from src.pytorch.utils.default_args import DEFAULT_WEIGHTS_METHOD
import torch
import torch.nn as nn
import random
from math import sqrt

# H(euristic) Neural Network
class HNN(nn.Module):
    def __init__(
        self,
        input_units: int,
        hidden_units: [int],
        output_units: int,
        hidden_layers: int,
        activation: str,
        output_layer: str,
        dropout_rate: float,
        linear_output: bool,
        use_bias: bool,
        weights_method: str,
        weights_seed: int,
    ):
        super(HNN, self).__init__()
        self.input_units = input_units
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.output_layer = output_layer
        self.linear_output = linear_output

        hu = [input_units]
        if len(hidden_units) == 0:  # scalable
            unit_diff = input_units - output_units
            step = int(unit_diff / (hidden_layers + 1))
            for i in range(self.hidden_layers):
                hu.append(input_units - (i + 1) * step)
        elif len(hidden_units) == 1:  # all the same
            hu += hidden_units * self.hidden_layers
        else:
            hu += hidden_units

        self.hid = nn.ModuleList()
        for i in range(self.hidden_layers):
            self.hid.append(nn.Linear(hu[i], hu[i + 1], bias=use_bias))
        self.opt = nn.Linear(hu[-1], output_units, bias=use_bias)

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate, inplace=False)

        if activation == "sigmoid":
            self.activation = torch.sigmoid
        elif activation == "relu":
            self.activation = torch.relu
        else:
            raise NotImplementedError(f"{activation} function not implemented!")

        if output_layer == "regression":
            self.output_activation = torch.relu
        elif output_layer == "prefix":
            self.output_activation = torch.sigmoid
        elif output_layer == "one-hot":
            self.output_activation = torch.softmax
        else:
            raise NotImplementedError(
                f"{output_layer} not implemented for output layer!"
            )

        if weights_method != DEFAULT_WEIGHTS_METHOD:
            self.initialize_weights(weights_method, weights_seed)


    def initialize_weights(self, method, seed):
        random.seed(seed)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if method == "sqrt_k":
                    k = 1.0/m.in_features
                    a, b = -sqrt(k), sqrt(k)
                    nn.init.uniform_(m.weight, a, b)
                    nn.init.uniform_(m.bias, a, b)
                elif method == "1":
                    a, b = -1.0, 1.0
                    nn.init.uniform_(m.weight, a, b)
                    nn.init.uniform_(m.bias, a, b)
                elif "xavier" in method:
                    gain = 1.0

                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    std = gain * sqrt(2.0 / float(fan_in+fan_out))
                    if "uniform" in method:
                        a = sqrt(3.0) * std
                        with torch.no_grad():
                            nn.init.uniform_(m.weight, -a, a)
                    elif "normal" in method:
                        with torch.no_grad():
                            nn.init.normal_(m.weight, 0.0, std)
                    
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.bias)
                    std = gain * sqrt(2.0 / float(fan_in+fan_out))
                    if "uniform" in method:
                        a = sqrt(3.0) * std
                        with torch.no_grad():
                            nn.init.uniform_(m.bias, -a, a)
                    elif "normal" in method:
                        with torch.no_grad():
                            nn.init.normal_(m.bias, 0.0, std)
                else:
                    raise NotImplementedError(f"Weights method {method} not implemented!")
                    
    def forward(self, x):
        for h in self.hid:
            x = self.activation(h(x))
            if self.dropout_rate > 0:
                x = self.dropout(x)

        if self.linear_output:
            return self.opt(x)
        return self.output_activation(self.opt(x))
