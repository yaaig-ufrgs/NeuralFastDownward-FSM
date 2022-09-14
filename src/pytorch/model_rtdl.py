import torch
import torch.nn as nn
from math import sqrt
import typing as ty

class ResNetRTDL(nn.Module):
    def __init__(
        self,
        input_units: int,
        hidden_units: int,
        output_units: int,
        num_layers: int,
        activation: str,
        output_layer: str,
        hidden_dropout: float,
        residual_dropout: float,
        linear_output: bool,
        use_bias: bool,
        use_bias_output: bool,
        weights_method: str,
        model: str,
    ):
        super(ResNetRTDL, self).__init__()
        self.input_units = input_units
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.num_layers = num_layers
        self.hidden_dropout = hidden_dropout
        self.residual_dropout = residual_dropout
        self.output_layer = output_layer
        self.linear_output = linear_output
        self.use_bias = use_bias
        self.use_bias_output = use_bias_output
        self.model = model

        self.first_layer = nn.Linear(input_units, hidden_units)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        'linear0': nn.Linear(hidden_units, hidden_units),
                        'linear1': nn.Linear(hidden_units, hidden_units),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        bias_output = False if self.use_bias is False else use_bias_output
        self.head = nn.Linear(hidden_units, output_units, bias=bias_output)

        self.activation = self.set_activation(activation)
        self.output_activation = self.set_output_activation(activation)

        # In PyTorch 1.9, the default initialization used is an adapted Kaiming,
        # "Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features))"
        # See:
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L92
        # https://github.com/pytorch/pytorch/issues/57109
        if weights_method != "default":
            self.initialize_weights(weights_method)

    def set_activation(self, activation: str) -> nn.functional:
        if activation == "sigmoid":
            return nn.functional.sigmoid
        elif activation == "relu":
            return nn.functional.relu
        elif activation == "leakyrelu":
            return nn.functional.leaky_relu
        else:
            raise NotImplementedError(f"{activation} function not implemented!")

    def set_output_activation(self, activation: str) -> nn.functional:
        if self.output_layer == "regression":
            return (
                nn.functional.relu
                if activation != "leakyrelu"
                else nn.functional.leaky_relu
            )
        else:
            raise NotImplementedError(
                f"{self.output_layer} not implemented for output layer!"
            )

    def initialize_weights(self, method):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if method == "rai":
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    m.weight, m.bias = RAI(fan_in, fan_out)
                elif "kaiming" in method:
                    self.set_kaiming_init(m, method)
                elif method == "sqrt_k":
                    k = 1.0 / m.in_features
                    self.set_uniform_init(m, a=-sqrt(k), b=sqrt(k))
                elif method == "1":
                    self.set_uniform_init(m, a=-1.0, b=1.0)
                elif method == "01":
                    self.set_uniform_init(m, a=0.0, b=1.0)
                elif "xavier" in method:
                    self.set_xavier_init(m, method)
                else:
                    raise NotImplementedError(
                        f"Weights method {method} not implemented!"
                    )

    def set_uniform_init(self, m: nn.modules.linear.Linear, a: float, b: float):
        torch.nn.init.uniform_(m.weight, a, b)
        if self.use_bias:
            torch.nn.init.uniform_(m.bias, a, b)

    def set_kaiming_init(
        self, m: nn.modules.linear.Linear, method: str, zero_bias: bool = True
    ):
        if "uniform" in method:
            torch.nn.init.kaiming_uniform_(m.weight)
        elif "normal" in method:
            torch.nn.init.kaiming_normal_(m.weight)
        if zero_bias and self.use_bias:
            torch.nn.init.zeros_(m.bias)

    def set_xavier_init(
        self,
        m: nn.modules.linear.Linear,
        method: str,
        zero_bias: bool = True,
        gain_: float = 1.0,
    ):
        if "uniform" in method:
            torch.nn.init.xavier_uniform_(m.weight, gain=gain_)
        elif "normal" in method:
            torch.nn.init.xavier_normal_(m.weight, gain=gain_)
        if zero_bias and self.use_bias:
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.first_layer(x)

        for layer in self.layers:
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            z = x
            z = layer['linear0'](z)
            z = self.activation(z)
            if self.hidden_dropout > 0:
                z = nn.functional.dropout(z, self.hidden_dropout, True) # training = True?
            z = layer['linear1'](z)
            if self.residual_dropout > 0:
                z = nn.functional.dropout(z, self.residual_dropout, True)
            x = x + z

        x = self.head(x)
        #x = x.squeeze(-1)
        return x

