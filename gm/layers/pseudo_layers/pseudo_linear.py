from typing import List

import torch

from gm.layers.pseudo_layers.pseudo_layer import PseudoLayer

from gm.layers.weights_storage import WeightsStorage


class PseudoLinear(PseudoLayer):
    _in_features: int
    _out_features: int

    _kernel_shape: torch.Size
    _bias_shape: torch.Size

    def __init__(
            self,
            weights_storage: WeightsStorage,
            in_features: int,
            out_features: int,
            **kwargs
    ):
        super().__init__(weights_storage, **kwargs)

        self._in_features = in_features
        self._out_features = out_features

        self._kernel_shape = torch.Size([out_features, in_features])
        self._bias_shape = torch.Size([out_features])

        self._pseudo_shapes = [self._kernel_shape, self._bias_shape]

        self.register_layer()

    def forward(
            self,
            weights: List[torch.Tensor],
            inputs: torch.Tensor,
    ):
        kernel, bias = weights
        output = torch.bmm(kernel, inputs.unsqueeze(-1)).squeeze() + bias

        return output
