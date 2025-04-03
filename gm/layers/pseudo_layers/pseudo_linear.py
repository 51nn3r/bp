from __future__ import annotations
from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from gm.layers.pseudo_layers.pseudo_layer import PseudoLayer

from gm.layers.weights_storage.weights_storage import WeightsStorage


class PseudoLinear(PseudoLayer):
    _in_features: int
    _out_features: int
    _use_bias: bool

    _kernel_shape: torch.Size
    _bias_shape: torch.Size

    def __init__(
            self,
            weights_storage: WeightsStorage,
            in_features: int,
            out_features: int,
            use_bias: bool = True,
            **kwargs,
    ):
        super().__init__(weights_storage, **kwargs)

        self._in_features = in_features
        self._out_features = out_features
        self._use_bias = use_bias

        self._kernel_shape = torch.Size([out_features, in_features])
        self._bias_shape = torch.Size([out_features])

        self._pseudo_shapes = [self._kernel_shape]
        if self._use_bias:
            self._pseudo_shapes.append(self._bias_shape)

        self.register_layer()

    @staticmethod
    def from_module(
            weights_storage: WeightsStorage,
            module: nn.Linear,
    ) -> PseudoLayer:
        """Extracts the required parameters from the source module and creates a pseudo-layer instance"""
        pseudo_linear = PseudoLinear(
            weights_storage=weights_storage,
            in_features=module.in_features,
            out_features=module.out_features,
            use_bias=module.bias is not None,
        )
        weights_storage.set_parameters(pseudo_linear.storage_index, list(module.parameters()))

        return pseudo_linear

    '''
    def forward(
            self,
            weights: List[torch.Tensor],
            inputs: torch.Tensor,
            **kwargs,
    ):
        kernel, bias = weights
        output = torch.bmm(kernel, inputs.unsqueeze(-1)).squeeze() + bias

        return output
    '''

    def forward(
            self,
            weights: List[torch.Tensor],
            inputs: torch.Tensor,
            **kwargs,
    ):
        if self._use_bias:
            kernel, bias = weights
        else:
            kernel = weights[0]
            bias = None

        return F.linear(inputs, kernel, bias)

    @property
    def weight(self):
        return self._weights_storage.forward(self._storage_index)[0]

    @property
    def bias(self):
        return self._weights_storage.forward(self._storage_index)[1]
