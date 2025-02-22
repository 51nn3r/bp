from typing import List

import torch
from torch import nn
from torch.nn import init

from gm.layers.pseudo_layers.configs.pseudo_layer_config import PseudoLayerConfig
from gm.layers.shaped_layer import ShapedLayer


class WeightsStorage(nn.Module):
    # list of layers, can be used to get layer index
    _layers: List[ShapedLayer]
    # list of weights for each layer, [layers_count, current_layer_shapes]
    _storage: List[List[nn.Parameter]]
    _device: torch._C.device | None

    def __init__(
            self,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self._layers = []
        self._storage = []

    def add_layer(
            self,
            layer: ShapedLayer
    ) -> int:
        layer_id = len(self._layers)
        self._layers.append(layer)
        return layer_id

    def build_storage(self):
        # init weights
        for layer in self._layers:
            layer_weights = [nn.Parameter(torch.Tensor(torch.Size([self._storage_size]) + shape))
                             for shape in layer.shapes]

            for weights in layer_weights:
                init.xavier_normal(weights)

            self._storage.append(layer_weights)

        for layer_index, layer_weights in enumerate(self._storage):
            for weight_index, weights in enumerate(layer_weights):
                self.register_parameter(f'l{layer_index}w{weight_index}', weights)

    def forward(
            self,
            layer_index: int,
            config: PseudoLayerConfig,
    ) -> List[torch.Tensor]:
        return self._storage[layer_index]
