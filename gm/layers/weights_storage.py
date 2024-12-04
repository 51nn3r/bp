from typing import List
from typing import Dict

import logging

import torch
from torch import nn
from torch.nn import init

from gm.layers.shaped_layer import ShapedLayer
from gm.utils import move_to_device


class WeightsStorage(nn.Module):
    _groups_count: int
    _storage_size: int

    # list of layers, can be used to get layer index
    _layers: List[ShapedLayer]
    # layer_index -> group_index
    _layers_distribution: List[torch.Tensor | None]
    # offsets for each layer in the corresponding group, the order matches _layers
    _layers_offsets: List[int]
    # list of weights for each layer, [layers_count, current_layer_shapes]
    _storage: List[List[nn.Parameter]]
    _device: torch._C.device | None

    def __init__(
            self,
            groups_count: int,
            storage_size: int,
            device: torch._C.device | None,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self._groups_count = groups_count
        self._storage_size = storage_size
        self._layers = []
        self._layers_distribution = []
        self._layers_offsets = []
        self._storage = []
        self._device = device

    def add_layer(
            self,
            layer: ShapedLayer
    ) -> int:
        layer_id = len(self._layers)
        self._layers.append(layer)
        self._layers_distribution.append(None)
        return layer_id

    def build_storage(self):
        # init useful params
        shapes = []
        for layer in self._layers:
            shapes += layer.shapes

        unused_params_count = sum(shape.numel() for shape in shapes)
        layers_count = len(self._layers)

        # create distribution
        current_params_count = 0
        group_index = 0
        inverse_distribution: Dict[int: List[ShapedLayer]] = {}
        for layer in self._layers:
            layer_params_count = sum(shape.numel() for shape in layer.shapes)
            unused_params_count -= layer_params_count
            current_params_count += layer_params_count

            if group_index not in inverse_distribution:
                inverse_distribution[group_index] = []

            inverse_distribution[group_index].append(layer)
            if layer not in self._layers:
                raise "no such layer"

            layer_index = self._layers.index(layer)
            self._layers_distribution[layer_index] = torch.tensor(group_index)

            avg_params_count = (unused_params_count + current_params_count) / (self._groups_count - group_index)
            layers_count -= 1
            if current_params_count >= avg_params_count:
                current_params_count = 0
                group_index += 1

        if self._device is not None:
            self._layers_distribution = move_to_device(self._layers_distribution, self._device)

        # self._layers_distribution = self._layers_distribution.to

        if group_index != self._groups_count:
            logging.warning(f'real groups count ({group_index}) != wanted groups count ({self._groups_count})')

        if unused_params_count > 0:
            logging.critical(f'not all params are used: {unused_params_count} unused')

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
            selector: torch.Tensor,
    ) -> List[torch.Tensor]:
        """  selector shape is [batch_size, groups_count] from [0, storage_size]  """

        group_index = self._layers_distribution[layer_index]
        weights_index: torch.Tensor = torch.squeeze(torch.index_select(selector, -1, group_index))

        layer_weights: List[nn.Parameter] = self._storage[layer_index]

        return [torch.index_select(weights, 0, weights_index) for weights in layer_weights]
