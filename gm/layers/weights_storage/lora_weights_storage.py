from typing import List

import torch
from torch import nn
from torch.nn import init

from gm.layers.pseudo_layers.configs.lora_config import LoRAConfig
from gm.layers.shaped_layer import ShapedLayer
from gm.lora.init_strategy.base_lora_strategy import BaseLoRAStrategy
from gm.lora.base_lora import BaseLoRA


class LoRAWeightsStorage(nn.Module):
    # list of layers, can be used to get layer index
    _layers: List[ShapedLayer]
    # list of weights for each layer, [layers_count, current_layer_shapes]
    _storage: List[List[nn.Parameter]]
    # array of LoRA modules for each set of weights in each layer, if LoRA is not used for these heights,
    # then Null is stored
    _lora_modules: List[List[BaseLoRA | None]]
    # lora strategy, used for layer distribution
    _lora_strategy: BaseLoRAStrategy

    def __init__(
            self,
            lora_strategy: BaseLoRAStrategy,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self._layers = []
        self._storage = []
        self._lora_modules = []
        self._lora_strategy = lora_strategy

    def add_layer(
            self,
            layer: ShapedLayer
    ) -> int:
        layer_id = len(self._layers)
        self._layers.append(layer)
        return layer_id

    def build_storage(self):
        # init LoRA modules
        self._lora_modules = self._lora_strategy.distribute_lora_modules(self._layers)

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
            config: LoRAConfig,
    ) -> List[torch.Tensor]:

        return [
            weights @ self._lora_modules[layer_index][weights_idx]
            for weights_idx, weights in enumerate(self._storage[layer_index])
        ]
