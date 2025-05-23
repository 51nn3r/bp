from typing import List

import torch
from torch import nn
from torch.nn import init

from gm.layers.pseudo_layers.argument_parsing_strategy.argument_parsing_strategy import ArgumentParsingStrategy
from gm.layers.shaped_layer import ShapedLayer
from gm.layers.weights_storage.configs.weights_storage_config import WeightsStorageConfig

from gm.settings import META_DEVICE, CPU_DEVICE


class WeightsStorage(nn.Module):
    # List of layers, can be used to get layer index
    _layers: List[ShapedLayer]
    # List of weights for each layer, [layers_count, current_layer_shapes]
    _storage: List[List[nn.Parameter]]
    _device: torch._C.device | None
    # Config
    _config: WeightsStorageConfig

    def __init__(
            self,
            config: WeightsStorageConfig,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self._device = config.device if config.device is not None else torch.device(CPU_DEVICE)
        self._config = config
        self._layers = []
        self._storage = []

    def add_layer(
            self,
            layer: ShapedLayer,
    ) -> int:
        """
        Registers a new layer and creates a placeholder for its weights.
        The placeholder is constructed using meta tensors so that it has the correct shape
        without allocating real memory.

        :param layer: The ShapedLayer instance to add.
        :return: The index of the added layer.
        """
        layer_idx = len(self._layers)
        self._layers.append(layer)
        # For each expected weight shape in the layer, create a placeholder parameter on the META_DEVICE
        self._storage.append([
            nn.Parameter(torch.empty(shape, device=META_DEVICE, dtype=self._config.dtype)) for shape in layer.shapes
        ])
        return layer_idx

    def build_storage(self):
        """
        Initializes the weights for all registered layers.
        For each layer, creates new parameters with shape: (storage_size, *weight_shape),
        initializes them using Xavier normal initialization, and registers them as module parameters.
        """
        for layer_idx, layer in enumerate(self._layers):

            # Create a new list of weight parameters for the current layer.
            layer_weights = [
                nn.Parameter(torch.Tensor(shape))
                for shape in layer.shapes
            ]

            # Initialize each weight parameter using Xavier normal initialization.
            for weights in layer_weights:
                if weights.dim() > 1:
                    init.xavier_normal_(weights)
                else:
                    init.normal_(weights)

            # Replace the placeholder parameters with the newly initialized weights,
            # and register each parameter with the module.
            for weight_idx, weights in enumerate(layer_weights):
                if self._storage[layer_idx][weight_idx].is_meta:
                    self._storage[layer_idx][weight_idx] = weights
                    self.register_parameter(f'l{layer_idx}w{weight_idx}', weights)

    def set_parameters(
            self,
            layer_id: int,
            parameters: List[nn.Parameter],
    ):
        """
        Updates the weight parameters for the specified layer.
        This method can be called either immediately after registering a layer
        (when placeholder parameters exist) or after all layers have been registered.

        :param layer_id: The index of the layer to update.
        :param parameters: A list of new nn.Parameter objects for that layer.
        """
        layer_parameters = self._storage[layer_id]
        if len(layer_parameters) != len(parameters):
            raise ValueError(
                f"Parameter count mismatch for layer {layer_id}: "
                f"expected {len(layer_parameters)}, got {len(parameters)}"
            )

        for idx, param in enumerate(parameters):
            if param.shape != layer_parameters[idx].shape:
                raise ValueError(
                    f"Shape mismatch for parameter at index {idx} in layer {layer_id}: "
                    f"expected {layer_parameters[idx].shape}, got {param.shape}"
                )

            layer_parameters[idx] = param
            self.register_parameter(f'l{layer_id}w{idx}', param)

    def reset_parameters(self):
        for layer in self._storage:
            for param in layer:
                if len(param.shape) > 1:
                    nn.init.xavier_uniform_(param)

    def get_storage_parameter(
            self,
            storage_idx: int,
            layer_idx: int,
    ) -> nn.Parameter:
        return self._storage[storage_idx][layer_idx]

    def forward(
            self,
            layer_index: int,
    ) -> List[nn.Parameter]:
        """
        Retrieves the weight parameters for the given layer, optionally based on a configuration.

        :param layer_index: The index of the layer for which weights are requested.
        :return: A list of nn.Parameter objects for the specified layer.
        """
        return self._storage[layer_index]

    @property
    def storage(self) -> List[List[nn.Parameter]]:
        return self._storage

    @property
    def get_argument_parsing_strategy(self) -> ArgumentParsingStrategy:
        return self._config.argument_parsing_strategy
