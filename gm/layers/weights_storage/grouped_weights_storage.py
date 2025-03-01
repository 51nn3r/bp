from typing import List, Dict

import logging

import torch
from torch import nn

from gm.layers.pseudo_layers.argument_parsing_strategy.argument_parsing_strategy import ArgumentParsingStrategy
from gm.layers.pseudo_layers.configs.gm_config import GrossMachineConfig
from gm.layers.shaped_layer import ShapedLayer
from gm.utils import move_to_device
from gm.layers.weights_storage.weights_storage import WeightsStorage

from gm.settings import CPU_DEVICE


class GroupedWeightsStorage(WeightsStorage):
    _groups_count: int
    _storage_size: int

    # List of layers, can be used to get the layer index.
    _layers: List[ShapedLayer]
    # For each layer (by index), stores the assigned group index as a tensor (or None initially).
    _layers_distribution: List[torch.Tensor | None]
    # Offsets for each layer within the corresponding group (order matches _layers).
    _layers_offsets: List[int]
    # List of weights for each layer, organized as [number of layers, list of weight parameters per layer].
    _storage: List[List[nn.Parameter]]
    _device: torch._C.device | None

    ADDITIONAL_FORWARD_ARGUMENTS = ['selector']

    def __init__(
            self,
            argument_parsing_strategy: ArgumentParsingStrategy,
            groups_count: int,
            storage_size: int,
            device: torch._C.device | None = None,
            **kwargs,
    ):
        """
        Initialize GroupedWeightsStorage.

        :param groups_count: The number of groups to distribute layer parameters into.
        :param storage_size: The number of weight variants per parameter.
        :param device: Optional target device.
        """
        super().__init__(argument_parsing_strategy, **kwargs)

        self._groups_count = groups_count
        self._storage_size = storage_size
        self._layers = []
        self._layers_distribution = []
        self._layers_offsets = []
        self._storage = []
        self._device = device or CPU_DEVICE

    def add_layer(
            self,
            layer: ShapedLayer,
    ) -> int:
        """
        Registers a new layer and creates placeholder parameters for it using meta tensors.

        :param layer: The ShapedLayer instance to add.
        :return: The index of the added layer.
        """
        layer_idx = super().add_layer(
            layer=layer,
        )
        # Initialize the layer's group distribution as None; it will be determined later.
        self._layers_distribution.append(None)
        return layer_idx

    def build_storage(self):
        """
        Initializes the weights for all registered layers and computes a distribution of layers across groups.
        First, it calls the parent build_storage() to initialize weights.
        Then, it calculates how to distribute the total parameters among the specified number of groups.

        If the final group count does not match the desired count, a warning is logged.
        If any parameters remain unused, a critical log message is issued.
        """
        super().build_storage()

        # Gather all weight shapes from all layers.
        shapes = []
        for layer in self._layers:
            shapes += layer.shapes

        # Calculate total number of parameters across all layers.
        unused_params_count = sum(shape.numel() for shape in shapes)
        layers_count = len(self._layers)

        # Variables to control the distribution of parameters into groups.
        current_params_count = 0
        group_index = 0
        inverse_distribution: Dict[int, List[ShapedLayer]] = {}

        for layer in self._layers:
            # Calculate total parameter count for this layer.
            layer_params_count = sum(shape.numel() for shape in layer.shapes)
            unused_params_count -= layer_params_count
            current_params_count += layer_params_count

            # Initialize distribution list for current group if needed.
            if group_index not in inverse_distribution:
                inverse_distribution[group_index] = []

            inverse_distribution[group_index].append(layer)

            # Determine the index of this layer.
            layer_index = self._layers.index(layer)
            # Assign the current group index to this layer.
            self._layers_distribution[layer_index] = torch.tensor(group_index)

            # Calculate the average parameter count that should be assigned to the current group.
            avg_params_count = (unused_params_count + current_params_count) / (self._groups_count - group_index)
            layers_count -= 1

            # If the current group has accumulated enough parameters, move to the next group.
            if current_params_count >= avg_params_count:
                current_params_count = 0
                group_index += 1

        # Move the layers' distribution to the desired device, if specified.
        if self._device is not None:
            self._layers_distribution = move_to_device(self._layers_distribution, self._device)

        # Log a warning if the number of groups created does not match the desired groups count.
        if group_index != self._groups_count:
            logging.warning(
                f"Real groups count ({group_index}) != wanted groups count ({self._groups_count})"
            )

        # Log a critical error if not all parameters were used in the distribution.
        if unused_params_count > 0:
            logging.critical(f"Not all parameters are used: {unused_params_count} unused")

    def forward(
            self,
            layer_index: int,
            selector: torch.Tensor | None = None,
    ) -> List[torch.Tensor]:
        """
        Returns a list of weight tensors for the given layer based on a selector provided in the configuration.

        :param layer_index: The index of the layer whose weights are requested.
        :param selector: The selector is expected to have shape [batch_size, groups_count] with values in [0, storage_size].
        :return: A list of weight tensors for the specified layer, selected according to the provided selector.
        """

        if selector is None:
            raise ValueError(f'Set selector!')

        # Retrieve the group index for the given layer.
        group_index = self._layers_distribution[layer_index]
        # Use the group index to select the appropriate weight indices from the config.selector.
        # The selector tensor is indexed along the last dimension.
        weights_index: torch.Tensor = torch.squeeze(
            torch.index_select(selector, -1, group_index)
        )

        # Retrieve the list of weight parameters for the specified layer.
        layer_weights: List[nn.Parameter] = self._storage[layer_index]

        # For each weight parameter, select and return the subset of weights according to weights_index.
        return [torch.index_select(weights, 0, weights_index) for weights in layer_weights]
