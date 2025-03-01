from typing import List, Optional

import torch
from torch import nn

from gm.layers.pseudo_layers.argument_parsing_strategy.argument_parsing_strategy import ArgumentParsingStrategy
from gm.layers.shaped_layer import ShapedLayer
from gm.layers.weights_storage.weights_storage import WeightsStorage
from gm.lora.init_strategy.base_lora_init_strategy import BaseLoRAInitStrategy
from gm.lora.base_lora import BaseLoRA


class LoRAWeightsStorage(WeightsStorage):
    # List of layers; used to retrieve layer indices.
    _layers: List[ShapedLayer]
    # List of weight placeholders for each layer; each element is a list of nn.Parameter placeholders.
    _storage: List[List[nn.Parameter]]
    # For each layer, a list of LoRA modules corresponding to each weight.
    # If LoRA is not applied for a given weight, the element is None.
    _lora_modules: List[List[Optional[BaseLoRA]]]
    # LoRA initialization strategy used to distribute LoRA modules across layers.
    _lora_strategy: BaseLoRAInitStrategy

    def __init__(
            self,
            argument_parsing_strategy: ArgumentParsingStrategy,
            lora_strategy: BaseLoRAInitStrategy,
            **kwargs,
    ):
        """
        Initialize the LoRAWeightsStorage.

        :param lora_strategy: The strategy to use for distributing LoRA modules among layers.
        :param kwargs: Additional keyword arguments passed to the parent WeightsStorage.
        """
        super().__init__(argument_parsing_strategy, **kwargs)

        self._layers = []
        self._storage = []
        self._lora_modules = []
        self._lora_strategy = lora_strategy

    def build_storage(
            self,
            rank: int = 1,
    ):
        """
        Initializes the LoRA modules and the actual weight parameters for all registered layers.

        First, it uses the provided LoRA strategy to distribute LoRA modules across layers.
        Then, it calls the parent's build_storage() to initialize and register the weight parameters.

        Note: This implementation assumes that the parent WeightsStorage.build_storage() properly
        updates self._storage (i.e. replaces meta tensor placeholders with real parameters) and registers them.
        """
        # Distribute and initialize LoRA modules for all layers.
        self._lora_modules = self._lora_strategy.distribute_lora_modules(self._layers, rank)
        for layer_idx, layer_modules in enumerate(self._lora_modules):
            for module_idx, module in enumerate(layer_modules):
                self.add_module(f'lora_{layer_idx}_{module_idx}', module)

        # Initialize weights using the parent's method.
        # This will create and initialize real weight parameters (e.g., using Xavier initialization),
        # replace the meta tensor placeholders in self._storage, and register each parameter.
        super().build_storage()

    def update_weights_and_reinit_lora(
            self,
            rank: int = 1,
    ):
        """
        Updates the main weights and reinitializes the LoRA modules.

        This method is useful when the main weights have been updated (for example, during fine-tuning)
        and the corresponding LoRA modules should be reinitialized (or redistributed) accordingly.

        The process is as follows:
          1. Update or reinitialize the main weights.
             (In this implementation, we assume that the main weights in _storage are already updated.)
          2. Re-distribute and reinitialize the LoRA modules using the current LoRA strategy.
        """
        # Update main weights
        for layer_idx, lora_module_group in enumerate(self._lora_modules):
            for parameters_idx, lora_module in enumerate(lora_module_group):
                if lora_module is None:
                    continue

                self._storage[layer_idx][parameters_idx] += lora_module.compute_lora_delta()

        # Reinitialize the LoRA modules based on the current state of the layers.
        self._lora_modules = self._lora_strategy.distribute_lora_modules(self._layers, rank)

    def forward(
            self,
            layer_index: int,
    ) -> List[torch.Tensor]:
        """
        Performs a forward pass for the given layer by summing the main weight parameters with their
        corresponding LoRA deltas.

        Instead of performing matrix multiplication, the LoRA delta is added to the main weight:
            output_weight = main_weight + loRA_delta

        :param layer_index: The index of the layer for which to obtain updated weights.
        :return: A list of updated weight tensors for the specified layer.
        """
        return [
            weights + self._lora_modules[layer_index][weights_idx].compute_lora_delta()
            if self._lora_modules[layer_index][weights_idx] is not None else weights
            for weights_idx, weights in enumerate(self._storage[layer_index])
        ]

    def eval(self):
        for layer_modules in self._lora_modules:
            for module in layer_modules:
                module.disable()

    def train(self, mode: bool = True):
        for layer_weights in self._storage:
            for weights in layer_weights:
                weights.requires_grad = False

        for layer_modules in self._lora_modules:
            for module in layer_modules:
                module.enable()
