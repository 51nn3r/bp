from typing import List, Optional

import torch
from torch import nn

from gm.layers.shaped_layer import ShapedLayer
from gm.layers.weights_storage.configs.lora_weights_storage_config import LoraWeightsStorageConfig
from gm.layers.weights_storage.weights_storage import WeightsStorage
from gm.lora.enable_strategy.base_lora_enable_strategy import BaseLoraEnableStrategy
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
    _lora_init_strategy: BaseLoRAInitStrategy
    # LaRA enable strategy. For example while using XGBLoRA we randomly deside which module should be enabled
    _lora_enable_strategy: BaseLoraEnableStrategy | None
    # Config
    _config: LoraWeightsStorageConfig

    def __init__(
            self,
            config: LoraWeightsStorageConfig,
            **kwargs,
    ):
        """
        Initialize the LoRAWeightsStorage.

        :param lora_init_strategy: The strategy to use for distributing LoRA modules among layers.
        :param kwargs: Additional keyword arguments passed to the parent WeightsStorage.
        """
        if config.lora_init_strategy is None:
            raise "Missing lora strategy"

        super().__init__(config, **kwargs, )

        self._layers = []
        self._storage = []
        self._lora_modules = []
        self._lora_init_strategy = config.lora_init_strategy

    def build_storage(
            self,
    ):
        """
        Initializes the LoRA modules and the actual weight parameters for all registered layers.

        First, it uses the provided LoRA strategy to distribute LoRA modules across layers.
        Then, it calls the parent's build_storage() to initialize and register the weight parameters.

        Note: This implementation assumes that the parent WeightsStorage.build_storage() properly
        updates self._storage (i.e. replaces meta tensor placeholders with real parameters) and registers them.
        """
        # Distribute and initialize LoRA modules for all layers.
        lora_modules = self._lora_init_strategy.distribute_lora_modules(
            weights_storage=self,
            rank=self._config.rank,
            svd_rank=self._config.svd_rank,
            alpha=self._config.alpha,
            lora_dropout=self._config.lora_dropout,
            dtype=self._config.dtype,
            device=self._config.device,
        )

        self._lora_enable_strategy = self._config.lora_enable_strategy_cls(
            adapters_count=len(lora_modules),
            enabled_adapters_proportion=self._config.enabled_adapters_proportion,
        )

        for layer_modules, should_be_activated in zip(
                self._lora_modules,
                self._lora_enable_strategy.get_activation_mask()
        ):
            for lora_module in layer_modules:
                if lora_module is None:
                    continue

                lora_module.to(self._device)
                if should_be_activated:
                    lora_module.enable()
                else:
                    lora_module.disable()

        self._lora_modules = lora_modules

        for layer_idx, layer_modules in enumerate(self._lora_modules):
            for module_idx, module in enumerate(layer_modules):
                self.add_module(f'lora_{layer_idx}_{module_idx}', module)

        # Initialize weights using the parent's method.
        # This will create and initialize real weight parameters (e.g., using Xavier initialization),
        # replace the meta tensor placeholders in self._storage, and register each parameter.
        super().build_storage()

    def set_enable_strategy(
            self,
            enable_strategy: BaseLoraEnableStrategy,
    ):
        self._lora_enable_strategy = enable_strategy

    def apply_weights(
            self,
    ):
        for layer_idx, lora_module_group in enumerate(self._lora_modules):
            for parameters_idx, lora_module in enumerate(lora_module_group):
                if lora_module is None or not lora_module.enabled:
                    continue

                with torch.no_grad():
                    self._storage[layer_idx][parameters_idx] += lora_module.compute_lora_delta()

    def reset_lora(self):
        for layer_modules, should_be_activated in zip(
                self._lora_modules,
                self._lora_enable_strategy.get_activation_mask()
        ):
            for lora_module in layer_modules:
                if lora_module is not None:
                    if should_be_activated:
                        lora_module.reset_matrices()
                        lora_module.enable()
                        lora_module.enable_grad()
                    else:
                        lora_module.disable()

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
            if self._lora_modules[layer_index][weights_idx] is not None and
               self._lora_modules[layer_index][weights_idx].enabled else weights
            for weights_idx, weights in enumerate(self._storage[layer_index])
        ]

    def disable_lora(self):
        for layer_modules in self._lora_modules:
            for module in layer_modules:
                if module is None:
                    continue

                module.disable()

    def enable_lora(self):
        for layer_modules in self._lora_modules:
            for module in layer_modules:
                if module is None:
                    continue

                module.enable()

    def eval(self):
        super().eval()
        for layer_modules in self._lora_modules:
            for module in layer_modules:
                if module is None:
                    continue

                module.eval()

    def train(self, mode: bool = True):
        super().train()
        for layer_modules in self._lora_modules:
            for module in layer_modules:
                if module is None:
                    continue

                module.train()

    def disable_grad(self):
        super().eval()
        for layer_modules in self._lora_modules:
            for module in layer_modules:
                if module is None:
                    continue

                module.disable_grad()

    def enable_grad(self):
        super().train()
        for layer_modules in self._lora_modules:
            for module in layer_modules:
                if module is None:
                    continue

                module.enable_grad()

    @property
    def lora_modules(self) -> List[List[BaseLoRA]]:
        return self._lora_modules

    @property
    def enabled_modules_count(self) -> int:
        count = 0
        for modules in self._lora_modules:
            for module in modules:
                if module is not None and module.enabled:
                    count += 1

        return count
