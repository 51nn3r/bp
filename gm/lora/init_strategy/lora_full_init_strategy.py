from abc import ABC
from typing import List, Callable

import torch
from torch import nn

from gm.layers.weights_storage.weights_storage import WeightsStorage
from gm.lora.base_lora import BaseLoRA

from gm.lora.init_strategy.base_lora_init_strategy import BaseLoRAInitStrategy


class LoRAFullInitStrategy(BaseLoRAInitStrategy, ABC):
    _lora_cls: type[BaseLoRA]
    _init_function: Callable

    def __init__(
            self,
            lora_cls: type[BaseLoRA],
            init_function=nn.init.xavier_uniform_,
    ):
        self._lora_cls = lora_cls
        self._init_function = init_function

    def distribute_lora_modules(
            self,
            weights_storage: WeightsStorage,
            rank: int = 1,
            svd_rank: int = 1,
            svd_config: int = 1,
            alpha: float = 1.,
            lora_dropout: float = 0.0,
            dtype: torch.dtype = torch.float16,
            device: torch.xpu.device | None = None,
            **kwargs,
    ) -> List[List[BaseLoRA | None]]:
        lora_adapters = []
        for storage_index in range(len(weights_storage.storage)):
            layer_adapters = []
            for layer_index in range(len(weights_storage.storage[storage_index])):
                weight = weights_storage.get_storage_parameter(
                    storage_idx=storage_index,
                    layer_idx=layer_index,
                )
                shape = weight.shape
                if len(shape) > 1:
                    layer_adapters.append(self._lora_cls(
                        weights_storage=weights_storage,
                        storage_index=storage_index,
                        layer_index=layer_index,
                        rank=rank,
                        svd_rank=svd_rank,
                        alpha=alpha,
                        init_function=self._init_function,
                        lora_dropout=lora_dropout,
                        dtype=dtype,
                        device=device,
                    ))
                else:
                    layer_adapters.append(None)

            lora_adapters.append(layer_adapters)

        return lora_adapters
