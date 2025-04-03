from abc import ABC
from typing import List, Callable

import torch
from torch import nn

from gm.layers.shaped_layer import ShapedLayer
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
            layers: List[ShapedLayer],
            rank: int = 1,
            alpha: float = 1.,
            dtype: torch.dtype = torch.float16,
            lora_dropout: float = 0.0,
    ) -> List[List[BaseLoRA | None]]:
        lora_adapters = []
        for layer in layers:
            layer_adapters = []
            for shape in layer.shapes:
                if len(shape) > 1:
                    layer_adapters.append(self._lora_cls(
                        shape=shape,
                        rank=rank,
                        alpha=alpha,
                        init_function=self._init_function,
                        dtype=dtype,
                        lora_dropout=lora_dropout,
                    ))
                else:
                    layer_adapters.append(None)

            lora_adapters.append(layer_adapters)

        return lora_adapters

