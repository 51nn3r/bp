from abc import ABC
from typing import List, Callable

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
    ) -> List[List[BaseLoRA | None]]:
        return [[
            self._lora_cls(shape, rank, self._init_function) for shape in layer.shapes
        ] for layer in layers]
