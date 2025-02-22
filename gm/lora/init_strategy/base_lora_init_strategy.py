from abc import ABC, abstractmethod
from typing import List

from gm.layers.shaped_layer import ShapedLayer
from gm.lora.base_lora import BaseLoRA


class BaseLoRAStrategy(ABC):
    @abstractmethod
    def distribute_lora_modules(
            self,
            layers: List[ShapedLayer],
    ) -> List[List[BaseLoRA | None]]:
        """Spreads LoRA modules across model layers"""
        pass
