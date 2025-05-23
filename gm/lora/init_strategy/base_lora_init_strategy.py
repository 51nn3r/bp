from abc import ABC, abstractmethod
from typing import List

from torch import nn

from gm.layers.weights_storage.weights_storage import WeightsStorage
from gm.lora.base_lora import BaseLoRA


class BaseLoRAInitStrategy(ABC):
    @abstractmethod
    def distribute_lora_modules(
            self,
            weights_storage: WeightsStorage,
            rank: int = 1,
            **kwargs,
    ) -> List[List[BaseLoRA | None]]:
        """Spreads LoRA modules across model layers"""
        pass
