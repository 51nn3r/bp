from typing import List

from gm.lora.enable_strategy.base_lora_enable_strategy import BaseLoraEnableStrategy


class LoraEnableAllStrategy(BaseLoraEnableStrategy):
    def get_activation_mask(self) -> List[bool]:
        return [True] * self._adapters_count
