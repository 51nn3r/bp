import random
from typing import List

from gm.lora.enable_strategy.base_lora_enable_strategy import BaseLoraEnableStrategy


class RandomSubsetEnableStrategy(BaseLoraEnableStrategy):
    """
    Simple random subset selection for LoRA adapters.
    Does not address any balance concerns.
    """

    def __init__(self, adapters_count: int, enabled_adapters_proportion: float = 1 / 3):
        super().__init__(adapters_count)
        self._adapters_count = adapters_count
        self._subset_size = int(self._adapters_count * enabled_adapters_proportion)

    def get_activation_mask(self) -> List[bool]:
        """
        Randomly selects a subset of adapters (of size subset_size)
        and returns a boolean mask indicating which adapters are active.
        """
        chosen_indices = random.sample(range(self._adapters_count), self._subset_size)
        mask = [False] * self._adapters_count
        for idx in chosen_indices:
            mask[idx] = True

        return mask
