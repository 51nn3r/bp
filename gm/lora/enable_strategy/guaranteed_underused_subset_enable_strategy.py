import random
import math
from typing import List

from gm.lora.enable_strategy.base_lora_enable_strategy import BaseLoraEnableStrategy


class GuaranteedUnderusedSubsetEnableStrategy(BaseLoraEnableStrategy):
    """
    Guarantees that underused adapters are selected first,
    ensuring that the difference in activation counts does not grow too large.
    """

    def __init__(self, adapters_count: int, enabled_adapters_proportion: float = 1 / 3):
        super().__init__(adapters_count)
        self._adapters_count = adapters_count
        self._subset_size = int(self._adapters_count * enabled_adapters_proportion)
        self._activation_counts = [0] * self._adapters_count

    def get_activation_mask(self) -> List[bool]:
        """
        Always picks adapters with the minimum activation count first.
        If there aren't enough underused adapters to fill the subset, the
        remainder is selected randomly from the rest.
        """
        # Find the lowest activation count
        min_count = min(self._activation_counts)
        # Get all adapters whose counts are equal to min_count
        candidates = [i for i, count in enumerate(self._activation_counts) if count == min_count]

        # If we have enough underused adapters to fill the subset, sample among them
        if len(candidates) >= self._subset_size:
            chosen_indices = random.sample(candidates, self._subset_size)
        else:
            # Use all underused adapters plus random picks for the remainder
            chosen_indices = candidates.copy()
            remaining = list(set(range(self._adapters_count)) - set(candidates))
            needed = self._subset_size - len(candidates)
            chosen_indices.extend(random.sample(remaining, needed))

        # Update activation counts
        for idx in chosen_indices:
            self._activation_counts[idx] += 1

        # Build the final mask
        mask = [False] * self._adapters_count
        for idx in chosen_indices:
            mask[idx] = True

        return mask
