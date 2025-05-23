import random
import math
from typing import List

from gm.lora.enable_strategy.base_lora_enable_strategy import BaseLoraEnableStrategy


class WeightedSubsetEnableStrategy(BaseLoraEnableStrategy):
    """
    Uses weighted sampling to increase the probability of
    selecting underused adapters without guaranteeing selection.
    """

    def __init__(self, adapters_count: int, enabled_adapters_proportion: float = 1 / 3):
        super().__init__(adapters_count)
        self._adapters_count = adapters_count
        self._subset_size = int(self._adapters_count * enabled_adapters_proportion)
        # Track how many times each adapter has been activated
        self._activation_counts = [0] * self._adapters_count
        # Counts the total number of calls made to get_activation_mask
        self._call_count = 0

    def get_activation_mask(self) -> List[bool]:
        """
        Increases the probability of picking adapters that are underused.
        Adapters that are far behind the 'expected' activation count get a higher weight,
        thus a higher chance of being included in the subset.
        """
        self._call_count += 1
        # Expected activations for each adapter so far
        expected_count = self._call_count * (self._subset_size / self._adapters_count)

        # Compute weights based on how far behind each adapter is
        weights = []
        for count in self._activation_counts:
            error = expected_count - count
            # If adapter is overused (error < 0), give it a small epsilon
            weights.append(error if error > 0 else 0.01)

        # Efraimidis-Spirakis sampling without replacement:
        # key_i = -log(U) / weight_i
        # then pick subset_size smallest keys
        keys = []
        for i, w in enumerate(weights):
            # Generate random key
            key = -math.log(random.random()) / w
            keys.append((key, i))

        # Sort by key and pick the first subset_size items
        keys.sort(key=lambda x: x[0])
        chosen_indices = [i for _, i in keys[:self._subset_size]]

        # Update activation counts
        for idx in chosen_indices:
            self._activation_counts[idx] += 1

        # Build the final mask
        mask = [False] * self._adapters_count
        for idx in chosen_indices:
            mask[idx] = True

        return mask
