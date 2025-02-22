from typing import List

import torch
from torch import nn


class LoRAMixin:
    shape: torch.Size

    def __init__(
            self,
            shape: torch.Size
    ):
        self.shape = shape

    def compute_lora_delta(self) -> List[nn.Parameter]:
        return ...
