from typing import Callable
from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseLoRA(nn.Module, ABC):
    _shape: torch.Size
    _rank: int
    _init_function: Callable
    _enabled: bool

    def __init__(
            self,
            shape: torch.Size,
            rank: int = 1,
            init_function=nn.init.xavier_uniform_,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self._shape = shape
        self._rank = rank
        self._init_function = init_function
        self._enabled = True

    @abstractmethod
    def compute_lora_delta(self) -> nn.Parameter:
        pass

    def enable(self):
        for param in self.parameters():
            param.requires_grad = True

        self._enabled = True

    def disable(self):
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled
