from typing import Callable
from abc import ABC, abstractmethod
import weakref

import torch
from torch import nn
from torch.xpu import device as Dev

from gm.settings import CUDA_DEVICE
from gm.layers.weights_storage.weights_storage import WeightsStorage


class BaseLoRA(nn.Module, ABC):
    _weights_storage_ref: weakref.ReferenceType[WeightsStorage]
    _shape: torch.Size
    _rank: int
    _init_function: Callable
    _enabled: bool
    _dtype: torch.dtype
    _device: Dev

    def __init__(
            self,
            weights_storage: WeightsStorage,
            storage_index: int,
            layer_index: int,
            rank: int = 1,
            init_function=nn.init.xavier_uniform_,
            dtype: torch.dtype = torch.float16,
            device: Dev | None = None,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self._weights_storage_ref = weakref.ref(weights_storage)
        self._storage_index = storage_index
        self._layer_index = layer_index
        weights = self._weights_storage_ref().get_storage_parameter(
            storage_idx=self._storage_index,
            layer_idx=self._layer_index,
        )
        self._shape = weights.shape
        self._rank = rank
        self._init_function = init_function
        self._dtype = dtype
        self._device = device if device is not None else torch.device(CUDA_DEVICE)
        self._enabled = True

    @abstractmethod
    def compute_lora_delta(self) -> nn.Parameter:
        pass

    def enable_grad(self):
        for param in self.parameters():
            param.requires_grad = True

    def disable_grad(self):
        for param in self.parameters():
            param.requires_grad = False

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled
