from __future__ import annotations
from typing import List

import torch
from torch import nn

from gm.layers.shaped_layer import ShapedLayer
from gm.layers.weights_storage.weights_storage import WeightsStorage


class PseudoLayer(ShapedLayer):
    _weights_storage: WeightsStorage
    _storage_index: int | bool
    _pseudo_shapes: List[torch.Size] | None

    def __init__(
            self,
            weights_storage: WeightsStorage,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self._weights_storage = weights_storage
        self._pseudo_shapes = None
        self._storage_index = False

    def register_layer(self):
        self._storage_index = True

        if self._pseudo_shapes is None:
            raise "define _pseudo_shapes attribute"

        self._storage_index = self._weights_storage.add_layer(self)

    def __call__(
            self,
            *args,
            **kwargs,
    ):
        if self._storage_index is None:
            raise "call PseudoLayer.register_layer() first"

        args, kwargs = self._weights_storage.get_argument_parsing_strategy.parse(*args, **kwargs)

        return super().__call__(
            self._weights_storage(self._storage_index, **kwargs),
            *args,
            **kwargs
        )

    @staticmethod
    def from_module(
            weights_storage: WeightsStorage,
            module: nn.Module,
    ) -> PseudoLayer:
        """Extracts the required parameters from the source module and creates a pseudo-layer instance"""
        raise NotImplementedError(f"Not implemented")

    @property
    def shapes(self) -> List[torch.Size]:
        return self._pseudo_shapes

    @property
    def storage_index(self):
        return self._storage_index
