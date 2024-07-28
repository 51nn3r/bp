from typing import List

import torch

from gm.layers.shaped_layer import ShapedLayer
from gm.layers.weights_storage import WeightsStorage


class PseudoLayer(ShapedLayer):
    _weights_storage: WeightsStorage
    _storage_index: int
    _pseudo_shapes: List[torch.Size]

    def __init__(
            self,
            weights_storage: WeightsStorage,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self._weights_storage = weights_storage
        self._storage_index = weights_storage.add_layer(self)
        self._pseudo_shapes = []

    def __call__(
            self,
            selector: torch.Tensor,
            *args,
            **kwargs,
    ):
        return super().__call__(self._weights_storage(self._storage_index, selector), *args, **kwargs)

    @property
    def shapes(self) -> List[torch.Size]:
        return self._pseudo_shapes
