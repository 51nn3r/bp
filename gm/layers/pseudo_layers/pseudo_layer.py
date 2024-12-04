from typing import List

import torch

from gm.layers.shaped_layer import ShapedLayer
from gm.layers.weights_storage import WeightsStorage


class PseudoLayer(ShapedLayer):
    _weights_storage: WeightsStorage
    _storage_index: int
    _pseudo_shapes: List[torch.Size] | None
    _registered: bool

    def __init__(
            self,
            weights_storage: WeightsStorage,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self._weights_storage = weights_storage
        self._pseudo_shapes = None
        self._registered = False

    def register_layer(self):
        self._registered = True

        if self._pseudo_shapes is None:
            raise "define _pseudo_shapes attribute"

        self._storage_index = self._weights_storage.add_layer(self)

    def __call__(
            self,
            selector: torch.Tensor,
            *args,
            **kwargs,
    ):
        if self._registered is False:
            raise "call PseudoLayer.register_layer() first"

        return super().__call__(self._weights_storage(self._storage_index, selector), *args, **kwargs)

    @property
    def shapes(self) -> List[torch.Size]:
        return self._pseudo_shapes
