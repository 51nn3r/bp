import tensorflow as tf
from keras.layers import Layer
from keras.layers import Input

from gm.layers.weights_storage import WeightsStorage


class WeightsManager:
    _storage_size: int
    _storages_count: int
    _input: tf.Tensor
    _storage: WeightsStorage

    def __init__(
            self,
            storage_groups_count: int,
            storage_size: int,
    ):
        self._storage_size = storage_size
        self._input = Input((storage_groups_count, storage_size,))
        self._storage = WeightsStorage(
            storage_groups_count,
            storage_size,
        )

    def fetch_weights(
            self,
            layer: Layer,
    ):
        if layer.name not in self._storage.layers:
            self._storage.add_layer(layer)

        return self._storage(layer.name, tf.argmax(self.input, axis=-1))

    @property
    def input(self) -> tf.Tensor:
        return self._input
