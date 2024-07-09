from typing import List
from typing import Dict

import tensorflow as tf
from keras.layers import Layer


class WeightsStorage(Layer):
    _size: int
    _groups_count: int
    _layers: Dict[str, List[tf.Variable]]
    _part_index: int
    _grouped: bool
    _weights_distribution: Dict[str, int]

    def __init__(
            self,
            groups_count: int,
            size: int,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self._size = size
        self._groups_count = groups_count
        self._layers: Dict[str, List[tf.Variable]] = {}
        self._weights_distribution: Dict[str, int] = {}
        self._grouped = False

    def set_indices(
            self,
            indices,
    ):
        self._part_index = indices

    def add_layer(
            self,
            layer: Layer,
    ):
        if layer.name in self._layers:
            return

        self._layers[layer.name] = [
            self.add_weight(shape=[self._size] + shape, dtype=self.dtype, trainable=True)
            for shape in layer.pseudo_shapes
        ]

    def add_layers_list(
            self,
            layers: List[Layer],
    ):
        for layer in layers:
            self.add_layer(layer)

    def _group_layers(self):
        weights_count = tf.size([weights for layer in self._layers for weights in layer])
        target_sum = tf.cast(weights_count / self._groups_count, tf.int32)
        layers_count = len(self._layers)
        current_sum = used_layers_count = full_groups_count = 0

        for key, weights in self._layers.items():
            weights = self._layers[key]
            current_weights_count = sum(tf.size(w) for w in weights)

            if (current_sum + current_weights_count < target_sum and
                self._groups_count - full_groups_count < layers_count - used_layers_count) or \
                    full_groups_count >= self._groups_count - 1:

                current_sum += current_weights_count
            else:
                current_sum = current_weights_count
                full_groups_count += 1

            self._weights_distribution[key] = full_groups_count

            used_layers_count += 1

    def call(
            self,
            layer_name: str,
            indices,
    ) -> List[tf.Variable] | tf.Variable:
        """  indices - indices of each part of pseudo model (like time steps)  """

        if self._grouped is False:
            self._group_layers()

        if layer_name not in self._layers:
            raise "no such layer"

        weights_index = indices[:, self._weights_distribution[layer_name]]

        return [
            tf.gather(weights, weights_index)
            for weights in self._layers[layer_name]
        ]

    @property
    def layers(self):
        return self._layers
