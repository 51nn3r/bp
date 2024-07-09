from typing import List, Tuple

import tensorflow as tf
from keras.layers import Layer

from gm.weights_manager import WeightsManager


class PseudoLayer(Layer):
    _weights_manager: WeightsManager | None

    def __init__(
            self,
            weights_manager: WeightsManager,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self._weights_manager = weights_manager
        self._built_from_signature = False

    def __call__(self, *args, **kwargs):
        if self.built:
            return super().__call__(*args, **kwargs)

        if not self._built_from_signature and args:
            self.build_from_signature(args[0].shape)

        weights = self._weights_manager.fetch_weights(self)

        return super().__call__(weights, *args, **kwargs)

    def build_from_signature(self, input_shape: tf.TensorShape):
        pass

    @property
    def params_count(self) -> int:
        return 0

    @property
    def pseudo_shapes(self) -> List[tf.TensorShape] | tf.TensorShape | None:
        return None
