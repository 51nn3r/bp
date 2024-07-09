from typing import List

import string

import tensorflow as tf

from gm.layers.pseudo_layer import PseudoLayer

_CHR_IDX = string.ascii_lowercase


class PseudoDense(PseudoLayer):
    _units: int
    _kernel_shape: tf.TensorShape
    _bias_shape: tf.TensorShape
    _params_count: int
    _dot_product_equation: str

    def __init__(
            self,
            units: int,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self._units = units

        self._built_from_signature = False

    def build_from_signature(
            self,
            input_shape: tf.TensorShape,
    ):
        self._built_from_signature = True

        self._kernel_shape = tf.TensorShape((input_shape[-1], self._units,))
        self._bias_shape = tf.TensorShape((self._units,))
        self._params_count = self._kernel_shape.num_elements() + \
                             self._bias_shape.num_elements()

        extra_axes = input_shape.rank - 1
        extension = _CHR_IDX[-extra_axes:]
        small_extension = _CHR_IDX[-extra_axes:-2]
        self._dot_product_equation = f'{extension}a,{small_extension}ab->{extension}b'

    def call(
            self,
            weights,
            inputs,
            *args,
            **kwargs
    ):
        if not self._built_from_signature:
            self.build_from_signature(inputs.shape)

        kernel, bias = weights

        pod_product = tf.einsum(self._dot_product_equation, inputs, kernel)
        output = pod_product + bias

        return output

    @property
    def kernel_shape(self) -> tf.TensorShape:
        return self._kernel_shape

    @property
    def bias_shape(self) -> tf.TensorShape:
        return self._bias_shape

    @property
    def params_count(self) -> int:
        return self._params_count

    @property
    def pseudo_shapes(self) -> List[tf.TensorShape] | tf.TensorShape | None:
        return [
            self._kernel_shape,
            self._bias_shape,
        ]
