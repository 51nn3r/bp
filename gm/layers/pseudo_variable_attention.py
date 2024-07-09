from typing import List

import string
import math

import tensorflow as tf
from keras.src.layers import activation

from gm.layers.pseudo_layer import PseudoLayer

_CHR_IDX = string.ascii_lowercase


class PseudoVariableAttention(PseudoLayer):
    _num_heads: int
    _key_dim: int
    _out_types: int | None
    _out_steps: int | None
    _out_depth: int | None
    _qkv_shape: tf.TensorShape
    _convert_types_dense_shape: tf.TensorShape
    _convert_steps_dense_shape: tf.TensorShape
    _output_dense_shape: tf.TensorShape
    _params_count: int
    _qkv_equation: str
    _scores_equation: str
    _apply_scores_equation: str
    _convert_types_dense_equation: str
    _convert_steps_dense_equation: str
    _output_dense_equation: str

    def __init__(
            self,
            num_heads: int,
            key_dim: int,
            out_types=None,
            out_steps=None,
            out_depth=None,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self._num_heads = num_heads
        self._key_dim = key_dim
        self._out_types = out_types
        self._out_steps = out_steps
        self._out_depth = out_depth

        self._built_from_signature = False

    def build_from_signature(
            self,
            input_shape: tf.TensorShape,
    ):
        self._built_from_signature = True

        types = input_shape[-3]
        steps = input_shape[-2]
        depth = input_shape[-1]

        if self._out_types is None:
            self._out_types = types

        if self._out_steps is None:
            self._out_steps = steps

        if self._out_depth is None:
            self._out_depth = depth

        self._qkv_shape = tf.TensorShape((types, depth, self._num_heads, self._key_dim,))
        self._convert_types_dense_shape = tf.TensorShape(
            (self._num_heads, self._key_dim, types, self._out_types,)) if \
            types != self._out_types else tf.TensorShape(0, )

        self._convert_steps_dense_shape = tf.TensorShape(
            (self._num_heads, self._key_dim, steps, self._out_steps,)) if \
            steps != self._out_steps else tf.TensorShape(0, )

        self._output_dense_shape = tf.TensorShape((self._num_heads, self._key_dim, self._out_depth,))

        self._params_count = self._qkv_shape.num_elements() * 3 + \
                             self._convert_types_dense_shape.num_elements() + \
                             self._convert_steps_dense_shape.num_elements() + \
                             self._output_dense_shape.num_elements()

        extra_axes = input_shape.rank - 3

        if extra_axes:
            extension = _CHR_IDX[-extra_axes:]
        else:
            extension = ''

        # (batches, types, steps, depth) * (types, depth, heads, key_dim) -> (batches, heads, types, steps, key_dim)
        self._qkv_equation = f'{extension}bcd,{extension}bdef->{extension}ebcf'
        # (batches, heads, types, steps, key_dim) * (batches, heads, types, steps, key_dim) ->
        # -> (batches, heads, types, steps, types, steps)
        self._scores_equation = f'{extension}bcde,{extension}bfge->{extension}bcdfg'
        # (batches, heads, types, steps, key_dim) * (batches, heads, types, steps, types, steps) ->
        # -> (batches, heads, types, steps, key_dim)
        self._apply_scores_equation = f'{extension}bcde,{extension}bcdfg->{extension}bfge'
        # (batches, heads, types, steps, key_dim) * (heads, key_dim, types, out_types) ->
        # -> (batches, out_types, steps, key_dim)
        self._convert_types_dense_equation = f'{extension}abcd,{extension}adbe->{extension}aecd'
        # (batches, heads, out_types, steps, key_dim) * (heads, key_dim, steps, out_steps) ->
        # -> (batches, out_types, out_steps, key_dim)
        self._convert_steps_dense_equation = f'{extension}abcd,{extension}adce->{extension}abed'
        # (heads, out_types, out_steps, key_dim) * (heads, key_dim, out_dim) ->
        # -> (out_types, out_steps, out_dim)
        self._output_dense_equation = f'{extension}abcd,{extension}ade->{extension}bce'

    def _masked_softmax(self, attention_scores, attention_mask=None):
        # Normalize the attention scores to probabilities.
        # `attention_scores` = [B, N, T, S]
        if attention_mask is not None:
            # The expand dim happens starting from the `num_heads` dimension,
            # (<batch_dims>, num_heads, <query_attention_dims,
            # key_attention_dims>)
            mask_expansion_axis = 1
            for _ in range(
                    len(attention_scores.shape) - len(attention_mask.shape)
            ):
                attention_mask = tf.expand_dims(
                    attention_mask, axis=mask_expansion_axis
                )

        return activation.Softmax(axis=(-2, -1,))(attention_scores, attention_mask)

    def call(
            self,
            weights,
            inputs,
            attention_mask=None,
            *args,
            **kwargs
    ):
        if not self._built_from_signature:
            self.build_from_signature(inputs.shape)

        query_dense, key_dense, value_dense, convert_types_dense, convert_steps_dense, output_dense = weights

        query = tf.einsum(self._qkv_equation, inputs, query_dense)
        key = tf.einsum(self._qkv_equation, inputs, key_dense)
        value = tf.einsum(self._qkv_equation, inputs, value_dense)

        query = tf.multiply(query, 1.0 / math.sqrt(float(self._key_dim)))

        scores = tf.einsum(self._scores_equation, query, key)
        scores = self._masked_softmax(scores, attention_mask)

        out = tf.einsum(self._apply_scores_equation, value, scores)

        if self._convert_types_dense_shape.num_elements() > 0:
            out = tf.einsum(self._convert_types_dense_equation, out, convert_types_dense)

        if self._convert_steps_dense_shape.num_elements() > 0:
            out = tf.einsum(self._convert_steps_dense_equation, out, convert_steps_dense)

        out = tf.einsum(self._output_dense_equation, out, output_dense)

        return out

    @property
    def params_count(self) -> int:
        return self._params_count

    @property
    def pseudo_shapes(self) -> List[tf.TensorShape] | tf.TensorShape | None:
        return [
            self._qkv_shape,
            self._qkv_shape,
            self._qkv_shape,
            self._convert_types_dense_shape,
            self._convert_steps_dense_shape,
            self._output_dense_shape,
        ]
