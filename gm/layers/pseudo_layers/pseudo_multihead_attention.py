from typing import List, Optional
import torch
from gm.layers.pseudo_layers.pseudo_layer import PseudoLayer
from gm.layers.weights_storage.weights_storage import WeightsStorage


class PseudoMultiHeadAttention(PseudoLayer):
    def __init__(
            self,
            weights_storage: WeightsStorage,
            embed_dim: int,
            num_heads: int,
            **kwargs,
    ):
        """
        Initializes standard multi-head attention.
        """
        super().__init__(weights_storage, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.head_dim = embed_dim // num_heads

        # Define weight shapes for query, key, value, and output projections.
        self._q_kernel_shape = torch.Size([embed_dim, num_heads * self.head_dim])
        self._q_bias_shape = torch.Size([num_heads * self.head_dim])
        self._k_kernel_shape = torch.Size([embed_dim, num_heads * self.head_dim])
        self._k_bias_shape = torch.Size([num_heads * self.head_dim])
        self._v_kernel_shape = torch.Size([embed_dim, num_heads * self.head_dim])
        self._v_bias_shape = torch.Size([num_heads * self.head_dim])
        self._out_kernel_shape = torch.Size([num_heads * self.head_dim, embed_dim])
        self._out_bias_shape = torch.Size([embed_dim])

        self._pseudo_shapes = [
            self._q_kernel_shape, self._q_bias_shape,
            self._k_kernel_shape, self._k_bias_shape,
            self._v_kernel_shape, self._v_bias_shape,
            self._out_kernel_shape, self._out_bias_shape,
        ]
        self.register_layer()

    def forward(
            self,
            weights: List[torch.Tensor],
            query: torch.Tensor,
            key: Optional[torch.Tensor] = None,
            value: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes standard multi-head attention.
        """
        if key is None:
            key = query
        if value is None:
            value = query

        q_kernel, q_bias, k_kernel, k_bias, v_kernel, v_bias, out_kernel, out_bias = weights

        # Linear projections
        Q = torch.matmul(query, q_kernel) + q_bias  # (batch, time, num_heads*head_dim)
        K = torch.matmul(key, k_kernel) + k_bias
        V = torch.matmul(value, v_kernel) + v_bias

        batch_size, seq_len, _ = Q.shape

        # Reshape to (batch, time, num_heads, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to (batch, num_heads, time, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn, V)  # (batch, num_heads, time, head_dim)

        # Reshape back to (batch, time, num_heads*head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.num_heads * self.head_dim)

        # Final output projection
        output = torch.matmul(attn_output, out_kernel) + out_bias
        return output
