from typing import List, Optional
import torch
from gm.layers.pseudo_layers.pseudo_layer import PseudoLayer
from gm.layers.weights_storage.weights_storage import WeightsStorage


class PseudoGroupedQueryAttention(PseudoLayer):
    def __init__(
            self,
            weights_storage: WeightsStorage,
            embed_dim: int,
            num_heads: int,
            num_query_groups: int,
            **kwargs,
    ):
        """
        Initializes grouped query attention layer.
        """
        super().__init__(weights_storage, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_query_groups = num_query_groups
        if num_heads % num_query_groups != 0:
            raise ValueError("num_heads must be divisible by num_query_groups")
        self.group_size = num_heads // num_query_groups
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.head_dim = embed_dim // num_heads

        # Query projection uses grouped weights: (embed_dim, num_query_groups * head_dim)
        self._q_kernel_shape = torch.Size([embed_dim, num_query_groups * self.head_dim])
        self._q_bias_shape = torch.Size([num_query_groups * self.head_dim])
        # Standard key and value projections.
        self._k_kernel_shape = torch.Size([embed_dim, num_heads * self.head_dim])
        self._k_bias_shape = torch.Size([num_heads * self.head_dim])
        self._v_kernel_shape = torch.Size([embed_dim, num_heads * self.head_dim])
        self._v_bias_shape = torch.Size([num_heads * self.head_dim])
        # Output projection: (num_heads * head_dim, embed_dim)
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
        Computes grouped query attention.
        """
        if key is None:
            key = query
        if value is None:
            value = query

        q_kernel, q_bias, k_kernel, k_bias, v_kernel, v_bias, out_kernel, out_bias = weights

        # Compute grouped query projection: (batch, time, num_query_groups * head_dim)
        Q_grouped = torch.matmul(query, q_kernel) + q_bias
        batch_size, seq_len, _ = Q_grouped.shape
        Q_grouped = Q_grouped.view(batch_size, seq_len, self.num_query_groups, self.head_dim)
        # Expand grouped queries to full head dimension by repeating each group.
        Q = Q_grouped.repeat(1, 1, self.group_size, 1)  # (batch, time, num_heads, head_dim)

        # Standard key and value projections.
        K = torch.matmul(key, k_kernel) + k_bias
        V = torch.matmul(value, v_kernel) + v_bias

        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to (batch, num_heads, time, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Scaled dot-product attention.
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn, V)

        # Reshape back to (batch, time, num_heads*head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.num_heads * self.head_dim)

        # Final output projection.
        output = torch.matmul(attn_output, out_kernel) + out_bias
        return output
