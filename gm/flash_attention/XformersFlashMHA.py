from __future__ import annotations

import torch.nn as nn
from xformers.ops import memory_efficient_attention

from gm.flash_attention.base_flash_attention import BaseFlashAttention


class XformersFlashMHA(BaseFlashAttention):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        """
        Multi-Head Attention с использованием xFormers memory_efficient_attention.

        :param embed_dim: размерность эмбеддингов (например, 512)
        :param num_heads: количество голов (например, 8)
        :param dropout: вероятность dropout для внимания
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim должен делиться на num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        # Проекции для Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        """
        :param x: входной тензор размера [batch, seq_len, embed_dim]
        :return: выходной тензор [batch, seq_len, embed_dim]
        """
        batch, seq_len, embed_dim = x.shape

        # Получаем Q, K, V
        Q = self.q_proj(x)  # [batch, seq_len, embed_dim]
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Изменяем форму в [batch, num_heads, seq_len, head_dim]
        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Объединяем batch и num_heads, получая [batch*num_heads, seq_len, head_dim]
        Q = Q.reshape(batch * self.num_heads, seq_len, self.head_dim)
        K = K.reshape(batch * self.num_heads, seq_len, self.head_dim)
        V = V.reshape(batch * self.num_heads, seq_len, self.head_dim)

        # Определяем scale = 1/sqrt(head_dim)
        scale = 1.0 / (self.head_dim ** 0.5)

        # Вызываем xFormers memory_efficient_attention
        # p задаёт вероятность dropout
        attn_out = memory_efficient_attention(
            query=Q,
            key=K,
            value=V,
            attn_bias=None,
            p=self.dropout,
            scale=scale,
            output_dtype=Q.dtype
        )
        # attn_out имеет форму [batch*num_heads, seq_len, head_dim]

        # Возвращаем форму в [batch, num_heads, seq_len, head_dim]
        attn_out = attn_out.reshape(batch, self.num_heads, seq_len, self.head_dim)
        # Транспонируем и объединяем, получая [batch, seq_len, embed_dim]
        attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, embed_dim)

        # Финальная линейная проекция
        out = self.out_proj(attn_out)
        return out

    @staticmethod
    def from_module(
            module: nn.Module,
    ) -> BaseFlashAttention:
        if not isinstance(module, nn.MultiheadAttention):
            raise f"Not implemented for {module.__class__}"

        flash_mha = XformersFlashMHA(
            embed_dim=module.embed_dim,
            num_heads=module.num_heads,
            dropout=module.dropout,
        )
        flash_mha.q_proj.weight.data.copy_(module.q_proj_weight.data)
        flash_mha.k_proj.weight.data.copy_(module.k_proj_weight.data)
        flash_mha.v_proj.weight.data.copy_(module.v_proj_weight.data)
        flash_mha.out_proj.weight.data.copy_(module.out_proj.weight.data)

        return flash_mha
