from __future__ import annotations

from torch import nn


class BaseFlashAttention(nn.Module):
    @staticmethod
    def from_module(
            module: nn.Module,
    ) -> BaseFlashAttention:
        raise "Not implemented"
