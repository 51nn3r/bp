from abc import ABC

import torch
from torch import nn

from gm.lora.base_lora import BaseLoRA


class LoRA(BaseLoRA, ABC):
    _matrices: nn.ParameterList
    _alpha: float
    _dropout: nn.Module

    def __init__(
            self,
            shape: torch.Size,
            rank: int = 1,
            alpha: float = 1.,
            init_function=nn.init.xavier_uniform_,
            dtype: torch.dtype = torch.float16,
            lora_dropout: float = 0.0,
    ):
        super().__init__(
            shape=shape,
            rank=rank,
            init_function=init_function,
            dtype=dtype,
        )
        self._alpha = alpha

        self._dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

        dims = tuple(shape)
        n_dims = len(dims)
        self._matrices = nn.ParameterList()

        if n_dims == 1:
            first_core = nn.Parameter(torch.empty(dims[0], rank))
            init_function(first_core)
            self._matrices.append(first_core)

            last_core = nn.Parameter(torch.zeros(rank))
            self._matrices.append(last_core)
        elif n_dims == 2:
            first_core = nn.Parameter(torch.empty(dims[0], rank))
            last_core = nn.Parameter(torch.zeros(rank, dims[0]))
            nn.init.normal_(first_core, std=1 / rank)
            nn.init.zeros_(last_core)
            self._matrices.append(first_core)
            self._matrices.append(last_core)
        else:
            first_core = nn.Parameter(torch.empty(dims[0], rank))
            init_function(first_core)
            self._matrices.append(first_core)

            for i in range(1, n_dims - 1):
                core = nn.Parameter(torch.empty(rank, dims[i], rank))
                init_function(core)
                self._matrices.append(core)

            if n_dims > 1:
                last_core = nn.Parameter(torch.empty(rank, dims[-1]))
                init_function(last_core)
                self._matrices.append(last_core)

        # cast dtype
        for matrix in self._matrices:
            matrix.to(dtype=dtype)

    def compute_lora_delta(self) -> nn.Parameter:
        delta = self._matrices[0]
        for core in self._matrices[1:]:
            delta = torch.tensordot(delta, core, dims=([-1], [0]))

        delta = self._dropout(delta)
        return self._alpha * delta / self._rank

    def reset_matrices(self):
        if len(self._matrices) == 2:
            nn.init.normal_(self._matrices[0], std=1 / self._rank)
            nn.init.zeros_(self._matrices[1])
        else:
            for matrix in self._matrices:
                if len(matrix.shape) == 1:
                    nn.init.normal_(matrix)
                else:
                    self._init_function(matrix)

        self.enable()
