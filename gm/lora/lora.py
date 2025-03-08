from abc import ABC

import torch
from torch import nn

from gm.lora.base_lora import BaseLoRA


class LoRA(BaseLoRA, ABC):
    _matrices: nn.ParameterList

    def __init__(
            self,
            shape: torch.Size,
            rank: int = 1,
            init_function=nn.init.xavier_uniform_,
    ):
        super().__init__(shape, rank, init_function)
        dims = tuple(shape)
        n_dims = len(dims)
        self._matrices = nn.ParameterList()

        if n_dims == 1:
            first_core = nn.Parameter(torch.empty(dims[0], rank))
            init_function(first_core)
            self._matrices.append(first_core)

            last_core = nn.Parameter(torch.zeros(rank))
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

    def compute_lora_delta(self) -> nn.Parameter:
        delta = self._matrices[0]
        for core in self._matrices[1:]:
            delta = torch.tensordot(delta, core, dims=([-1], [0]))

        return delta

    def reset_matrices(self):
        for matrix in self._matrices:
            if len(matrix.shape) == 1:
                nn.init.normal_(matrix)
            else:
                self._init_function(matrix)
