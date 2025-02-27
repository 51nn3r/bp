from abc import ABC

import torch
from torch import nn

from gm.lora.base_lora import BaseLoRA


class LoRA(BaseLoRA, ABC):
    matrices: nn.ParameterList

    def __init__(
            self,
            shape: torch.Size,
            rank: int = 1,
            init_function=nn.init.xavier_uniform_,
    ):
        super().__init__(shape)
        dims = tuple(shape)
        n_dims = len(dims)
        self.rank = rank
        self.matrices = nn.ParameterList()

        first_core = nn.Parameter(torch.empty(dims[0], rank))
        init_function(first_core)
        self.matrices.append(first_core)

        for i in range(1, n_dims - 1):
            core = nn.Parameter(torch.empty(rank, dims[i], rank))
            init_function(core)
            self.matrices.append(core)

        if n_dims > 1:
            last_core = nn.Parameter(torch.empty(rank, dims[-1]))
            init_function(last_core)
            self.matrices.append(last_core)

    def compute_lora_delta(self) -> nn.Parameter:
        delta = self.matrices[0]
        for core in self.matrices[1:]:
            delta = torch.tensordot(delta, core, dims=([-1], [0]))
        return delta
