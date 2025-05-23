from __future__ import annotations

import math

import torch
from numpy.ma.core import shape
from torch import nn
from torch.xpu import device as Dev

from gm.layers.weights_storage.weights_storage import WeightsStorage
from gm.lora.base_lora import BaseLoRA


class LoRA_SVD(BaseLoRA):
    _svd_rank: int
    _additional_dims: int
    _svd_matrices: nn.ParameterList
    _additional_matrices: nn.ParameterList
    _alpha: float
    _dropout1: nn.Module
    _dropout2: nn.Module

    def __init__(
            self,
            weights_storage: WeightsStorage,
            storage_index: int,
            layer_index: int,
            rank: int = 1,
            svd_rank: int = 1,
            alpha: float = 1.,
            init_function=nn.init.xavier_uniform_,
            dtype: torch.dtype = torch.float16,
            lora_dropout: float = 0.0,
            device: Dev | None = None,
            **kwargs,
    ):
        if rank < svd_rank:
            raise f"Invalid argument: rank {rank} < svd_rank {svd_rank}"

        super().__init__(
            weights_storage=weights_storage,
            storage_index=storage_index,
            layer_index=layer_index,
            rank=rank,
            init_function=init_function,
            dtype=dtype,
            device=device,
        )

        self._svd_rank = svd_rank
        self._additional_dims = rank - svd_rank
        self._alpha = alpha
        self._dropout1 = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        self._dropout2 = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

        base_weight = self._weights_storage_ref().get_storage_parameter(
            storage_idx=self._storage_index,
            layer_idx=self._layer_index,
        )
        dims = tuple(base_weight.shape)
        n_dims = len(dims)
        self._svd_matrices = nn.ParameterList()
        self._additional_matrices = nn.ParameterList()

        if n_dims == 1:
            first_core = nn.Parameter(torch.empty(dims[0], rank))
            init_function(first_core)
            self._svd_matrices.append(first_core)

            last_core = nn.Parameter(torch.zeros(rank))
            self._svd_matrices.append(last_core)
        elif n_dims == 2:
            u, s, vh = torch.linalg.svd(base_weight.float(), full_matrices=False)
            u_r = u[:, :svd_rank]
            s_r = s[:svd_rank]
            v_r = vh[:svd_rank, :]
            sqrt_sig = torch.diag(torch.sqrt(s_r))

            core_u = nn.Parameter((u_r @ sqrt_sig).to(dtype))
            core_v = nn.Parameter(torch.zeros_like(sqrt_sig @ v_r).to(dtype))

            additional_dims = rank - svd_rank
            extra_u = nn.Parameter(torch.empty(dims[0], additional_dims, dtype=dtype))
            nn.init.kaiming_uniform_(extra_u, a=math.sqrt(5))
            extra_v = nn.Parameter(torch.zeros(additional_dims, dims[1], dtype=dtype))

            self._svd_matrices.extend([core_u, core_v])
            self._additional_matrices.extend([extra_u, extra_v])
        else:
            first_core = nn.Parameter(torch.empty(dims[0], rank))
            init_function(first_core)
            self._svd_matrices.append(first_core)

            for i in range(1, n_dims - 1):
                core = nn.Parameter(torch.empty(rank, dims[i], rank))
                init_function(core)
                self._svd_matrices.append(core)

            if n_dims > 1:
                last_core = nn.Parameter(torch.empty(rank, dims[-1]))
                init_function(last_core)
                self._svd_matrices.append(last_core)

        for matrix in self._svd_matrices:
            matrix.to(dtype=dtype)

    def compute_lora_delta(self) -> torch.Tensor:
        delta = self._svd_matrices[0]
        for core in self._svd_matrices[1:]:
            delta = torch.tensordot(delta, core, dims=([-1], [0]))

        delta = self._dropout1(delta)

        additional_delta = self._additional_matrices[0]
        for core in self._additional_matrices[1:]:
            additional_delta = torch.tensordot(additional_delta, core, dims=([-1], [0]))

        additional_delta = self._dropout2(additional_delta)
        delta += additional_delta

        return self._alpha * delta / self._rank

    def reset_matrices(self) -> None:
        with torch.no_grad():
            for m in self._svd_matrices:
                if m.dim() == 2:
                    weights = self._weights_storage_ref().get_storage_parameter(
                        storage_idx=self._storage_index,
                        layer_idx=self._layer_index,
                    )
                    d_out, d_in = weights.shape

                    u, s, vh = torch.linalg.svd(weights.float(), full_matrices=False)
                    u_r = u[:, :self._svd_rank]
                    s_r = s[:self._svd_rank]
                    v_r = vh[:self._svd_rank, :]
                    diag_sqrt = torch.diag(torch.sqrt(s_r)).to(self._dtype)

                    core_u = (u_r @ diag_sqrt).to(self._dtype)
                    core_v = (torch.zeros_like(diag_sqrt @ v_r)).to(self._dtype)

                    extra_u = torch.empty(d_out, self._additional_dims, dtype=self._dtype, device=self._device)
                    nn.init.kaiming_uniform_(extra_u, a=math.sqrt(5))
                    extra_v = torch.zeros(self._additional_dims, d_in, dtype=self._dtype, device=self._device)

                    self._svd_matrices[0].data.copy_(core_u)
                    self._svd_matrices[1].data.copy_(core_v)
                    self._additional_matrices[0].data.copy_(extra_u)
                    self._additional_matrices[1].data.copy_(extra_v)
                elif m.dim() == 3:
                    nn.init.xavier_uniform_(m)
                else:
                    nn.init.normal_(m)

        self.enable()

    def enable_grad(self):
        if len(self._shape) == 2:
            self._svd_matrices[0].requires_grad = False
            self._svd_matrices[1].requires_grad = True
            self._additional_matrices[0].requires_grad = True
            self._additional_matrices[1].requires_grad = True
        else:
            super().enable_grad()

    def disable_grad(self):
        if len(self._shape) == 2:
            self._svd_matrices[0].requires_grad = False
            self._svd_matrices[1].requires_grad = False
            self._additional_matrices[0].requires_grad = False
            self._additional_matrices[1].requires_grad = False
        else:
            super().disable_grad()
