from __future__ import annotations
import torch
from torch import nn

from transformers.pytorch_utils import Conv1D

from gm.layers.weights_storage.weights_storage import WeightsStorage
from gm.layers.pseudo_layers.pseudo_layer import PseudoLayer


class PseudoConv1D(PseudoLayer):
    def __init__(self, weights_storage: WeightsStorage, nf: int, nx: int):
        super().__init__(weights_storage=weights_storage)
        self.nf = nf
        self.nx = nx
        self._pseudo_shapes = [torch.Size([nx, nf]), torch.Size([nf])]
        self.register_layer()

    def forward(self, weights, x, **kwargs):
        weight, bias = weights
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(bias, x.view(-1, x.size(-1)), weight)
        x = x.view(size_out)
        return x

    @staticmethod
    def from_module(weights_storage: WeightsStorage, module: Conv1D) -> PseudoConv1D:
        nf = module.nf
        nx = module.nx
        pseudo_layer = PseudoConv1D(weights_storage, nf, nx)
        weights_storage.set_parameters(pseudo_layer.storage_index, list(module.parameters()))
        return pseudo_layer

    def __repr__(self) -> str:
        return f"PseudoConv1D(nf={self.nf}, nx={self.nx})"
