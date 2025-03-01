from typing import List, Union, Tuple
import torch
from gm.layers.pseudo_layers.pseudo_layer import PseudoLayer
from gm.layers.weights_storage.weights_storage import WeightsStorage


class PseudoLayerNorm(PseudoLayer):
    def __init__(
            self,
            weights_storage: WeightsStorage,
            normalized_shape: Union[int, Tuple[int, ...]],
            eps: float = 1e-5,
            **kwargs,
    ):
        """
        Initializes layer normalization.
        """
        super().__init__(weights_storage, **kwargs)
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps

        # Learnable parameters: gamma (scale) and beta (shift)
        self._gamma_shape = torch.Size(self.normalized_shape)
        self._beta_shape = torch.Size(self.normalized_shape)
        self._pseudo_shapes = [self._gamma_shape, self._beta_shape]
        self.register_layer()

    def forward(
            self,
            weights: List[torch.Tensor],
            inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Applies layer normalization.
        """
        gamma, beta = weights
        dims = tuple(range(-len(self.normalized_shape), 0))
        mean = inputs.mean(dim=dims, keepdim=True)
        var = inputs.var(dim=dims, unbiased=False, keepdim=True)
        normalized = (inputs - mean) / torch.sqrt(var + self.eps)
        return gamma * normalized + beta
