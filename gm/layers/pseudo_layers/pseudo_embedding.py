from typing import List
import torch
from gm.layers.pseudo_layers.pseudo_layer import PseudoLayer
from gm.layers.weights_storage.weights_storage import WeightsStorage


class PseudoEmbedding(PseudoLayer):
    def __init__(
            self,
            weights_storage: WeightsStorage,
            num_embeddings: int,
            embedding_dim: int,
            **kwargs,
    ):
        """
        Initializes embedding layer.
        """
        super().__init__(weights_storage, **kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Embedding matrix shape: (num_embeddings, embedding_dim)
        self._embed_shape = torch.Size([num_embeddings, embedding_dim])
        self._pseudo_shapes = [self._embed_shape]
        self.register_layer()

    def forward(
            self,
            weights: List[torch.Tensor],
            indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Retrieves embeddings for given indices.
        """
        embed_matrix = weights[0]
        return embed_matrix[indices]
