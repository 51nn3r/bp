from dataclasses import dataclass
import torch

from gm.layers.pseudo_layers.configs.pseudo_layer_config import PseudoLayerConfig


@dataclass
class GrossMachineConfig(PseudoLayerConfig):
    selector: torch.Tensor
