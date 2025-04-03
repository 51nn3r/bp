from dataclasses import dataclass

import torch

from gm.layers.pseudo_layers.argument_parsing_strategy.argument_parsing_strategy import ArgumentParsingStrategy


@dataclass
class WeightsStorageConfig:
    argument_parsing_strategy: ArgumentParsingStrategy = ArgumentParsingStrategy({})
    device: torch._C.device | None = None
    dtype: torch.dtype = torch.float16
