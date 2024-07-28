from typing import List

import torch
from torch import nn


class ShapedLayer(nn.Module):
    @property
    def shapes(self) -> List[torch.Size]:
        return []
