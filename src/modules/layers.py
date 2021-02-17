from typing import Tuple

import torch.nn as nn


class Reshape(nn.Module):
    def __init__(self, shape: Tuple[int]):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view((-1, *self.shape))
