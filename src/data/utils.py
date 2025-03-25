from typing import Optional

import torch
from torch import Tensor


def to_onehot(
    label_tensor: Tensor,
    num_classes: Optional[int] = None,
) -> Tensor:
    """Convert  a dense label tensor to one-hot format.

    References:
        Copied from `torchmetrics.utilities.data.to_onehot` (v1.6.2, 2025-03-25):
        https://lightning.ai/docs/torchmetrics/stable/references/utilities.html#to-onehot

    Args:
        label_tensor: dense label tensor, with shape [N, d1, d2, ...]
        num_classes: number of classes C

    Returns:
        A sparse label tensor with shape [N, C, d1, d2, ...]

    Example:
        >>> x = torch.tensor([1, 2, 3])
        >>> to_onehot(x)
        tensor([[0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])
    """
    if num_classes is None:
        num_classes = int(label_tensor.max().detach().item() + 1)

    tensor_onehot = torch.zeros(
        label_tensor.shape[0],
        num_classes,
        *label_tensor.shape[1:],
        dtype=label_tensor.dtype,
        device=label_tensor.device,
    )
    index = label_tensor.long().unsqueeze(1).expand_as(tensor_onehot)
    return tensor_onehot.scatter_(1, index, 1.0)
