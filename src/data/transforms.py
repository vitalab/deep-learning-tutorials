import numpy as np
import torch
import torchvision.transforms.functional as F


class NormalizeSample(torch.nn.Module):
    """Normalize a tensor image w.r.t. to its mean and standard deviation.

    Args:
        inplace: Whether to make this operation in-place.
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalizes input tensor.

        Args:
            tensor: Tensor of size (N, {?}) to be normalized.

        Returns:
            Normalized tensor.
        """
        return F.normalize(tensor, [float(tensor.mean())], [float(tensor.std())], self.inplace)


class SegmentationToTensor(torch.nn.Module):
    """Converts a segmentation map to a tensor.

    Args:
        flip_channels: If ``True``, assumes that the input is in `channels_last` mode and will automatically convert it
            to `channels_first` mode. If ``False``, leaves the ordering of dimensions untouched.
    """

    def __init__(self, flip_channels: bool = False):
        super().__init__()
        self.flip_channels = flip_channels

    def __call__(self, data: np.ndarray) -> torch.Tensor:
        """Converts the segmentation map to a tensor.

        Args:
            data: ([N], H, W, [C]), Segmentation map to convert to a tensor.

        Returns:
            ([N], [C], H, W), Segmentation map converted to a tensor.
        """
        if self.flip_channels:  # If there is a specific channel dimension
            if len(data.shape) == 3:  # If it is a single segmentation
                dim_to_transpose = (2, 0, 1)
            elif len(data.shape) == 4:  # If there is a batch dimension to keep first
                dim_to_transpose = (0, 3, 1, 2)
            else:
                raise ValueError(
                    "Segmentation to convert to tensor is expected to be a single segmentation (2D), "
                    "or a batch of segmentations (3D). The segmentation requested to be converted is "
                    f"{len(data.shape)}D."
                )
            # Change format from `channel_last`, i.e. ([N], H, W, C), to `channel_first`, i.e. ([N], C, H, W)
            data = data.transpose(dim_to_transpose)
        return torch.from_numpy(data.astype("int64"))
