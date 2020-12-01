from typing import Callable, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from torch import Tensor
from torchvision.datasets import VisionDataset
from torchvision.transforms import transforms


def _hide_axis(ax: Axes, hide_ticks: bool = True, hide_ticklabels: bool = True) -> None:
    xaxis = ax.get_xaxis()
    yaxis = ax.get_yaxis()
    if hide_ticks:
        xaxis.set_ticks([])
        yaxis.set_ticks([])
    if hide_ticklabels:
        xaxis.set_ticklabels([])
        yaxis.set_ticklabels([])


def display_data_samples(
    data: Union[Tensor, Sequence[Tensor]],
    reconstructions: Union[Tensor, Sequence[Tensor]] = None,
) -> None:
    to_pil = transforms.ToPILImage()
    num_samples = len(data)
    fig, axes = plt.subplots(figsize=(20, 4), nrows=1 + bool(reconstructions is not None), ncols=num_samples)

    for sample_idx in range(num_samples):
        # Display original
        ax = plt.subplot(2, num_samples, sample_idx + 1)
        if ax.is_first_col():
            ax.set_ylabel("image", size="xx-large")
        plt.imshow(to_pil(data[sample_idx]))
        plt.gray()
        _hide_axis(ax)

        if reconstructions is not None:
            # Display reconstruction
            ax = plt.subplot(2, num_samples, sample_idx + 1 + num_samples)
            if ax.is_first_col():
                ax.set_ylabel("reconstruction", size="xx-large")
            plt.imshow(to_pil(reconstructions[sample_idx]))
            plt.gray()
            _hide_axis(ax)

    fig.tight_layout()
    plt.show()


def display_autoencoder_results(
    data: VisionDataset, reconstruction_fn: Callable[[Tensor], Tensor], num_samples: int = 10
) -> None:
    samples_indices = np.random.randint(len(data), size=num_samples)
    imgs, reconstructions = [], []
    for sample_idx in samples_indices:
        img, target = data[sample_idx]
        img_hat = reconstruction_fn(img[None]).squeeze(0)
        imgs.append(img)
        reconstructions.append(img_hat)

    display_data_samples(data=imgs, reconstructions=reconstructions)
