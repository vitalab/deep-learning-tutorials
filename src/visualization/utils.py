from typing import Callable, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from torch import Tensor
from torchvision.datasets import VisionDataset


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
    target: Union[Tensor, Sequence[Tensor]] = None,
) -> None:
    num_samples = len(data)
    fig, axes = plt.subplots(figsize=(20, 4), nrows=1 + bool(target is not None), ncols=num_samples)

    def _display_sample(sample: Tensor, ax_idx: int, sample_label: str) -> None:
        ax = plt.subplot(2, num_samples, ax_idx)
        if ax.is_first_col():
            ax.set_ylabel(sample_label, size="xx-large")
        if sample.ndim == 3 and sample.shape[0] == 1:
            sample = sample.squeeze(dim=0)
        elif sample.ndim != 2:
            raise RuntimeError(
                "Can't display sample that is not 2D or channel+2D. The sample you're trying to display has the "
                f"following shape: {sample.shape}."
            )
        plt.imshow(sample.detach().cpu().numpy())
        plt.gray()
        _hide_axis(ax)

    for sample_idx in range(num_samples):
        # Display data on the top row
        _display_sample(data[sample_idx], sample_idx + 1, "data")

        if target is not None:
            # Display target on the bottom row
            _display_sample(target[sample_idx], sample_idx + 1 + num_samples, "target")

    fig.tight_layout()
    plt.show()


def display_autoencoder_results(
    data: VisionDataset, reconstruction_fn: Callable[[Tensor], Tensor], num_samples: int = 10
) -> None:
    samples_indices = np.random.randint(len(data), size=num_samples)
    inputs, reconstructions = [], []
    for sample_idx in samples_indices:
        img, target = data[sample_idx]
        img_hat = reconstruction_fn(img[None]).squeeze(0)
        inputs.append(img)
        reconstructions.append(img_hat)

    display_data_samples(data=reconstructions, target=inputs)
