import random
from typing import Callable, Sequence, Union

import matplotlib.pyplot as plt
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


def display_data_samples(**data_samples: Union[Tensor, Sequence[Tensor]]) -> None:
    num_samples_by_src = {src_name: len(samples) for src_name, samples in data_samples.items()}
    num_samples = random.choice(list(num_samples_by_src.values()))
    if not all(src_num_samples == num_samples for src_num_samples in num_samples_by_src.values()):
        raise ValueError(
            f"`display_data_samples` requires all data sources to provide the same number of (corresponding) samples. "
            f"You provided the following data sources (with the number of samples for each one): {num_samples_by_src}."
        )
    fig, axes = plt.subplots(figsize=(20, 4), nrows=len(data_samples), ncols=num_samples)

    def _display_sample(sample: Tensor, ax_idx: int, sample_label: str) -> None:
        ax = plt.subplot(len(data_samples), num_samples, ax_idx)
        if ax.get_subplotspec().is_first_col():
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

    # For each column
    for sample_idx in range(num_samples):
        # Display the sample for the current column from each data source
        for src_idx, (src_name, samples) in enumerate(data_samples.items()):
            ax_idx = (
                (src_idx * num_samples)  # each row corresponds to a different data source
                + (sample_idx + 1)  # each column corresponds to a different sample from the data source
            )
            _display_sample(samples[sample_idx], ax_idx, src_name)

    fig.tight_layout()
    plt.show()


def display_autoencoder_results(
    data: VisionDataset,
    reconstruction_fn: Callable[[Tensor], Tensor],
    num_samples: int = 10,
    reconstruct_target: bool = False,
) -> None:
    samples_indices = random.sample(range(len(data)), num_samples)
    inputs, reconstructions = [], []
    for sample_idx in samples_indices:
        img, target = data[sample_idx]
        x = img if not reconstruct_target else target
        x_hat = reconstruction_fn(x[None]).squeeze(0)
        inputs.append(x)
        reconstructions.append(x_hat)

    display_data_samples(data=inputs, reconstruction=reconstructions)
