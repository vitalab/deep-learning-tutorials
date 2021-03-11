from typing import Callable

import holoviews as hv
import numpy as np
import torch
import umap
from holoviews import opts, streams
from panel.layout import Panel
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

hv.extension("bokeh")


def explore_latent_space(
    data: VisionDataset,
    encode_fn: Callable[[Tensor], Tensor],
    decode_fn: Callable[[Tensor], Tensor],
    data_to_encode: str = "img",
    encodings_label: str = None,
    device: str = "cuda",
    batch_size: int = 256,
) -> Panel:
    """Generates panel to interactively visualize an autoencoder's latent space distribution and individual samples.

    Args:
        data: Dataset that returns pairs of images and targets, where the target can either be a classification label or
            a segmentation map.
        encode_fn: Function that uses an encoder neural net to predict a latent vector associated with the high
            dimensional input data.
        decode_fn: Function that uses a decoder neural net to decode a latent vector back to a high dimensional data
            sample.
        data_to_encode: One of "img" or "target", indicating which data provided by the dataset to feed to the encoding
            function.
        encodings_label: Either "target" or `None`, indicating whether to use the data items' targets (in classification
            tasks) as labels by which to color the latent vectors in the plot.
        device: Device on which to perform the neural nets' computations.
        batch_size: Size of the batch to use when initially encoding the dataset's items in the latent space.

    Returns:
        Interactive panel to interactively visualize the autoencoder's latent space distribution and individual samples.
    """
    data_to_encode_values = ["img", "target"]
    if data_to_encode not in data_to_encode_values:
        raise ValueError(
            f"Invalid value for `data_to_encode` flag. You passed: '{data_to_encode}', "
            f"but it should be one of {data_to_encode_values}."
        )

    if encodings_label == "target" and data_to_encode == "target":
        raise ValueError(
            "You requested conflicting options: encode the dataset's targets / label encodings w.r.t targets. "
            "Either switch to encode the images instead (`data_to_encode='img'`), "
            "or give up labelling the encodings (`encodings_label=None`)."
        )

    # Encode the dataset
    print("Encoding the dataset items in the latent space...")
    encodings, targets = [], []
    for img, target in DataLoader(data, batch_size=batch_size):
        img = img.to(device)
        target = target.to(device)

        data = img if data_to_encode == "img" else target
        z = encode_fn(data)

        encodings.extend(z.cpu().detach().numpy())

        if encodings_label == "target":
            targets.extend(target.cpu().detach().numpy())

    encodings = np.array(encodings)
    targets = np.array(targets)[:, None]

    # If the latent space is not already 2D, use the UMAP dimensionality reduction algorithm
    # to learn a projection between the latent space and a 2D space ready to be displayed
    latent_space_ndim = encodings.shape[-1]
    high_dim_latent_space = latent_space_ndim > 2
    if high_dim_latent_space:
        print("Learning UMAP embedding for latent space vectors...")
        reducer = umap.UMAP()
        encodings = reducer.fit_transform(encodings)

    if encodings_label == "target":
        encoded_points = hv.Points(np.hstack((encodings, targets)), vdims=["target"]).opts(
            color="target", cmap="Category10", colorbar=True
        )
    else:
        encoded_points = hv.Points(encodings)

    # Track the user's pointer in the scatter plot
    pointer = streams.PointerXY(x=0.0, y=0.0, source=encoded_points)

    # Setup callbacks to automatically decode selected points
    def decode_point(x, y) -> hv.Image:
        latent_space_point = np.array([x, y])[None]
        if high_dim_latent_space:
            # Project the 2D sample back into the higher-dimensionality latent space
            # using UMAP's learned inverse transform
            latent_space_point = reducer.inverse_transform(latent_space_point)
        point_tensor = torch.tensor(latent_space_point, dtype=torch.float, device=device)
        decoded_img = decode_fn(point_tensor).squeeze(dim=0)
        return hv.Image(decoded_img.cpu().detach().numpy())

    decoded_point = hv.DynamicMap(decode_point, streams=[pointer]).opts(opts.Image(axiswise=True))

    # Common options for the main panels to display
    encodings_title = (
        "Latent space" if not high_dim_latent_space else f"2D UMAP embedding of the {latent_space_ndim}D latent space"
    )
    return encoded_points.opts(width=600, height=600, title=encodings_title) + decoded_point.opts(
        xaxis=None, yaxis=None, cmap="gray", title="Decoded sample"
    )
