from typing import Literal

import holoviews as hv
import numpy as np
import torch
from holoviews import streams
from panel.layout import Panel
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

hv.extension("bokeh")


def explore_latent_space(
    data: VisionDataset,
    encoder: nn.Module,
    decoder: nn.Module,
    ae_type: Literal["ae", "vae"] = "vae",
    device: str = "cuda",
    batch_size: int = 256,
) -> Panel:
    # Ensure modules are on the requested device and in inference mode
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder.eval()
    decoder.eval()

    # Inspect the data's shape
    shape = data[0][0].shape

    # Encode the dataset
    encodings, targets = [], []
    for img, target in DataLoader(data, batch_size=batch_size):
        img = img.to(device)
        img = torch.flatten(img, start_dim=1)
        if ae_type == "ae":
            z = encoder(img)
        elif ae_type == "vae":
            z = encoder(img)[:, :2]
        else:
            raise ValueError(f"Unknown type of autoencoders '{ae_type}'.")
        encodings.extend(z.cpu().detach().numpy())
        targets.extend(target.cpu().detach().numpy())
    encodings = np.array(encodings)
    targets = np.array(targets)[:, None]
    encoded_points = hv.Points(np.hstack((encodings, targets)), vdims=["target"]).opts(
        color="target", cmap="Category10", colorbar=True
    )

    # Track the user's pointer in the scatter plot
    pointer = streams.PointerXY(x=0.0, y=0.0, source=encoded_points)

    # Setup callbacks to automatically decode selected points
    bounds = (-17, -22, 26, 16)  # Hardcoded axis limits so that decoded point is displayed at a decent resolution

    def decode_point(x, y) -> hv.Image:
        point_tensor = torch.tensor([x, y], device=device)
        decoded_point = decoder(point_tensor[None]).reshape(shape).squeeze(dim=0)
        return hv.Image(decoded_point.cpu().detach().numpy(), bounds=bounds)

    decoded_point = hv.DynamicMap(decode_point, streams=[pointer])

    # Common options for the main panels to display
    axis_opts = {"xaxis": None, "yaxis": None}
    return encoded_points.opts(**axis_opts, width=600, height=600, title="Latent space") + decoded_point.opts(
        **axis_opts, cmap="gray", title="Decoded sample"
    )
