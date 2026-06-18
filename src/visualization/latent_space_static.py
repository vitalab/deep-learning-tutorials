from math import ceil, sqrt
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset


def plot_latent_space_with_decodings(
    data: VisionDataset,
    encode_fn: Callable[[Tensor], Tensor],
    decode_fn: Optional[Callable[[Tensor], Tensor]] = None,
    data_to_encode: str = "img",
    encodings_label: Optional[str] = "target",
    device: str = "cuda",
    batch_size: int = 256,
    n_decode: int = 9,
    sample_region_pct: float = 5.0,
    cmap: str = "tab10",
    sample_cmap: str = "hsv",
    image_cmap: str = "gray",
    point_size: float = 8.0,
    alpha: float = 0.5,
    seed: Optional[int] = None,
    figsize: Tuple[float, float] = (15.0, 7.5),
):
    """Static latent-space scatter next to a grid of decoded random samples.

    The latent space is shown on the left (projected to 2D with UMAP if it has more than 2
    dimensions). If ``decode_fn`` is given, ``n_decode`` random points are drawn from within the
    spread of the real encodings, decoded, and shown in a square grid on the right. Each sampled
    point is overlaid on the scatter and numbered/colored to match its decoded image, so you can
    read off "this region of the map decodes to this".

    Sampling happens in the *displayed* 2D space and is mapped back to the latent space via UMAP's
    inverse transform (or used directly when the latent space is already 2D). This keeps every
    marker exactly aligned with the image it produced. If you instead want to sample from the VAE
    prior, draw ``torch.randn(n_decode, latent_dim)`` yourself and call ``decode_fn`` on it.

    Args:
        data: Dataset returning (image, target) pairs.
        encode_fn: Maps input data to a latent vector.
        decode_fn: Maps a batch of latent vectors back to images. If ``None``, only the scatter is
            drawn. Output is squeezed to 2D per sample; a leftover channel dim is argmax-ed (handy
            for one-hot / logit segmentation maps).
        data_to_encode: "img" or "target" - which item to feed to ``encode_fn``.
        encodings_label: "target" to color the scatter by class, or ``None`` for a single color.
        device: Device for the neural net computations.
        batch_size: Batch size used while encoding the dataset.
        n_decode: Number of random points to decode (laid out in the squarest grid that fits).
        sample_region_pct: Percentile inset used to define the sampling box, avoiding extreme
            outliers (e.g. 5.0 samples within the 5th-95th percentile box of each axis).
        cmap: Colormap for the class-colored scatter.
        sample_cmap: Colormap used to color the decoded samples and their markers.
        image_cmap: Colormap for the decoded images (use a discrete map for segmentation maps).
        point_size: Scatter marker size for the encoded dataset.
        alpha: Opacity of the encoded dataset points (kept low so samples stand out).
        seed: Optional RNG seed for reproducible sampling.
        figsize: Figure size.

    Returns:
        (fig, ax_latent, image_axes): the figure, the scatter Axes, and the list of image Axes
        (empty if ``decode_fn`` is None).
    """
    if data_to_encode not in ("img", "target"):
        raise ValueError(f"`data_to_encode` must be 'img' or 'target', got '{data_to_encode}'.")
    if encodings_label == "target" and data_to_encode == "target":
        raise ValueError(
            "Conflicting options: cannot encode targets and also color by targets. "
            "Use data_to_encode='img', or encodings_label=None."
        )

    rng = np.random.default_rng(seed)

    # --- Encode the dataset ---
    print("Encoding the dataset items in the latent space...")
    encodings, targets = [], []
    for img, target in DataLoader(data, batch_size=batch_size):
        img = img.to(device)
        target = target.to(device)

        batch = img if data_to_encode == "img" else target
        z = encode_fn(batch)

        encodings.extend(z.cpu().detach().numpy())
        if encodings_label == "target":
            targets.extend(target.cpu().detach().numpy())

    encodings = np.array(encodings)
    targets = np.array(targets) if encodings_label == "target" else None

    # --- Reduce to 2D if needed (keep the reducer for the inverse transform) ---
    latent_ndim = encodings.shape[-1]
    reducer = None
    if latent_ndim > 2:
        print(f"Learning UMAP embedding for {latent_ndim}D latent vectors...")
        import umap  # imported lazily so 2D latent spaces don't require it

        reducer = umap.UMAP()
        embedding = reducer.fit_transform(encodings)
        title = f"2D UMAP embedding of the {latent_ndim}D latent space"
    elif latent_ndim < 2:
        raise ValueError(f"Latent space must be at least 2D, got {latent_ndim}D.")
    else:
        embedding = encodings
        title = "Latent space"

    # --- Sample points to decode (in the displayed 2D space) ---
    sampled_2d = None
    decoded_imgs = None
    if decode_fn is not None and n_decode > 0:
        lo, hi = np.percentile(embedding, [sample_region_pct, 100 - sample_region_pct], axis=0)
        sampled_2d = rng.uniform(lo, hi, size=(n_decode, 2))

        # Map the 2D samples back into the latent space, then decode them in one batch.
        if reducer is not None:
            latent_pts = reducer.inverse_transform(sampled_2d)
        else:
            latent_pts = sampled_2d
        latent_tensor = torch.as_tensor(latent_pts, dtype=torch.float32, device=device)
        with torch.no_grad():
            decoded = decode_fn(latent_tensor)
        decoded_imgs = decoded.detach().cpu().numpy()

    # --- Figure layout: scatter on the left, square grid of decodings on the right ---
    image_axes = []
    if decode_fn is not None and n_decode > 0:
        n_cols = ceil(sqrt(n_decode))
        n_rows = ceil(n_decode / n_cols)
        sample_colors = plt.get_cmap(sample_cmap, n_decode)

        fig = plt.figure(figsize=figsize)
        outer = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.12)
        ax_latent = fig.add_subplot(outer[0, 0])
        inner = outer[0, 1].subgridspec(n_rows, n_cols, wspace=0.08, hspace=0.18)
    else:
        fig, ax_latent = plt.subplots(figsize=(figsize[1], figsize[1]))

    # --- Scatter of the encoded dataset ---
    if targets is not None:
        classes = np.unique(targets)
        class_colors = plt.get_cmap(cmap, len(classes))
        for i, c in enumerate(classes):
            mask = targets == c
            ax_latent.scatter(
                embedding[mask, 0], embedding[mask, 1],
                s=point_size, alpha=alpha, color=class_colors(i), label=str(c),
            )
        ax_latent.legend(title="class", markerscale=2, loc="best", framealpha=0.9)
    else:
        ax_latent.scatter(embedding[:, 0], embedding[:, 1], s=point_size, alpha=alpha, color="0.4")

    ax_latent.set_title(title)
    ax_latent.set_xlabel("dim 1")
    ax_latent.set_ylabel("dim 2")
    ax_latent.set_aspect("equal", adjustable="datalim")

    # --- Overlay the sampled points and fill the decoded grid ---
    if sampled_2d is not None:
        for i in range(n_decode):
            color = sample_colors(i)

            ax_latent.scatter(
                sampled_2d[i, 0], sampled_2d[i, 1],
                s=220, marker="*", color=color, edgecolors="black", linewidths=1.2, zorder=5,
            )
            ax_latent.annotate(
                str(i + 1), (sampled_2d[i, 0], sampled_2d[i, 1]),
                textcoords="offset points", xytext=(6, 6), fontsize=9, fontweight="bold",
                color="black", zorder=6,
            )

            ax_img = fig.add_subplot(inner[i // n_cols, i % n_cols])
            img = np.squeeze(decoded_imgs[i])
            if img.ndim == 3:  # leftover channel dim (e.g. one-hot / logits) -> categorical map
                img = img.argmax(axis=0)
            ax_img.imshow(img, cmap=image_cmap)
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            ax_img.set_title(f"#{i + 1}", color=color, fontsize=10, fontweight="bold")
            for spine in ax_img.spines.values():
                spine.set_color(color)
                spine.set_linewidth(3)
            image_axes.append(ax_img)

    return fig, ax_latent, image_axes