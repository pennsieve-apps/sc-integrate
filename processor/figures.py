"""
PNG figures (150 dpi): the UMAP of all observations coloured by a categorical label (sample,
condition, cluster). The colour system of the spatial suite: a fixed, ordered categorical list
(identity follows the label, never its rank; a legend is always drawn), muted inks for chrome.
"""
from __future__ import annotations

from pathlib import Path

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
          "#184f95", "#9a3d12", "#0f6e4c", "#8a5c00", "#a63d6a", "#5c8a3a", "#3b2f8f", "#8f2727",
          "#7fb3ee", "#f3a07a", "#7fd3b2", "#f5cb5c", "#f3b5cc", "#7fc07f", "#aa9ee8", "#f09a9a"]
DPI = 150


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold",
        "legend.frameon": False, "legend.fontsize": 8,
    })
    return plt


def colour_map(labels: list[str]) -> dict[str, str]:
    """A fixed colour per label in the given order."""
    return {lab: SERIES[i % len(SERIES)] for i, lab in enumerate(labels)}


def _legend(ax, labels, colours, counts=None, ncol=1, title=None):
    from matplotlib.lines import Line2D

    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=6, markerfacecolor=colours[lab],
                      markeredgecolor=SURFACE, markeredgewidth=0.5,
                      label=f"{lab} ({counts[lab]:,})" if counts else lab) for lab in labels]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=ncol, title=title,
              handletextpad=0.3, borderaxespad=0.0, labelspacing=0.3, markerscale=1.0)


def _point_size(n: int) -> float:
    return 6.0 if n < 5000 else 2.5 if n < 50000 else 1.0


def scatter_categorical(coords, labels, order, title: str, path: Path, seed: int = 0) -> Path:
    """Points coloured by a categorical label, drawn in a seeded random order so that no label hides
    another (overlap is what batch mixing looks like)."""
    import numpy as np

    plt = _plt()
    colours = colour_map(order)
    labels = np.asarray(labels, dtype=object)
    sizes = {lab: int((labels == lab).sum()) for lab in order}
    ncol = 1 if len(order) <= 24 else 2 if len(order) <= 60 else 3
    fig, ax = plt.subplots(figsize=(7.5 + 1.8 * ncol, 6.4))
    draw = np.random.default_rng(seed).permutation(len(labels))
    ax.scatter(coords[draw, 0], coords[draw, 1], s=_point_size(len(labels)), c=[colours[l] for l in labels[draw]],
               linewidths=0, rasterized=True, alpha=0.9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.tick_params(length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    _legend(ax, [l for l in order if sizes[l] > 0], colours, counts=sizes, ncol=ncol)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path
