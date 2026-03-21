"""
visualizer.py
Topographic Z-score maps for each frequency band.
Saves one PNG per band (for PDF report) plus one combined overview image.
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from band_power import FREQ_BANDS

# ── Visual style ──────────────────────────────────────────────────────────────
BG_COLOR    = "#0f1117"
TEXT_COLOR  = "#e0e0e0"
CMAP        = "RdBu_r"     # red = elevated, blue = suppressed
Z_LIMIT     = 3.0          # colorbar range: -3 to +3 SD
CONTOURS    = 4
DPI         = 150


def plot_topomaps(
    zscores:    dict,
    raw:        mne.io.Raw,
    output_dir: str = "qeeg_output",
) -> dict[str, str]:
    """
    Render one Z-score topomap per frequency band.

    Saves:
      - qeeg_output/topomap_<band>.png   — one per band (for PDF)
      - qeeg_output/topomaps_all_bands.png — combined overview

    Parameters
    ----------
    zscores    : {band: {channel: z_score}} from normative.compute_zscores()
    raw        : MNE Raw object (used for channel positions only)
    output_dir : folder to write PNGs into

    Returns
    -------
    paths : {"Delta": "qeeg_output/topomap_Delta.png", ...}
    """
    print(f"\n[VIZ] Rendering topomaps → {output_dir}/")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    bands    = list(FREQ_BANDS.keys())
    n_bands  = len(bands)
    channels = raw.ch_names
    info     = raw.info

    saved_paths = {}

    # ── Individual band PNGs (used by PDF report) ─────────────────────────────
    for band in bands:
        z_values = np.array([zscores[band].get(ch, 0.0) for ch in channels])

        fig, ax = plt.subplots(figsize=(4, 4), facecolor=BG_COLOR)
        ax.set_facecolor(BG_COLOR)

        im, _ = mne.viz.plot_topomap(
            z_values,
            info,
            axes=ax,
            cmap=CMAP,
            vlim=(-Z_LIMIT, Z_LIMIT),
            contours=CONTOURS,
            show=False,
        )

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.04)
        cbar.set_label("Z-score", color=TEXT_COLOR, fontsize=9)
        cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelsize=8)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)
        cbar.outline.set_edgecolor("#444444")

        # Band label + frequency range
        fmin, fmax = FREQ_BANDS[band]
        ax.set_title(
            f"{band}\n{fmin}–{fmax} Hz",
            color=TEXT_COLOR,
            fontsize=11,
            pad=8
        )

        path = str(Path(output_dir) / f"topomap_{band}.png")
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG_COLOR)
        plt.close(fig)
        saved_paths[band] = path
        print(f"    ✓ {band:8s} → {path}")

    # ── Combined overview PNG ─────────────────────────────────────────────────
    fig, axes = plt.subplots(
        1, n_bands,
        figsize=(4 * n_bands, 4.5),
        facecolor=BG_COLOR
    )

    last_im = None
    for ax, band in zip(axes, bands):
        ax.set_facecolor(BG_COLOR)
        z_values = np.array([zscores[band].get(ch, 0.0) for ch in channels])
        fmin, fmax = FREQ_BANDS[band]

        last_im, _ = mne.viz.plot_topomap(
            z_values,
            info,
            axes=ax,
            cmap=CMAP,
            vlim=(-Z_LIMIT, Z_LIMIT),
            contours=CONTOURS,
            show=False,
        )
        ax.set_title(
            f"{band}\n{fmin}–{fmax} Hz",
            color=TEXT_COLOR,
            fontsize=10,
            pad=6
        )

    # Single shared colorbar on the right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar    = fig.colorbar(last_im, cax=cbar_ax)
    cbar.set_label("Z-score", color=TEXT_COLOR, fontsize=10)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelsize=9)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)
    cbar.outline.set_edgecolor("#444444")

    fig.suptitle(
        "QEEG Z-Score Topomaps",
        color=TEXT_COLOR,
        fontsize=13,
        y=1.01
    )

    overview_path = str(Path(output_dir) / "topomaps_all_bands.png")
    fig.savefig(
        overview_path,
        dpi=DPI,
        bbox_inches="tight",
        facecolor=BG_COLOR
    )
    plt.close(fig)
    print(f"    ✓ Overview  → {overview_path}")

    saved_paths["_overview"] = overview_path
    return saved_paths