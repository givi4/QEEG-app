"""
artifact_detection.py
Heuristic artifact detection for EEG Raw data using short windows.
Designed to be simple, interpretable, and easy to extend later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.stats import kurtosis


@dataclass
class ArtifactDetectionConfig:
    window_length_s: float = 2.0
    step_length_s: float | None = None
    amplitude_threshold_uV: float = 150.0
    std_threshold_uV: float = 60.0
    kurtosis_threshold: float = 8.0
    flatline_ptp_uV: float = 1.0
    min_correlation: float = 0.3
    bad_window_channel_fraction: float = 0.25
    bad_channel_window_fraction: float = 0.30
    score_threshold: int = 2
    annotation_label: str = "bad_artifact"


@dataclass
class ArtifactDetectionResult:
    ch_names: list[str]
    sfreq: float
    window_starts_s: np.ndarray
    window_duration_s: float
    metrics: dict[str, np.ndarray]
    flags: dict[str, np.ndarray]
    channel_scores: np.ndarray
    bad_channel_windows: np.ndarray
    window_scores: np.ndarray
    bad_windows: np.ndarray
    bad_channels: list[str]
    config: ArtifactDetectionConfig = field(repr=False)


def _prepare_windows(raw: mne.io.Raw, config: ArtifactDetectionConfig) -> tuple[np.ndarray, list[str], np.ndarray]:
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(picks) == 0:
        raise ValueError("No EEG channels available for artifact detection.")

    data = raw.get_data(picks=picks)
    ch_names = [raw.ch_names[idx] for idx in picks]
    sfreq = raw.info["sfreq"]

    step_length_s = config.step_length_s or config.window_length_s
    win_samples = int(round(config.window_length_s * sfreq))
    step_samples = int(round(step_length_s * sfreq))

    if win_samples <= 1 or step_samples <= 0:
        raise ValueError("Window length and step length must be positive.")
    if data.shape[1] < win_samples:
        raise ValueError("Raw recording is shorter than the artifact window length.")

    starts = np.arange(0, data.shape[1] - win_samples + 1, step_samples, dtype=int)
    windows = np.stack([data[:, start:start + win_samples] for start in starts], axis=0)
    return windows, ch_names, starts / sfreq


def detect_artifacts(
    raw: mne.io.Raw,
    config: ArtifactDetectionConfig | None = None,
) -> ArtifactDetectionResult:
    """
    Detect artifacts in raw EEG using interpretable heuristics.
    """
    config = config or ArtifactDetectionConfig()
    windows, ch_names, window_starts_s = _prepare_windows(raw, config)

    data_uV = windows * 1e6
    peak_to_peak_uV = data_uV.max(axis=2) - data_uV.min(axis=2)
    std_uV = data_uV.std(axis=2)
    kurtosis_values = kurtosis(data_uV, axis=2, fisher=False, bias=False, nan_policy="omit")

    correlation_metric = np.zeros((data_uV.shape[0], data_uV.shape[1]), dtype=float)
    for w_idx, window in enumerate(data_uV):
        if window.shape[0] < 2:
            continue
        corr = np.corrcoef(window)
        corr = np.nan_to_num(corr, nan=0.0)
        abs_corr = np.abs(corr)
        np.fill_diagonal(abs_corr, np.nan)
        correlation_metric[w_idx] = np.nanmedian(abs_corr, axis=1)

    flags = {
        "amplitude": peak_to_peak_uV > config.amplitude_threshold_uV,
        "std": std_uV > config.std_threshold_uV,
        "kurtosis": kurtosis_values > config.kurtosis_threshold,
        "flatline": peak_to_peak_uV < config.flatline_ptp_uV,
        "correlation": correlation_metric < config.min_correlation,
    }

    channel_scores = sum(flag.astype(int) for flag in flags.values())
    bad_channel_windows = channel_scores >= config.score_threshold
    window_scores = bad_channel_windows.mean(axis=1)
    bad_windows = window_scores >= config.bad_window_channel_fraction

    channel_bad_fraction = bad_channel_windows.mean(axis=0)
    bad_channels = [
        ch_names[idx]
        for idx, frac in enumerate(channel_bad_fraction)
        if frac >= config.bad_channel_window_fraction
    ]

    return ArtifactDetectionResult(
        ch_names=ch_names,
        sfreq=raw.info["sfreq"],
        window_starts_s=window_starts_s,
        window_duration_s=config.window_length_s,
        metrics={
            "peak_to_peak_uV": peak_to_peak_uV,
            "std_uV": std_uV,
            "kurtosis": kurtosis_values,
            "median_abs_correlation": correlation_metric,
        },
        flags=flags,
        channel_scores=channel_scores,
        bad_channel_windows=bad_channel_windows,
        window_scores=window_scores,
        bad_windows=bad_windows,
        bad_channels=bad_channels,
        config=config,
    )


def result_to_annotations(result: ArtifactDetectionResult) -> mne.Annotations:
    """
    Convert bad windows into merged MNE annotations.
    """
    onsets: list[float] = []
    durations: list[float] = []
    descriptions: list[str] = []

    bad_indices = np.where(result.bad_windows)[0]
    if len(bad_indices) == 0:
        return mne.Annotations([], [], [])

    start_idx = bad_indices[0]
    prev_idx = bad_indices[0]

    for idx in bad_indices[1:]:
        contiguous = np.isclose(
            result.window_starts_s[idx],
            result.window_starts_s[prev_idx] + result.window_duration_s,
        )
        if contiguous:
            prev_idx = idx
            continue

        onset = float(result.window_starts_s[start_idx])
        duration = float(result.window_starts_s[prev_idx] - result.window_starts_s[start_idx] + result.window_duration_s)
        onsets.append(onset)
        durations.append(duration)
        descriptions.append(result.config.annotation_label)
        start_idx = idx
        prev_idx = idx

    onset = float(result.window_starts_s[start_idx])
    duration = float(result.window_starts_s[prev_idx] - result.window_starts_s[start_idx] + result.window_duration_s)
    onsets.append(onset)
    durations.append(duration)
    descriptions.append(result.config.annotation_label)

    return mne.Annotations(onsets, durations, descriptions)


def apply_artifact_annotations(
    raw: mne.io.Raw,
    result: ArtifactDetectionResult,
    mark_bad_channels: bool = True,
) -> mne.io.Raw:
    """
    Return a copy of raw with bad segments annotated and bad channels marked.
    """
    annotated = raw.copy()
    new_annotations = result_to_annotations(result)
    if len(annotated.annotations) > 0:
        annotated.set_annotations(annotated.annotations + new_annotations)
    else:
        annotated.set_annotations(new_annotations)

    if mark_bad_channels and result.bad_channels:
        annotated.info["bads"] = sorted(set(annotated.info.get("bads", []) + result.bad_channels))

    return annotated


def plot_artifact_summary(
    result: ArtifactDetectionResult,
    output_path: str | None = None,
    show: bool = False,
):
    """
    Save or show a static summary of channel/window artifact scores.
    """
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        constrained_layout=True,
        height_ratios=[3, 1],
    )

    heatmap = axes[0].imshow(
        result.channel_scores.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="magma",
    )
    axes[0].set_title("Artifact Score by Window and Channel")
    axes[0].set_ylabel("Channel")
    axes[0].set_yticks(np.arange(len(result.ch_names)))
    axes[0].set_yticklabels(result.ch_names)
    axes[0].set_xticks(np.arange(len(result.window_starts_s)))
    axes[0].set_xticklabels([f"{start:.0f}" for start in result.window_starts_s], rotation=90)
    axes[0].set_xlabel("Window Start (s)")
    fig.colorbar(heatmap, ax=axes[0], label="Metric count")

    axes[1].plot(result.window_starts_s, result.window_scores, color="#d32f2f", linewidth=2)
    axes[1].axhline(result.config.bad_window_channel_fraction, color="black", linestyle="--", linewidth=1)
    axes[1].fill_between(
        result.window_starts_s,
        0,
        result.window_scores,
        where=result.bad_windows,
        color="#ffb3b3",
        alpha=0.7,
    )
    axes[1].set_title("Fraction of Flagged Channels Per Window")
    axes[1].set_xlabel("Window Start (s)")
    axes[1].set_ylabel("Bad fraction")
    axes[1].set_ylim(0, 1)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_artifact_view(
    raw: mne.io.Raw,
    result: ArtifactDetectionResult,
    duration: float = 20.0,
    start: float = 0.0,
    block: bool = True,
):
    """
    Open MNE's raw viewer with bad segments shaded and bad channels marked.
    """
    annotated = apply_artifact_annotations(raw, result)
    return annotated.plot(
        start=start,
        duration=duration,
        n_channels=min(len(result.ch_names), 20),
        scalings="auto",
        block=block,
        title="Artifact Detection View",
    )
