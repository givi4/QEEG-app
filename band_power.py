"""
band_power.py
FFT-based absolute and relative band power extraction using Welch's method.
"""

import mne
import numpy as np


FREQ_BANDS = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 12),
    "Beta": (12, 25),
    "HiBeta": (25, 30),
    "Gamma": (30, 40),
}


def compute_band_power(
    clean_data: np.ndarray,
    sfreq: float,
    ch_names: list[str],
) -> dict:
    """
    Compute absolute and relative band power using Welch's method.
    """
    n_epochs, n_channels, n_times = clean_data.shape

    print("\n[BAND POWER] Computing Welch PSD...")
    print(f"    Epochs used  : {n_epochs}")
    print(f"    Channels     : {n_channels}")
    print(f"    Epoch length : {n_times / sfreq:.1f}s @ {sfreq} Hz")
    print("    Method       : Welch (MNE compute_psd)")

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types=["eeg"] * n_channels,
        verbose=False,
    )

    events = np.column_stack([
        np.arange(n_epochs) * n_times,
        np.zeros(n_epochs, dtype=int),
        np.ones(n_epochs, dtype=int),
    ])

    epochs = mne.EpochsArray(
        clean_data,
        info,
        events=events,
        tmin=0.0,
        verbose=False,
    )

    montage = mne.channels.make_standard_montage("standard_1020")
    epochs.set_montage(montage, match_case=False, on_missing="ignore")

    n_fft = n_times

    psd = epochs.compute_psd(
        method="welch",
        fmin=1.0,
        fmax=40.0,
        n_fft=n_fft,
        n_overlap=n_fft // 2,
        window="hamming",
        verbose=False,
    )

    psd_data = psd.get_data()
    freqs = psd.freqs

    psd_mean = psd_data.mean(axis=0)
    psd_std = psd_data.std(axis=0)

    print(f"    Frequency resolution : {freqs[1] - freqs[0]:.3f} Hz")
    print(f"    Frequency bins       : {len(freqs)} ({freqs[0]:.1f}-{freqs[-1]:.1f} Hz)")

    broadband_mask = (freqs >= 1.0) & (freqs <= 40.0)
    total_power = np.trapezoid(psd_mean[:, broadband_mask], freqs[broadband_mask], axis=1)
    total_power = np.where(total_power > 0, total_power, np.nan)

    absolute_power = {}
    relative_power = {}

    for band, (fmin, fmax) in FREQ_BANDS.items():
        band_mask = (freqs >= fmin) & (freqs < fmax)

        if not band_mask.any():
            print(f"    [WARN] No frequency bins in {band} ({fmin}-{fmax} Hz); skipping")
            absolute_power[band] = {ch: 0.0 for ch in ch_names}
            relative_power[band] = {ch: 0.0 for ch in ch_names}
            continue

        band_power_ch = np.trapezoid(psd_mean[:, band_mask], freqs[band_mask], axis=1)

        absolute_power[band] = {
            ch: float(band_power_ch[i])
            for i, ch in enumerate(ch_names)
        }
        relative_power[band] = {
            ch: float(band_power_ch[i] / total_power[i])
            for i, ch in enumerate(ch_names)
        }

    _print_power_table(ch_names, relative_power, label="relative power, Welch")

    return {
        "absolute": absolute_power,
        "relative": relative_power,
        "freqs": freqs,
        "psd_mean": psd_mean,
        "psd_std": psd_std,
        "sfreq": sfreq,
        "ch_names": ch_names,
        "n_epochs": n_epochs,
    }


def _print_power_table(ch_names: list[str], power: dict, label: str = ""):
    bands = list(FREQ_BANDS.keys())
    header = f"    {'Channel':<8}" + "".join(f"  {b:<10}" for b in bands)
    print(f"\n    {'-' * 60}")
    print(f"    Band power ({label})")
    print(f"    {'-' * 60}")
    print(header)
    for ch in ch_names:
        row = f"    {ch:<8}"
        for band in bands:
            val = power[band].get(ch, float("nan"))
            row += f"  {val:.3f}     "
        print(row)


def get_band_matrix(
    band_power: dict,
    power_type: str = "relative",
) -> tuple[np.ndarray, list[str], list[str]]:
    ch_names = band_power["ch_names"]
    bands = list(FREQ_BANDS.keys())
    power = band_power[power_type]

    matrix = np.array([
        [power[band].get(ch, np.nan) for band in bands]
        for ch in ch_names
    ])

    return matrix, ch_names, bands
