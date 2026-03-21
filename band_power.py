"""
band_power.py
FFT-based absolute and relative band power extraction using Welch's method.
Computes PSD per epoch via MNE's built-in compute_psd(), then averages spectra.
This is the standard approach used in published qEEG research.
"""

import numpy as np
import mne

# ── Frequency band definitions ────────────────────────────────────────────────
FREQ_BANDS = {
    "Delta":  (1,  4),
    "Theta":  (4,  8),
    "Alpha":  (8,  12),
    "Beta":   (12, 25),
    "HiBeta": (25, 30),
    "Gamma":  (30, 40),
}


def compute_band_power(
    clean_data: np.ndarray,
    sfreq:      float,
    ch_names:   list[str],
) -> dict:
    """
    Compute absolute and relative band power using Welch's method.

    Pipeline:
      1. Reconstruct MNE Epochs object from clean numpy array
      2. Compute PSD per epoch using MNE's compute_psd() (Welch's method)
      3. Average PSDs across epochs
      4. Integrate power within each frequency band
      5. Compute relative power as fraction of broadband (1-40 Hz) power

    Parameters
    ----------
    clean_data : np.ndarray, shape (n_epochs, n_channels, n_times)
        Clean epoched EEG data in volts, from preprocessor.finalize_epochs()
    sfreq      : float
        Sampling frequency in Hz
    ch_names   : list[str]
        Channel names matching axis 1 of clean_data

    Returns
    -------
    result : dict with keys:
        "absolute" : {band: {channel: float}}  — power in V²/Hz
        "relative" : {band: {channel: float}}  — fraction of broadband power
        "freqs"    : np.ndarray                — frequency bins
        "psd_mean" : np.ndarray                — mean PSD (n_channels, n_freqs)
        "psd_std"  : np.ndarray                — std PSD across epochs
        "sfreq"    : float
        "ch_names" : list[str]
        "n_epochs" : int
    """
    n_epochs, n_channels, n_times = clean_data.shape

    print(f"\n[BAND POWER] Computing Welch PSD...")
    print(f"    Epochs used  : {n_epochs}")
    print(f"    Channels     : {n_channels}")
    print(f"    Epoch length : {n_times / sfreq:.1f}s @ {sfreq} Hz")
    print(f"    Method       : Welch (MNE compute_psd)")

    # ── Reconstruct MNE Epochs from numpy array ───────────────────────────────
    # We need a proper MNE Epochs object to use compute_psd()
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types=["eeg"] * n_channels,
        verbose=False
    )

    # Build minimal events array — one event per epoch
    events = np.column_stack([
        np.arange(n_epochs) * n_times,
        np.zeros(n_epochs, dtype=int),
        np.ones(n_epochs, dtype=int)
    ])

    epochs = mne.EpochsArray(
        clean_data,
        info,
        events=events,
        tmin=0.0,
        verbose=False
    )

    # Set standard montage so electrode positions are correct
    montage = mne.channels.make_standard_montage("standard_1020")
    epochs.set_montage(montage, match_case=False, on_missing="ignore")

    # ── Welch PSD via MNE ─────────────────────────────────────────────────────
    # n_fft = one full epoch length for maximum frequency resolution
    # This gives df = sfreq / n_times = 256/1280 = 0.2 Hz resolution
    n_fft = n_times

    psd = epochs.compute_psd(
        method="welch",
        fmin=1.0,
        fmax=40.0,
        n_fft=n_fft,
        n_overlap=n_fft // 2,    # 50% overlap — standard Welch's
        window="hamming",        # Hamming window — standard for EEG
        verbose=False,
    )

    # psd.get_data() returns shape: (n_epochs, n_channels, n_freqs)
    psd_data = psd.get_data()    # units: V²/Hz
    freqs    = psd.freqs         # frequency bins

    # Average across epochs and compute std for variance tracking
    psd_mean = psd_data.mean(axis=0)   # (n_channels, n_freqs)
    psd_std  = psd_data.std(axis=0)    # (n_channels, n_freqs)

    print(f"    Frequency resolution : {freqs[1]-freqs[0]:.3f} Hz")
    print(f"    Frequency bins       : {len(freqs)} ({freqs[0]:.1f}–{freqs[-1]:.1f} Hz)")

    # ── Broadband power for relative normalization ────────────────────────────
    # Integrate using the trapezoidal rule — more accurate than simple sum
    broadband_mask = (freqs >= 1.0) & (freqs <= 40.0)
    total_power = np.trapz(
        psd_mean[:, broadband_mask],
        freqs[broadband_mask],
        axis=1
    )   # shape: (n_channels,)
    total_power = np.where(total_power > 0, total_power, np.nan)

    # ── Per-band integration ──────────────────────────────────────────────────
    absolute_power = {}
    relative_power = {}

    for band, (fmin, fmax) in FREQ_BANDS.items():
        band_mask  = (freqs >= fmin) & (freqs < fmax)

        if not band_mask.any():
            print(f"    ⚠️  No frequency bins in {band} ({fmin}–{fmax} Hz) — skipping")
            absolute_power[band] = {ch: 0.0 for ch in ch_names}
            relative_power[band] = {ch: 0.0 for ch in ch_names}
            continue

        # Trapezoidal integration within band
        band_power_ch = np.trapz(
            psd_mean[:, band_mask],
            freqs[band_mask],
            axis=1
        )   # shape: (n_channels,)

        absolute_power[band] = {
            ch: float(band_power_ch[i])
            for i, ch in enumerate(ch_names)
        }
        relative_power[band] = {
            ch: float(band_power_ch[i] / total_power[i])
            for i, ch in enumerate(ch_names)
        }

    # ── Print summary table ───────────────────────────────────────────────────
    _print_power_table(ch_names, relative_power, label="relative power, Welch")

    return {
        "absolute":  absolute_power,
        "relative":  relative_power,
        "freqs":     freqs,
        "psd_mean":  psd_mean,
        "psd_std":   psd_std,
        "sfreq":     sfreq,
        "ch_names":  ch_names,
        "n_epochs":  n_epochs,
    }


def _print_power_table(ch_names: list[str], power: dict, label: str = ""):
    """Pretty-print a channels × bands table to the console."""
    bands  = list(FREQ_BANDS.keys())
    header = f"    {'Channel':<8}" + "".join(f"  {b:<10}" for b in bands)
    print(f"\n    {'─' * 60}")
    print(f"    Band power ({label})")
    print(f"    {'─' * 60}")
    print(header)
    for ch in ch_names:
        row = f"    {ch:<8}"
        for band in bands:
            val = power[band].get(ch, float("nan"))
            row += f"  {val:.3f}     "
        print(row)


def get_band_matrix(
    band_power: dict,
    power_type: str = "relative"
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Convert band power dict to a 2D numpy matrix for downstream use.

    Returns
    -------
    matrix   : np.ndarray, shape (n_channels, n_bands)
    ch_names : list[str]
    bands    : list[str]
    """
    ch_names = band_power["ch_names"]
    bands    = list(FREQ_BANDS.keys())
    power    = band_power[power_type]

    matrix = np.array([
        [power[band].get(ch, np.nan) for band in bands]
        for ch in ch_names
    ])

    return matrix, ch_names, bands