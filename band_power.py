"""
band_power.py
FFT-based absolute and relative band power extraction from clean epoched data.
Operates on numpy arrays output by preprocessor.py.
"""

import numpy as np

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
    sfreq: float,
    ch_names: list[str],
) -> dict:
    """
    Compute absolute and relative FFT band power per channel,
    averaged across all clean epochs.

    Parameters
    ----------
    clean_data : np.ndarray, shape (n_epochs, n_channels, n_times)
        Clean epoched EEG data in volts, from preprocessor.finalize_epochs()
    sfreq      : float
        Sampling frequency in Hz (e.g. 256.0)
    ch_names   : list[str]
        Channel names in the same order as axis 1 of clean_data

    Returns
    -------
    result : dict with keys:
        "absolute" : {band: {channel: float}}  — power in V²
        "relative" : {band: {channel: float}}  — fraction of broadband power
        "sfreq"    : float
        "ch_names" : list[str]
        "n_epochs" : int
    """
    n_epochs, n_channels, n_times = clean_data.shape

    print(f"\n[BAND POWER] Computing FFT power...")
    print(f"    Epochs used : {n_epochs}")
    print(f"    Channels    : {n_channels}")
    print(f"    Epoch length: {n_times / sfreq:.1f}s @ {sfreq} Hz")

    # ── Average epochs before FFT ─────────────────────────────────────────────
    data = clean_data.mean(axis=0)          # shape: (n_channels, n_times)

    # ── FFT power spectrum ────────────────────────────────────────────────────
    freqs     = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    fft_power = (np.abs(np.fft.rfft(data, axis=1)) ** 2) / n_times

    # ── Broadband power (1–40 Hz) for relative normalization ─────────────────
    broadband_mask = (freqs >= 1.0) & (freqs <= 40.0)
    total_power    = fft_power[:, broadband_mask].sum(axis=1)
    total_power    = np.where(total_power > 0, total_power, np.nan)

    # ── Per-band extraction ───────────────────────────────────────────────────
    absolute_power = {}
    relative_power = {}

    for band, (fmin, fmax) in FREQ_BANDS.items():
        band_mask     = (freqs >= fmin) & (freqs < fmax)
        band_power_ch = fft_power[:, band_mask].sum(axis=1)

        absolute_power[band] = {
            ch: float(band_power_ch[i])
            for i, ch in enumerate(ch_names)
        }
        relative_power[band] = {
            ch: float(band_power_ch[i] / total_power[i])
            for i, ch in enumerate(ch_names)
        }

    # ── Print summary table ───────────────────────────────────────────────────
    _print_power_table(ch_names, relative_power, label="relative power")

    return {
        "absolute":  absolute_power,
        "relative":  relative_power,
        "sfreq":     sfreq,
        "ch_names":  ch_names,
        "n_epochs":  n_epochs,
    }


def _print_power_table(ch_names: list[str], power: dict, label: str = ""):
    """Pretty-print a channels × bands table to the console."""
    bands = list(FREQ_BANDS.keys())
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