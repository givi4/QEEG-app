"""
qeeg_pipeline.py
Standalone QEEG analysis script — no UI.
Run with: python qeeg_pipeline.py

Requires: mne, scipy, numpy, matplotlib
"""

import mne
from edf_loader import load_edf
from preprocessor import preprocess
from band_power import compute_band_power, FREQ_BANDS
from normative import load_norms, compute_zscores
from visualizer import plot_topomaps
from report import generate_report, default_metadata
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURATION — edit these
# ─────────────────────────────────────────────
EDF_PATH = r"C:\EDFS\1.edf"          # ← Replace with your actual EDF path
OUTPUT_DIR = "qeeg_output"          # Folder for topomaps and results
PLACEHOLDER_NORMS_PATH = "normative_data/placeholder_norms.json"









# ─────────────────────────────────────────────
# MAIN — Wire everything together
# ─────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  QEEG PIPELINE — Standalone Test")
    print("=" * 50)



    # Check EDF file exists
    if not os.path.exists(EDF_PATH):
        print(f"\n⚠️  EDF file not found: '{EDF_PATH}'")
        print("   Update EDF_PATH at the top of this script and re-run.")
        print("\n   Running in DEMO MODE with synthetic data instead...\n")
        raw = _make_synthetic_raw()    # ← lets you test without a real EDF
    else:
        raw = load_edf(EDF_PATH)

    clean_data, epochs = preprocess(raw, interactive=True)
    band_power = compute_band_power(clean_data, raw.info["sfreq"], raw.ch_names)
    norms   = load_norms()
    zscores = compute_zscores(band_power, norms)
    topomap_paths = plot_topomaps(zscores, raw, OUTPUT_DIR)

    metadata = default_metadata(raw, band_power["n_epochs"], EDF_PATH)
    report_path = generate_report(
        metadata      = metadata,
        band_power    = band_power,
        zscores       = zscores,
        topomap_paths = topomap_paths,
        output_path   = os.path.join(OUTPUT_DIR, "report.pdf")
    )

    print("\n✓ Pipeline complete.")
    print(f"  Output folder: {os.path.abspath(OUTPUT_DIR)}")


# ─────────────────────────────────────────────
# DEMO MODE — Synthetic raw object (no EDF needed)
# ─────────────────────────────────────────────
def _make_synthetic_raw() -> mne.io.Raw:
    """
    Generates a synthetic 19-channel 10-20 raw object for pipeline testing.
    Produces plausible EEG-like power spectrum via colored noise.
    """
    print("[DEMO] Generating synthetic 10-20 EEG data...")
    sfreq = 256
    duration = 60   # seconds
    n_times = int(sfreq * duration)

    ch_names = [
        "FP1","FP2","F7","F3","FZ","F4","F8",
        "T7","C3","CZ","C4","T8",
        "P7","P3","PZ","P4","P8",
        "O1","O2"
    ]
    n_ch = len(ch_names)

    rng = np.random.default_rng(42)
    # 1/f noise (pink noise approximation) — more realistic than white noise
    white = rng.standard_normal((n_ch, n_times))
    freqs = np.fft.rfftfreq(n_times, 1.0 / sfreq)
    freqs[0] = 1   # avoid divide by zero
    pink_fft = np.fft.rfft(white, axis=1) / np.sqrt(freqs)
    data = np.fft.irfft(pink_fft, n=n_times, axis=1)
    data *= 20e-6   # scale to realistic EEG amplitude (~20 µV)

    ch_types = ["eeg"] * n_ch
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info, verbose=False)

    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False, on_missing="ignore")

    print(f"    Synthetic data: {n_ch} channels × {duration}s @ {sfreq} Hz")
    return raw


if __name__ == "__main__":
    main()