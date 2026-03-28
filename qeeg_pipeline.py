"""
qeeg_pipeline.py
Standalone QEEG analysis script with no UI.
Run with: python qeeg_pipeline.py
"""

import os

import mne
import numpy as np

from band_power import compute_band_power
from edf_loader import load_edf
from normative import compute_zscores, load_norms
from preprocessor import preprocess
from report import default_metadata, generate_report
from visualizer import plot_topomaps


EDF_PATH = r"C:\edfs\1.edf"
OUTPUT_DIR = "qeeg_output"
REFERENCE_MODE = "linked_ears"


def main():
    print("=" * 50)
    print("  QEEG PIPELINE - Standalone Test")
    print("=" * 50)

    if not os.path.exists(EDF_PATH):
        print(f"\n{'!' * 55}")
        print("  DEMO MODE - synthetic data only")
        print(f"  EDF file not found: '{EDF_PATH}'")
        print("  Results are NOT from real patient data.")
        print("  Update EDF_PATH at the top of this script.")
        print(f"{'!' * 55}\n")
        raw = _make_synthetic_raw()
    else:
        raw = load_edf(EDF_PATH, rereference=REFERENCE_MODE)

    clean_data, epochs = preprocess(raw, interactive=True)
    band_power = compute_band_power(clean_data, raw.info["sfreq"], raw.ch_names)

    metadata = default_metadata(raw, band_power["n_epochs"], EDF_PATH)
    norms = load_norms()
    zscores = compute_zscores(band_power, norms, patient_age=metadata.get("patient_age"))

    topomap_paths = plot_topomaps(zscores, raw, OUTPUT_DIR)
    generate_report(
        metadata=metadata,
        band_power=band_power,
        zscores=zscores,
        topomap_paths=topomap_paths,
        output_path=os.path.join(OUTPUT_DIR, "report.pdf"),
    )

    print("\n[OK] Pipeline complete.")
    print(f"  Output folder: {os.path.abspath(OUTPUT_DIR)}")


def _make_synthetic_raw() -> mne.io.Raw:
    print("[DEMO] Generating synthetic 10-20 EEG data...")
    sfreq = 256
    duration = 60
    n_times = int(sfreq * duration)

    ch_names = [
        "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
        "T7", "C3", "CZ", "C4", "T8",
        "P7", "P3", "PZ", "P4", "P8",
        "O1", "O2",
    ]
    n_ch = len(ch_names)

    rng = np.random.default_rng(42)
    white = rng.standard_normal((n_ch, n_times))
    freqs = np.fft.rfftfreq(n_times, 1.0 / sfreq)
    freqs[0] = 1
    pink_fft = np.fft.rfft(white, axis=1) / np.sqrt(freqs)
    data = np.fft.irfft(pink_fft, n=n_times, axis=1)
    data *= 20e-6

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * n_ch)
    raw = mne.io.RawArray(data, info, verbose=False)

    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False, on_missing="ignore")

    print(f"    Synthetic data: {n_ch} channels x {duration}s @ {sfreq} Hz")
    return raw


if __name__ == "__main__":
    main()
