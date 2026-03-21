"""
normative.py
Loads normative data from JSON, computes Z-scores per channel/band,
and provides utilities for updating or replacing the normative dataset.

Schema (normative_data/placeholder_norms.json):
{
  "metadata": {
    "version": "1.1",
    "source": "...",
    "power_type": "relative"
  },
  "norms": {
    "FP1": {
      "Delta":  {"mean": 0.28, "std": 0.08},
      "Theta":  {"mean": 0.18, "std": 0.06},
      ...
    },
    ...
  }
}
"""

import json
import os
import numpy as np
from pathlib import Path
from band_power import FREQ_BANDS

DEFAULT_NORMS_PATH = "normative_data/placeholder_norms.json"


# ─────────────────────────────────────────────────────────────────────────────
# LOAD / SAVE
# ─────────────────────────────────────────────────────────────────────────────
def load_norms(path: str = DEFAULT_NORMS_PATH) -> dict:
    """
    Load normative dataset from JSON.
    Falls back to a minimal built-in placeholder if file is missing.
    """
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        source = data.get("metadata", {}).get("source", "unknown")
        n_ch   = len(data.get("norms", {}))
        print(f"\n[NORMS] Loaded: {path}")
        print(f"    Source   : {source}")
        print(f"    Channels : {n_ch}")
        return data
    else:
        print(f"\n[NORMS] ⚠️  File not found: {path}")
        print(f"    Using built-in placeholder norms.")
        return _builtin_placeholder()


def save_norms(norms: dict, path: str = DEFAULT_NORMS_PATH):
    """Save normative dataset to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(norms, f, indent=2)
    print(f"[NORMS] Saved to: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Z-SCORE COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
def compute_zscores(band_power: dict, norms: dict) -> dict:
    """
    Compute Z-scores: (patient_value - norm_mean) / norm_std
    Operates on relative power. Returns {band: {channel: z_score}}

    Channels or bands missing from norms return 0.0 with a warning.
    """
    print(f"\n[NORMS] Computing Z-scores...")

    norm_data  = norms.get("norms", {})
    zscores    = {}
    missing    = []

    for band in FREQ_BANDS:
        zscores[band] = {}
        for ch, value in band_power["relative"][band].items():
            if ch in norm_data and band in norm_data[ch]:
                mean = norm_data[ch][band]["mean"]
                std  = norm_data[ch][band]["std"]
                z    = (value - mean) / std if std > 0 else 0.0
                zscores[band][ch] = round(float(z), 4)
            else:
                zscores[band][ch] = 0.0
                missing.append(f"{ch}/{band}")

    if missing:
        print(f"    ⚠️  Missing norm entries (scored 0.0): {missing}")

    _print_zscore_table(zscores, band_power["ch_names"])
    return zscores


def _print_zscore_table(zscores: dict, ch_names: list[str]):
    """Pretty-print Z-score table with ← flags at |Z| ≥ 2.0."""
    bands  = list(FREQ_BANDS.keys())
    header = f"    {'Channel':<8}" + "".join(f"  {b:<12}" for b in bands)

    print(f"\n    {'─' * 60}")
    print(f"    Z-Score Table (vs. normative, relative power)")
    print(f"    {'─' * 60}")
    print(header)

    for ch in ch_names:
        row = f"    {ch:<8}"
        for band in bands:
            z    = zscores[band].get(ch, 0.0)
            flag = " ←" if abs(z) >= 2.0 else "   "
            row += f"  {z:+.2f}{flag}    "
        print(row)


# ─────────────────────────────────────────────────────────────────────────────
# NORMATIVE DATABASE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def add_recording_to_norms(
    band_power: dict,
    norms: dict,
    path: str = DEFAULT_NORMS_PATH
) -> dict:
    """
    Add a new recording to the bootstrapped normative dataset.
    Updates running mean and std using Welford's online algorithm.

    Use this to build your own normative database over time:
        norms = load_norms()
        norms = add_recording_to_norms(band_power, norms)
        save_norms(norms)

    The metadata tracks how many recordings have been added.
    """
    norm_data = norms.setdefault("norms", {})
    meta      = norms.setdefault("metadata", {})
    n         = meta.get("n_recordings", 0) + 1
    meta["n_recordings"] = n

    for band in FREQ_BANDS:
        for ch, value in band_power["relative"][band].items():
            if ch not in norm_data:
                norm_data[ch] = {}
            if band not in norm_data[ch]:
                # First recording — initialize
                norm_data[ch][band] = {
                    "mean": value,
                    "std":  0.0,
                    "M2":   0.0,    # Welford's running sum of squares
                    "n":    1
                }
            else:
                # Welford's online update
                entry   = norm_data[ch][band]
                n_prev  = entry.get("n", 1)
                mean    = entry["mean"]
                M2      = entry.get("M2", 0.0)

                n_new   = n_prev + 1
                delta   = value - mean
                mean    += delta / n_new
                delta2  = value - mean
                M2      += delta * delta2

                entry["mean"] = round(mean, 6)
                entry["std"]  = round(float(np.sqrt(M2 / n_new)) if n_new > 1 else 0.0, 6)
                entry["M2"]   = M2
                entry["n"]    = n_new

    print(f"[NORMS] Recording added. Total in database: {n}")
    return norms


def get_norm_summary(norms: dict) -> str:
    """Return a readable summary of the normative dataset."""
    meta  = norms.get("metadata", {})
    n_ch  = len(norms.get("norms", {}))
    n_rec = meta.get("n_recordings", "unknown")
    src   = meta.get("source", "unknown")
    ver   = meta.get("version", "?")

    return (
        f"Normative database v{ver}\n"
        f"  Source      : {src}\n"
        f"  Channels    : {n_ch}\n"
        f"  Recordings  : {n_rec}\n"
        f"  Power type  : {meta.get('power_type', 'relative')}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# BUILT-IN PLACEHOLDER (fallback if JSON missing)
# ─────────────────────────────────────────────────────────────────────────────
def _builtin_placeholder() -> dict:
    channels = [
        "FP1","FP2","F7","F3","FZ","F4","F8",
        "T7","C3","CZ","C4","T8",
        "P7","P3","PZ","P4","P8",
        "O1","O2"
    ]
    return {
        "metadata": {
            "version": "1.1",
            "source": "built-in placeholder",
            "power_type": "relative"
        },
        "norms": {
            ch: {
                "Delta":  {"mean": 0.28, "std": 0.08},
                "Theta":  {"mean": 0.18, "std": 0.06},
                "Alpha":  {"mean": 0.30, "std": 0.09},
                "Beta":   {"mean": 0.10, "std": 0.04},
                "HiBeta": {"mean": 0.04, "std": 0.02},
                "Gamma":  {"mean": 0.03, "std": 0.02},
            }
            for ch in channels
        }
    }