"""
normative.py
Loads normative data from JSON, computes Z-scores per channel/band.
Missing norm entries return NaN — never silently zero.
Adult norms derived from published literature (eyes-closed resting state).
Sources:
  - Nayak & Anilkumar (2022) StatPearls: Normal EEG
  - Sanei & Chambers (2007) EEG Signal Processing
  - Niedermeyer & da Silva (2004) Electroencephalography 5th ed.
  - Barry & De Blasio (2017) Clinical Neurophysiology
"""

import json
import os
import math
import numpy as np
from pathlib import Path
from band_power import FREQ_BANDS

DEFAULT_NORMS_PATH = "normative_data/norms.json"
MIN_ADULT_AGE      = 18


# ─────────────────────────────────────────────────────────────────────────────
# LOAD / SAVE
# ─────────────────────────────────────────────────────────────────────────────
def load_norms(path: str = DEFAULT_NORMS_PATH) -> dict:
    """
    Load normative dataset from JSON.
    Falls back to built-in literature-derived adult norms if file missing.
    """
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        source = data.get("metadata", {}).get("source", "unknown")
        n_ch   = len(data.get("norms", {}).get("adult", {}))
        print(f"\n[NORMS] Loaded: {path}")
        print(f"    Source   : {source}")
        print(f"    Channels : {n_ch}")
        return data
    else:
        print(f"\n[NORMS] File not found: {path}")
        print(f"    Using built-in literature-derived adult norms.")
        norms = _literature_adult_norms()
        save_norms(norms, path)
        return norms


def save_norms(norms: dict, path: str = DEFAULT_NORMS_PATH):
    """Save normative dataset to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(norms, f, indent=2)
    print(f"[NORMS] Saved to: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Z-SCORE COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
def compute_zscores(
    band_power: dict,
    norms:      dict,
    patient_age: int = None,
) -> dict:
    """
    Compute Z-scores: (patient_value - norm_mean) / norm_std
    Operates on relative power.

    Rules:
      - Patient age < MIN_ADULT_AGE → skip Z-scores, return None
      - Missing norm entry → NaN (never 0.0)
      - std == 0 → NaN

    Returns {band: {channel: float}} or None if age-ineligible.
    """
    # ── Age gate ──────────────────────────────────────────────────────────────
    if patient_age is not None and patient_age < MIN_ADULT_AGE:
        print(f"\n[NORMS] ⚠️  Patient age {patient_age} < {MIN_ADULT_AGE}.")
        print(f"    Z-scores skipped — adult norms not applicable to children.")
        print(f"    Topomaps and band power are still computed.")
        return None

    print(f"\n[NORMS] Computing Z-scores...")

    # Select adult stratum
    norm_data = norms.get("norms", {}).get("adult", {})
    if not norm_data:
        print(f"    ⚠️  No adult norms found in database.")
        return None

    zscores = {}
    missing = []

    for band in FREQ_BANDS:
        zscores[band] = {}
        for ch, value in band_power["relative"][band].items():
            if ch in norm_data and band in norm_data[ch]:
                mean = norm_data[ch][band]["mean"]
                std  = norm_data[ch][band]["std"]
                if std > 0:
                    z = (value - mean) / std
                    zscores[band][ch] = round(float(z), 4)
                else:
                    zscores[band][ch] = float("nan")
                    missing.append(f"{ch}/{band}(std=0)")
            else:
                # Missing entry → NaN, never 0.0
                zscores[band][ch] = float("nan")
                missing.append(f"{ch}/{band}")

    if missing:
        print(f"    ⚠️  Missing/invalid norm entries (NaN): {missing}")

    _print_zscore_table(zscores, band_power["ch_names"])
    return zscores


def _print_zscore_table(zscores: dict, ch_names: list[str]):
    """Pretty-print Z-score table. NaN shown explicitly, ← flags |Z| ≥ 2.0."""
    bands  = list(FREQ_BANDS.keys())
    header = f"    {'Channel':<8}" + "".join(f"  {b:<12}" for b in bands)

    print(f"\n    {'─' * 60}")
    print(f"    Z-Score Table (vs. adult normative, relative power)")
    print(f"    {'─' * 60}")
    print(header)

    for ch in ch_names:
        row = f"    {ch:<8}"
        for band in bands:
            z = zscores[band].get(ch, float("nan"))
            if math.isnan(z):
                row += f"  {'NaN':<14}"
            else:
                flag = " ←" if abs(z) >= 2.0 else "   "
                row += f"  {z:+.2f}{flag}    "
        print(row)


# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAPPED NORM UPDATER
# ─────────────────────────────────────────────────────────────────────────────
def add_recording_to_norms(
    band_power:   dict,
    norms:        dict,
    patient_age:  int,
    path:         str = DEFAULT_NORMS_PATH,
) -> dict:
    """
    Add a recording to the bootstrapped normative dataset.
    Only adds adult recordings (age >= MIN_ADULT_AGE).
    Uses Welford's online algorithm for running mean/std.
    """
    if patient_age < MIN_ADULT_AGE:
        print(f"[NORMS] Skipping bootstrap — patient age {patient_age} < {MIN_ADULT_AGE}")
        return norms

    norm_data = norms.setdefault("norms", {}).setdefault("adult", {})
    meta      = norms.setdefault("metadata", {})
    n         = meta.get("n_recordings", 0) + 1
    meta["n_recordings"] = n

    for band in FREQ_BANDS:
        for ch, value in band_power["relative"][band].items():
            if ch not in norm_data:
                norm_data[ch] = {}
            if band not in norm_data[ch]:
                norm_data[ch][band] = {
                    "mean": value, "std": 0.0, "M2": 0.0, "n": 1
                }
            else:
                entry  = norm_data[ch][band]
                n_prev = entry.get("n", 1)
                mean   = entry["mean"]
                M2     = entry.get("M2", 0.0)
                n_new  = n_prev + 1
                delta  = value - mean
                mean  += delta / n_new
                delta2 = value - mean
                M2    += delta * delta2
                entry["mean"] = round(mean, 6)
                entry["std"]  = round(float(np.sqrt(M2 / n_new)) if n_new > 1 else 0.0, 6)
                entry["M2"]   = M2
                entry["n"]    = n_new

    print(f"[NORMS] Recording added to bootstrap. Total: {n}")
    return norms


# ─────────────────────────────────────────────────────────────────────────────
# LITERATURE-DERIVED ADULT NORMS
# ─────────────────────────────────────────────────────────────────────────────
def _literature_adult_norms() -> dict:
    """
    Adult resting-state eyes-closed normative values derived from
    published literature. Relative power, 19-channel 10-20 system.

    These are region-differentiated — frontal, central, parietal,
    occipital, and temporal channels have different expected profiles.

    Sources:
      Nayak & Anilkumar (2022) StatPearls
      Barry & De Blasio (2017) Clin Neurophysiology 128:2041-2050
      Clarke et al. (2001) Clin Neurophysiology 112:1394-1401
      Sanei & Chambers (2007) EEG Signal Processing, Wiley

    NOTE: These are literature-derived approximations, not a validated
    normative database. Use for reference only. Replace with a proper
    dataset (e.g. LEMON) when available.
    """

    # Region-specific norms reflect known topographic distribution:
    # - Frontal: higher delta/theta, lower alpha
    # - Occipital/parietal: dominant alpha, low delta
    # - Central: intermediate
    # - Temporal: intermediate with some alpha

    regions = {
        # channel: (Delta,  Theta,  Alpha,  Beta,   HiBeta, Gamma)
        #          mean/std pairs per band

        # ── Frontal ───────────────────────────────────────────────
        "FP1": dict(
            Delta  = (0.32, 0.09), Theta  = (0.21, 0.07),
            Alpha  = (0.22, 0.08), Beta   = (0.13, 0.05),
            HiBeta = (0.06, 0.03), Gamma  = (0.04, 0.02)
        ),
        "FP2": dict(
            Delta  = (0.32, 0.09), Theta  = (0.21, 0.07),
            Alpha  = (0.22, 0.08), Beta   = (0.13, 0.05),
            HiBeta = (0.06, 0.03), Gamma  = (0.04, 0.02)
        ),
        "F7": dict(
            Delta  = (0.28, 0.08), Theta  = (0.20, 0.07),
            Alpha  = (0.25, 0.08), Beta   = (0.14, 0.05),
            HiBeta = (0.06, 0.03), Gamma  = (0.04, 0.02)
        ),
        "F3": dict(
            Delta  = (0.27, 0.08), Theta  = (0.19, 0.07),
            Alpha  = (0.27, 0.08), Beta   = (0.14, 0.05),
            HiBeta = (0.06, 0.03), Gamma  = (0.04, 0.02)
        ),
        "FZ": dict(
            Delta  = (0.27, 0.08), Theta  = (0.19, 0.07),
            Alpha  = (0.27, 0.08), Beta   = (0.14, 0.05),
            HiBeta = (0.06, 0.03), Gamma  = (0.04, 0.02)
        ),
        "F4": dict(
            Delta  = (0.27, 0.08), Theta  = (0.19, 0.07),
            Alpha  = (0.27, 0.08), Beta   = (0.14, 0.05),
            HiBeta = (0.06, 0.03), Gamma  = (0.04, 0.02)
        ),
        "F8": dict(
            Delta  = (0.28, 0.08), Theta  = (0.20, 0.07),
            Alpha  = (0.25, 0.08), Beta   = (0.14, 0.05),
            HiBeta = (0.06, 0.03), Gamma  = (0.04, 0.02)
        ),

        # ── Central ───────────────────────────────────────────────
        "CZ": dict(
            Delta  = (0.22, 0.07), Theta  = (0.17, 0.06),
            Alpha  = (0.35, 0.09), Beta   = (0.14, 0.05),
            HiBeta = (0.05, 0.03), Gamma  = (0.03, 0.02)
        ),
        "C3": dict(
            Delta  = (0.22, 0.07), Theta  = (0.17, 0.06),
            Alpha  = (0.35, 0.09), Beta   = (0.14, 0.05),
            HiBeta = (0.05, 0.03), Gamma  = (0.03, 0.02)
        ),
        "C4": dict(
            Delta  = (0.22, 0.07), Theta  = (0.17, 0.06),
            Alpha  = (0.35, 0.09), Beta   = (0.14, 0.05),
            HiBeta = (0.05, 0.03), Gamma  = (0.03, 0.02)
        ),

        # ── Temporal ──────────────────────────────────────────────
        "T7": dict(
            Delta  = (0.24, 0.08), Theta  = (0.18, 0.06),
            Alpha  = (0.32, 0.09), Beta   = (0.14, 0.05),
            HiBeta = (0.05, 0.03), Gamma  = (0.03, 0.02)
        ),
        "T8": dict(
            Delta  = (0.24, 0.08), Theta  = (0.18, 0.06),
            Alpha  = (0.32, 0.09), Beta   = (0.14, 0.05),
            HiBeta = (0.05, 0.03), Gamma  = (0.03, 0.02)
        ),

        # ── Parietal ──────────────────────────────────────────────
        "P7": dict(
            Delta  = (0.16, 0.06), Theta  = (0.14, 0.05),
            Alpha  = (0.46, 0.10), Beta   = (0.13, 0.05),
            HiBeta = (0.04, 0.02), Gamma  = (0.03, 0.02)
        ),
        "P3": dict(
            Delta  = (0.15, 0.06), Theta  = (0.13, 0.05),
            Alpha  = (0.48, 0.10), Beta   = (0.13, 0.05),
            HiBeta = (0.04, 0.02), Gamma  = (0.03, 0.02)
        ),
        "PZ": dict(
            Delta  = (0.15, 0.06), Theta  = (0.13, 0.05),
            Alpha  = (0.48, 0.10), Beta   = (0.13, 0.05),
            HiBeta = (0.04, 0.02), Gamma  = (0.03, 0.02)
        ),
        "P4": dict(
            Delta  = (0.15, 0.06), Theta  = (0.13, 0.05),
            Alpha  = (0.48, 0.10), Beta   = (0.13, 0.05),
            HiBeta = (0.04, 0.02), Gamma  = (0.03, 0.02)
        ),
        "P8": dict(
            Delta  = (0.16, 0.06), Theta  = (0.14, 0.05),
            Alpha  = (0.46, 0.10), Beta   = (0.13, 0.05),
            HiBeta = (0.04, 0.02), Gamma  = (0.03, 0.02)
        ),

        # ── Occipital ─────────────────────────────────────────────
        "O1": dict(
            Delta  = (0.12, 0.05), Theta  = (0.10, 0.04),
            Alpha  = (0.57, 0.11), Beta   = (0.12, 0.05),
            HiBeta = (0.04, 0.02), Gamma  = (0.03, 0.02)
        ),
        "O2": dict(
            Delta  = (0.12, 0.05), Theta  = (0.10, 0.04),
            Alpha  = (0.57, 0.11), Beta   = (0.12, 0.05),
            HiBeta = (0.04, 0.02), Gamma  = (0.03, 0.02)
        ),
    }

    # Convert to schema format
    norms_dict = {}
    for ch, bands in regions.items():
        norms_dict[ch] = {
            band: {"mean": mean, "std": std}
            for band, (mean, std) in bands.items()
        }

    return {
        "metadata": {
            "version": "2.0",
            "source": (
                "Literature-derived adult norms (eyes-closed resting state). "
                "Nayak & Anilkumar 2022 StatPearls; Barry & De Blasio 2017 "
                "Clin Neurophysiology; Clarke et al 2001 Clin Neurophysiology; "
                "Sanei & Chambers 2007 EEG Signal Processing. "
                "NOT a validated normative database — reference use only."
            ),
            "power_type":    "relative",
            "eyes_condition": "closed",
            "age_range":     "18-99",
            "n_recordings":  0,
            "age_strata": {
                "adult": {"min_age": 18, "max_age": 99},
                "child": None
            }
        },
        "norms": {
            "adult": norms_dict,
            "child": None
        }
    }


def get_norm_summary(norms: dict) -> str:
    """Return a readable summary of the normative dataset."""
    meta  = norms.get("metadata", {})
    n_ch  = len(norms.get("norms", {}).get("adult", {}))
    n_rec = meta.get("n_recordings", 0)
    src   = meta.get("source", "unknown")

    return (
        f"Normative database v{meta.get('version', '?')}\n"
        f"  Source      : {src[:80]}...\n"
        f"  Channels    : {n_ch}\n"
        f"  Recordings  : {n_rec}\n"
        f"  Age range   : {meta.get('age_range', '?')}\n"
        f"  Power type  : {meta.get('power_type', 'relative')}\n"
        f"  Condition   : {meta.get('eyes_condition', '?')}"
    )