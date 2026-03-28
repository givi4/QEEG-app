"""
build_open_norms.py
Build a stratified open normative database from BIDS/OpenNeuro resting-state EEG.

First target:
  - OpenNeuro ds005385-style datasets
  - adult resting-state EEG
  - average-reference norms

Assumptions:
  - BIDS EEG files can be loaded with mne_bids
  - participant ages are available in participants.tsv
  - eyes condition can be inferred from task names containing open/closed or eo/ec
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import mne
import numpy as np

from artifact_detection import ArtifactDetectionConfig
from band_power import FREQ_BANDS, compute_band_power
from edf_loader import LEGACY_CHANNEL_MAP
from normative import save_norms
from preprocessor import preprocess


DEFAULT_OUTPUT = "normative_data/open_ds005385_norms.json"
DEFAULT_AGE_BINS = [(20, 29), (30, 39), (40, 49), (50, 59), (60, 70)]
CLINICAL_19 = [
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
    "T7", "C3", "CZ", "C4", "T8",
    "P7", "P3", "PZ", "P4", "P8",
    "O1", "O2",
]


def _parse_args():
    parser = argparse.ArgumentParser(description="Build open-source EEG norms from BIDS/OpenNeuro data.")
    parser.add_argument("bids_root", help="Path to the BIDS dataset root")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSON path")
    parser.add_argument("--reference-mode", default="average", choices=["average", "as_recorded"], help="Reference mode used during norm building")
    parser.add_argument("--eyes-condition", default="closed", choices=["open", "closed", "both"], help="Which eyes condition to include")
    parser.add_argument("--artifact-detection", action="store_true", help="Enable raw-window artifact detection before FFT")
    parser.add_argument("--window-length", type=float, default=2.0, help="Artifact detection window length in seconds")
    return parser.parse_args()


def _load_participants(participants_tsv: Path) -> dict[str, dict]:
    with participants_tsv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return {row["participant_id"]: row for row in reader}


def _infer_eyes_condition(task_name: str) -> str | None:
    task = (task_name or "").lower()
    if "closed" in task or task.endswith("ec") or "eyesclosed" in task:
        return "closed"
    if "open" in task or task.endswith("eo") or "eyesopen" in task:
        return "open"
    return None


def _age_bin_label(age: int, bins: list[tuple[int, int]]) -> tuple[str, int, int] | None:
    for age_min, age_max in bins:
        if age_min <= age <= age_max:
            return f"{age_min}-{age_max}", age_min, age_max
    return None


def _normalize_bids_raw(raw: mne.io.Raw, reference_mode: str) -> mne.io.Raw:
    rename_map = {}
    for ch in raw.ch_names:
        cleaned = ch.strip().upper()
        cleaned = LEGACY_CHANNEL_MAP.get(cleaned, cleaned)
        rename_map[ch] = cleaned
    raw = raw.copy()
    raw.rename_channels(rename_map)
    raw.pick([ch for ch in raw.ch_names if ch in CLINICAL_19])
    raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"), match_case=False, on_missing="ignore")
    if reference_mode == "average":
        raw.set_eeg_reference(ref_channels="average", projection=False, verbose=False)
    return raw


def _find_bids_eeg_files(bids_root: Path) -> list[Path]:
    return sorted(bids_root.rglob("*_eeg.*"))


def _read_raw_bids_file(eeg_file: Path, bids_root: Path):
    try:
        from mne_bids import BIDSPath, read_raw_bids
    except ImportError as exc:
        raise RuntimeError(
            "mne_bids is required to build open norms from BIDS data. "
            "Install it with: pip install mne-bids"
        ) from exc

    suffix = eeg_file.suffix.lower()
    ext = suffix
    filename = eeg_file.name
    parts = filename.split("_")
    entities = {}
    for part in parts[:-1]:
        if "-" in part:
            key, value = part.split("-", 1)
            entities[key] = value

    bids_path = BIDSPath(
        root=bids_root,
        subject=entities.get("sub"),
        session=entities.get("ses"),
        task=entities.get("task"),
        acquisition=entities.get("acq"),
        run=entities.get("run"),
        datatype="eeg",
        suffix="eeg",
        extension=ext,
    )
    raw = read_raw_bids(bids_path=bids_path, verbose=False)
    return raw, entities


def _init_running_stats():
    return {"n": 0, "mean": 0.0, "M2": 0.0}


def _update_running_stats(stats: dict, value: float):
    stats["n"] += 1
    delta = value - stats["mean"]
    stats["mean"] += delta / stats["n"]
    delta2 = value - stats["mean"]
    stats["M2"] += delta * delta2


def _finalize_running_stats(stats: dict) -> dict:
    n = stats["n"]
    variance = stats["M2"] / n if n > 1 else 0.0
    return {
        "mean": round(float(stats["mean"]), 6),
        "std": round(float(np.sqrt(max(variance, 0.0))), 6),
        "n": n,
    }


def build_open_norms(
    bids_root: Path,
    output_path: Path,
    reference_mode: str,
    eyes_condition: str,
    artifact_detection: bool,
    window_length: float,
):
    participants = _load_participants(bids_root / "participants.tsv")
    eeg_files = _find_bids_eeg_files(bids_root)
    if not eeg_files:
        raise RuntimeError("No BIDS EEG files were found under the provided root.")

    grouped = defaultdict(lambda: {
        "stats": defaultdict(lambda: defaultdict(_init_running_stats)),
        "n_recordings": 0,
    })

    artifact_config = ArtifactDetectionConfig(window_length_s=window_length)

    for eeg_file in eeg_files:
        if eeg_file.suffix.lower() not in {".edf", ".vhdr", ".set", ".bdf"}:
            continue

        raw, entities = _read_raw_bids_file(eeg_file, bids_root=bids_root)
        participant_id = f"sub-{entities.get('sub')}"
        participant = participants.get(participant_id)
        if not participant:
            continue

        try:
            age = int(float(participant["age"]))
        except (KeyError, ValueError, TypeError):
            continue

        task_eyes = _infer_eyes_condition(entities.get("task", ""))
        if eyes_condition != "both" and task_eyes != eyes_condition:
            continue

        age_bin = _age_bin_label(age, DEFAULT_AGE_BINS)
        if age_bin is None:
            continue

        raw = _normalize_bids_raw(raw, reference_mode=reference_mode)
        if set(raw.ch_names) != set(CLINICAL_19):
            continue

        clean_data, epochs, artifact_result = preprocess(
            raw,
            interactive=False,
            detect_artifacts_first=artifact_detection,
            artifact_config=artifact_config,
            artifact_visualize=False,
        )
        band_power = compute_band_power(clean_data, raw.info["sfreq"], raw.ch_names)

        age_label, age_min, age_max = age_bin
        stratum_key = f"reference={reference_mode}|eyes={task_eyes}|age={age_label}"
        grouped[stratum_key]["n_recordings"] += 1

        for band in FREQ_BANDS:
            for ch, value in band_power["relative"][band].items():
                _update_running_stats(grouped[stratum_key]["stats"][ch][band], value)

    norms = {"metadata": {
        "version": "3.0",
        "source": "Open-source BIDS/OpenNeuro norms built with this QEEG pipeline",
        "power_type": "relative",
        "reference_mode": reference_mode,
        "eyes_condition": eyes_condition,
        "format": "stratified_relative_power",
        "n_recordings": int(sum(group["n_recordings"] for group in grouped.values())),
        "strata": [],
    }, "norms": {}}

    for stratum_key, group in sorted(grouped.items()):
        reference_value = stratum_key.split("|")[0].split("=", 1)[1]
        eyes_value = stratum_key.split("|")[1].split("=", 1)[1]
        age_value = stratum_key.split("|")[2].split("=", 1)[1]
        age_min, age_max = [int(x) for x in age_value.split("-")]

        norms["metadata"]["strata"].append({
            "key": stratum_key,
            "reference_mode": reference_value,
            "eyes_condition": eyes_value,
            "age_min": age_min,
            "age_max": age_max,
            "n_recordings": group["n_recordings"],
        })
        norms["norms"][stratum_key] = {
            ch: {
                band: _finalize_running_stats(group["stats"][ch][band])
                for band in FREQ_BANDS
                if group["stats"][ch][band]["n"] > 0
            }
            for ch in CLINICAL_19
            if ch in group["stats"]
        }

    save_norms(norms, str(output_path))
    print(json.dumps(norms["metadata"], indent=2))


def main():
    args = _parse_args()
    build_open_norms(
        bids_root=Path(args.bids_root),
        output_path=Path(args.output),
        reference_mode=args.reference_mode,
        eyes_condition=args.eyes_condition,
        artifact_detection=args.artifact_detection,
        window_length=args.window_length,
    )


if __name__ == "__main__":
    main()
