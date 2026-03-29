"""
edf_loader.py
Load EDF/EDF+ files, remove non-EEG channels, normalize labels to a
standard 10-20 montage, detect the original recording reference, and
optionally rereference before downstream analysis.
"""

import re
from collections import Counter
from pathlib import Path

import mne
import numpy as np


LEGACY_CHANNEL_MAP = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
}

REFERENCE_SUFFIXES = {"A1", "A2", "M1", "M2", "LE", "RE"}

NON_EEG_PATTERNS = [
    r"^A[12]$",
    r"^M[12]$",
    r"^LE$",
    r"^RE$",
    r"^ECG",
    r"^EMG",
    r"^EOG",
    r"^EKG",
    r"^STI",
    r"^TRIGGER",
    r"^STATUS",
    r"^ANNOTATIONS",
]

STANDARD_MONTAGE_NAME = "standard_1020"

SIDE_MAP = {
    "LE": "L",
    "A1": "L",
    "M1": "L",
    "RE": "R",
    "A2": "R",
    "M2": "R",
}


def _is_non_eeg(channel_name: str) -> bool:
    name = channel_name.strip().upper()
    return any(re.match(pattern, name, re.IGNORECASE) for pattern in NON_EEG_PATTERNS)


def _extract_reference_suffix(name: str) -> str | None:
    """
    Extract channel reference suffix from labels like EEG FP1-LE or FP1-A1.
    """
    name = name.strip().upper()
    name = re.sub(r"^EEG[\s_\-]?", "", name)
    match = re.search(r"(?:\-|\s)?(A1|A2|M1|M2|LE|RE)$", name)
    return match.group(1) if match else None


def _clean_channel_name(name: str) -> str:
    """
    Normalize common EDF channel names to standard 10-20 labels.
    """
    name = name.strip().upper()
    name = re.sub(r"^EEG[\s_\-]?", "", name)
    name = re.sub(r"[\-]?(A[12]|M[12]|LE|RE)$", "", name)
    name = name.strip()
    return LEGACY_CHANNEL_MAP.get(name, name)


def _detect_recording_reference(raw: mne.io.Raw) -> str:
    refs = []
    for ch in raw.ch_names:
        cleaned = _clean_channel_name(ch)
        if _is_non_eeg(cleaned):
            continue
        ref = _extract_reference_suffix(ch)
        if ref in REFERENCE_SUFFIXES:
            refs.append(ref)

    if not refs:
        return "unknown"

    counts = Counter(refs)
    if len(counts) == 1:
        return next(iter(counts))

    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return "mixed:" + ",".join(f"{ref}={count}" for ref, count in ordered)


def _store_reference_metadata(raw: mne.io.Raw, recording_reference: str, analysis_reference: str):
    raw.info["description"] = (
        f"recording_reference={recording_reference};"
        f"analysis_reference={analysis_reference}"
    )


def _parse_reference_pair(name: str) -> tuple[str, str] | None:
    name = name.strip().upper()
    name = re.sub(r"^EEG[\s_\-]?", "", name)
    match = re.fullmatch(r"(A1|A2|M1|M2|LE|RE)[\-_](A1|A2|M1|M2|LE|RE)", name)
    if not match:
        return None
    return match.group(1), match.group(2)


def _find_linked_ears_bridge(raw: mne.io.Raw, recording_reference: str) -> tuple[str, float] | None:
    recording_side = SIDE_MAP.get(recording_reference)
    if recording_side is None:
        return None

    for ch in raw.ch_names:
        pair = _parse_reference_pair(ch)
        if not pair:
            continue
        left_side = SIDE_MAP.get(pair[0])
        right_side = SIDE_MAP.get(pair[1])
        if {left_side, right_side} != {"L", "R"}:
            continue
        sign = 1.0 if left_side == recording_side else -1.0
        return ch, sign

    return None


def _looks_like_cleaner_output(path: str) -> bool:
    return Path(path).stem.lower().endswith("_clean")


def load_edf(
    path: str,
    verbose: bool = True,
    rereference: str = "auto",
) -> mne.io.Raw:
    """
    Load an EDF/EDF+ file and return a clean MNE Raw object.

    Parameters
    ----------
    path : str
        EDF file path.
    verbose : bool
        Print loader details.
    rereference : str
        One of "auto", "average", "linked_ears", or "as_recorded".
    """
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    recording_reference = _detect_recording_reference(raw)
    analysis_reference = recording_reference
    linked_ears_bridge = _find_linked_ears_bridge(raw, recording_reference)
    linked_ears_bridge_data = None

    if linked_ears_bridge is not None:
        bridge_channel, _ = linked_ears_bridge
        linked_ears_bridge_data = raw.copy().pick([bridge_channel]).get_data()[0]

    if verbose:
        print(f"\n{'=' * 55}")
        print("  EDF LOADER REPORT")
        print(f"{'=' * 55}")
        print(f"  File            : {path}")
        print(f"  Sample rate     : {raw.info['sfreq']} Hz")
        print(f"  Duration        : {raw.times[-1]:.1f} s  ({raw.times[-1] / 60:.1f} min)")
        print(f"  Channels in file: {len(raw.ch_names)}")
        print(f"  Detected ref    : {recording_reference}")
        print("\n  RAW CHANNEL INVENTORY:")
        for i, ch in enumerate(raw.ch_names):
            print(f"    [{i:02d}] {ch}")

    rename_map = {ch: _clean_channel_name(ch) for ch in raw.ch_names}
    raw.rename_channels(rename_map)

    if verbose:
        changed = {k: v for k, v in rename_map.items() if k != v}
        if changed:
            print("\n  CHANNEL NAME NORMALIZATION:")
            for original, cleaned in changed.items():
                print(f"    '{original}' -> '{cleaned}'")
        else:
            print("\n  Channel names already clean; no renaming needed.")

    non_eeg = [ch for ch in raw.ch_names if _is_non_eeg(ch)]
    if non_eeg:
        if verbose:
            print("\n  NON-EEG CHANNELS EXCLUDED:")
            for ch in non_eeg:
                print(f"    x {ch}")
        raw.drop_channels(non_eeg)

    montage = mne.channels.make_standard_montage(STANDARD_MONTAGE_NAME)
    valid_10_20 = {ch.upper() for ch in montage.ch_names}

    matched = [ch for ch in raw.ch_names if ch.upper() in valid_10_20]
    unrecognized = [ch for ch in raw.ch_names if ch.upper() not in valid_10_20]

    if not matched:
        raise ValueError(
            f"\n[edf_loader] No standard 10-20 channels found after cleaning.\n"
            f"Remaining channels were: {raw.ch_names}\n"
            f"Check your EDF export settings in BrainMaster."
        )

    if unrecognized and verbose:
        print("\n  UNRECOGNIZED CHANNELS (excluded):")
        for ch in unrecognized:
            print(f"    x {ch}")

    raw.pick(matched)
    raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})
    raw.set_montage(montage, match_case=False, on_missing="ignore")

    if rereference == "linked_ears":
        if linked_ears_bridge is None or linked_ears_bridge_data is None:
            if _looks_like_cleaner_output(path):
                analysis_reference = "linked_ears"
                if verbose:
                    print(
                        "  [INFO] Cleaner output detected with no ear bridge channel."
                    )
                    print(
                        "  [INFO] Assuming the EDF is already in linked-ears reference."
                    )
            else:
                raise RuntimeError(
                    "linked_ears rereference was requested, but no left/right ear "
                    "difference channel was found in the EDF."
                )
        else:
            _, bridge_sign = linked_ears_bridge
            raw._data = raw._data + (bridge_sign * 0.5 * linked_ears_bridge_data[np.newaxis, :])
            analysis_reference = "linked_ears"
    elif rereference == "average":
        raw.set_eeg_reference(ref_channels="average", projection=False, verbose=False)
        analysis_reference = "average"
    elif rereference == "auto":
        if recording_reference in REFERENCE_SUFFIXES:
            raw.set_eeg_reference(ref_channels="average", projection=False, verbose=False)
            analysis_reference = "average"
    elif rereference == "as_recorded":
        analysis_reference = recording_reference
    else:
        raise ValueError(
            "rereference must be one of: auto, average, linked_ears, as_recorded"
        )

    _store_reference_metadata(raw, recording_reference, analysis_reference)

    clinical_19 = {
        "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
        "T7", "C3", "CZ", "C4", "T8",
        "P7", "P3", "PZ", "P4", "P8",
        "O1", "O2",
    }
    present = set(raw.ch_names)
    missing = clinical_19 - present

    if verbose:
        print(f"\n  FINAL CHANNEL SET ({len(raw.ch_names)} channels):")
        print(f"    {raw.ch_names}")
        print(f"  Analysis ref    : {analysis_reference}")
        if missing:
            print("\n  [WARN] MISSING FROM STANDARD 19-CH SET:")
            print(f"    {sorted(missing)}")
            print("    These channels will not appear in topomaps.")
        else:
            print("\n  [OK] Full standard 19-channel set present.")

        if analysis_reference == "linked_ears":
            print("\n  [OK] Linked-ears rereference applied for downstream qEEG analysis.")
        elif recording_reference in REFERENCE_SUFFIXES and analysis_reference == "average":
            print("\n  [OK] Average rereference applied for downstream qEEG analysis.")
        elif recording_reference in REFERENCE_SUFFIXES:
            print("\n  [WARN] Recording kept in its original reference.")

        print(f"{'=' * 55}\n")

    return raw


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python edf_loader.py path\\to\\your_file.edf")
        print("\nNo file provided. Pass an EDF path to inspect it.")
        raise SystemExit(0)

    edf_path = sys.argv[1]
    raw = load_edf(edf_path, verbose=True)
    print(f"Loader returned a clean Raw object with {len(raw.ch_names)} channels.")
    print("Ready for preprocessing.")
