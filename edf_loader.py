"""
edf_loader.py
Loads EDF/EDF+ files, inspects raw channel names, strips non-EEG channels
(reference ears, ECG, EMG, etc.), maps survivors to standard 10-20 montage,
and returns a clean MNE Raw object ready for preprocessing.
"""

import mne
import re

# ── Reference / non-EEG channel patterns to exclude ──────────────────────────
# Covers A1/A2, M1/M2, LE/RE, and common artifact channels.
# All matching is case-insensitive.
# Old 10-20 → New 10-20 name mapping (BrainMaster uses old convention)
LEGACY_CHANNEL_MAP = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
}
NON_EEG_PATTERNS = [
    r"^A[12]$",        # linked ears (A1, A2)
    r"^M[12]$",        # mastoids (M1, M2)
    r"^LE$", r"^RE$",  # left/right ear
    r"^ECG",           # electrocardiogram
    r"^EMG",           # electromyogram
    r"^EOG",           # electrooculogram
    r"^EKG",           # alternate ECG label
    r"^STI",           # stimulus channel
    r"^TRIGGER",       # trigger channel
    r"^STATUS",        # status channel
    r"^ANNOTATIONS",   # EDF+ annotation track
]

STANDARD_MONTAGE_NAME = "standard_1020"


def _is_non_eeg(channel_name: str) -> bool:
    """Return True if the channel matches any known non-EEG pattern."""
    name = channel_name.strip().upper()
    return any(re.match(p, name, re.IGNORECASE) for p in NON_EEG_PATTERNS)


def _clean_channel_name(name: str) -> str:
    """
    Normalize channel names to standard 10-20 format.
    Handles common BrainMaster export variants:
      'EEG FP1'  → 'FP1'
      'eeg_fp1'  → 'FP1'
      'FP1-A1'   → 'FP1'   (strips reference suffix)
      'FP1-LE'   → 'FP1'
    """
    name = name.strip().upper()
    name = re.sub(r"^EEG[\s_\-]?", "", name)       # strip EEG prefix
    name = re.sub(r"[\-]?(A[12]|M[12]|LE|RE)$", "", name)  # strip ref suffix
    name = name.strip()
    name = LEGACY_CHANNEL_MAP.get(name, name)   # remap legacy names
    return name


def load_edf(path: str, verbose: bool = True) -> mne.io.Raw:
    """
    Load an EDF/EDF+ file and return a clean MNE Raw object.

    Steps:
      1. Read raw EDF (all channels, preloaded into memory)
      2. Print a full channel inventory so you can see exactly what came in
      3. Normalize channel names (strip prefixes/suffixes)
      4. Exclude non-EEG channels (ears, ECG, EMG, triggers, etc.)
      5. Retain only channels present in the standard 10-20 montage
      6. Set channel types to EEG and apply the standard montage
      7. Report any expected 10-20 channels that are missing

    Parameters
    ----------
    path    : full path to the EDF/EDF+ file
    verbose : if True, print a full channel inventory report

    Returns
    -------
    raw : mne.io.Raw
        Preprocessed-ready Raw object with only clean 10-20 EEG channels.
    """

    # ── 1. Read raw file ──────────────────────────────────────────────────────
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)

    if verbose:
        print(f"\n{'='*55}")
        print(f"  EDF LOADER REPORT")
        print(f"{'='*55}")
        print(f"  File            : {path}")
        print(f"  Sample rate     : {raw.info['sfreq']} Hz")
        print(f"  Duration        : {raw.times[-1]:.1f} s  "
              f"({raw.times[-1]/60:.1f} min)")
        print(f"  Channels in file: {len(raw.ch_names)}")
        print(f"\n  RAW CHANNEL INVENTORY:")
        for i, ch in enumerate(raw.ch_names):
            print(f"    [{i:02d}] {ch}")

    # ── 2. Normalize channel names ────────────────────────────────────────────
    rename_map = {ch: _clean_channel_name(ch) for ch in raw.ch_names}
    raw.rename_channels(rename_map)

    if verbose:
        changed = {k: v for k, v in rename_map.items() if k != v}
        if changed:
            print(f"\n  CHANNEL NAME NORMALIZATION:")
            for original, cleaned in changed.items():
                print(f"    '{original}' → '{cleaned}'")
        else:
            print(f"\n  Channel names already clean — no renaming needed.")

    # ── 3. Identify and drop non-EEG channels ────────────────────────────────
    non_eeg = [ch for ch in raw.ch_names if _is_non_eeg(ch)]
    if non_eeg:
        if verbose:
            print(f"\n  NON-EEG CHANNELS EXCLUDED:")
            for ch in non_eeg:
                print(f"    ✗ {ch}")
        raw.drop_channels(non_eeg)

    # ── 4. Keep only valid 10-20 channels ────────────────────────────────────
    montage = mne.channels.make_standard_montage(STANDARD_MONTAGE_NAME)
    valid_10_20 = {ch.upper() for ch in montage.ch_names}

    matched     = [ch for ch in raw.ch_names if ch.upper() in valid_10_20]
    unrecognized = [ch for ch in raw.ch_names if ch.upper() not in valid_10_20]

    if not matched:
        raise ValueError(
            f"\n[edf_loader] No standard 10-20 channels found after cleaning.\n"
            f"Remaining channels were: {raw.ch_names}\n"
            f"Check your EDF export settings in BrainMaster."
        )

    if unrecognized and verbose:
        print(f"\n  UNRECOGNIZED CHANNELS (not in 10-20, also excluded):")
        for ch in unrecognized:
            print(f"    ✗ {ch}")

    raw.pick_channels(matched)

    # ── 5. Set channel types and montage ─────────────────────────────────────
    raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})
    raw.set_montage(montage, match_case=False, on_missing="ignore")

    # ── 6. Report missing 10-20 channels ─────────────────────────────────────
    # Standard 19-channel clinical set
    clinical_19 = {
        "FP1","FP2","F7","F3","FZ","F4","F8",
        "T7","C3","CZ","C4","T8",
        "P7","P3","PZ","P4","P8",
        "O1","O2"
    }
    present  = set(raw.ch_names)
    missing  = clinical_19 - present

    if verbose:
        print(f"\n  FINAL CHANNEL SET ({len(raw.ch_names)} channels):")
        print(f"    {raw.ch_names}")
        if missing:
            print(f"\n  ⚠️  MISSING FROM STANDARD 19-CH SET:")
            print(f"    {sorted(missing)}")
            print(f"    These channels won't appear in topomaps.")
        else:
            print(f"\n  ✓ Full standard 19-channel set present.")
        print(f"{'='*55}\n")

    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Quick test — run this file directly to inspect any EDF
# Usage: python edf_loader.py path\to\your_file.edf
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python edf_loader.py path\\to\\your_file.edf")
        print("\nNo file provided — printing what the loader would do with a real file.")
        print("Pass your EDF path as an argument to inspect it.")
        sys.exit(0)

    edf_path = sys.argv[1]
    raw = load_edf(edf_path, verbose=True)
    print(f"Loader returned a clean Raw object with {len(raw.ch_names)} channels.")
    print("Ready for preprocessing.")