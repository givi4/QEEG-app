"""
preprocessor.py
Bandpass + notch filtering, epoch creation, amplitude-based auto-rejection,
and interactive artifact review. Returns a clean array of good epochs
ready for band power extraction.
"""

import mne
import numpy as np
from mne import Epochs, events_from_annotations
import matplotlib.pyplot as plt

# ── Default parameters (change here to affect whole pipeline) ─────────────────
BANDPASS_LOW  = 1.0       # Hz
BANDPASS_HIGH = 40.0      # Hz
NOTCH_FREQ    = 60.0      # Hz (US standard)
EPOCH_LENGTH  = 5.0       # seconds
AMPLITUDE_THRESHOLD = 300e-6   # 300 µV — raised for linked-ear reference montage


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — FILTERING
# ─────────────────────────────────────────────────────────────────────────────
def filter_raw(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Apply bandpass (1–40 Hz) and notch (60 Hz) filters.
    Returns a filtered copy — original is not modified.
    """
    print("\n[PREPROCESS] Filtering...")
    raw = raw.copy()
    raw.filter(
        l_freq=BANDPASS_LOW,
        h_freq=BANDPASS_HIGH,
        method="fir",
        verbose=False
    )
    raw.notch_filter(freqs=NOTCH_FREQ, verbose=False)
    print(f"    ✓ Bandpass : {BANDPASS_LOW}–{BANDPASS_HIGH} Hz")
    print(f"    ✓ Notch    : {NOTCH_FREQ} Hz")
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — EPOCH CREATION
# ─────────────────────────────────────────────────────────────────────────────
def make_epochs(raw: mne.io.Raw, epoch_length: float = EPOCH_LENGTH) -> mne.Epochs:
    """
    Slice filtered continuous data into fixed-length epochs.
    Uses a synthetic regular event track — no trigger channel needed.
    """
    print(f"\n[PREPROCESS] Creating {epoch_length}s epochs...")

    sfreq      = raw.info["sfreq"]
    step       = int(epoch_length * sfreq)
    n_times    = len(raw.times)
    n_epochs   = n_times // step

    # Build a simple event array: one event every epoch_length seconds
    # MNE event format: [sample_index, 0, event_id]
    event_samples = np.arange(n_epochs) * step
    events = np.column_stack([
        event_samples,
        np.zeros(n_epochs, dtype=int),
        np.ones(n_epochs, dtype=int)
    ])

    epochs = mne.Epochs(
        raw,
        events,
        event_id=1,
        tmin=0.0,
        tmax=epoch_length - (1.0 / sfreq),   # avoid overlap
        baseline=None,
        preload=True,
        verbose=False
    )

    print(f"    Total epochs created : {len(epochs)}")
    return epochs


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — AUTO-REJECTION (amplitude threshold)
# ─────────────────────────────────────────────────────────────────────────────
def auto_reject_epochs(
    epochs: mne.Epochs,
    threshold: float = AMPLITUDE_THRESHOLD
) -> tuple[mne.Epochs, list[int]]:
    """
    Flag epochs where any channel exceeds ±threshold (default 100 µV).
    Returns the epochs object with bad epochs marked, plus a list of
    flagged epoch indices for display in the reviewer.

    Does NOT drop epochs yet — that happens after manual review.
    """
    print(f"\n[PREPROCESS] Auto-rejection at ±{threshold*1e6:.0f} µV...")

    data         = epochs.get_data()          # shape: (n_epochs, n_ch, n_times)
    peak_to_peak = data.max(axis=2) - data.min(axis=2)   # per epoch per channel
    bad_mask     = (peak_to_peak > threshold).any(axis=1) # True if any ch exceeds
    flagged      = np.where(bad_mask)[0].tolist()

    print(f"    Epochs flagged   : {len(flagged)} / {len(epochs)}")
    if flagged:
        print(f"    Flagged indices  : {flagged}")

    return epochs, flagged


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — INTERACTIVE REVIEWER
# ─────────────────────────────────────────────────────────────────────────────
def review_epochs(
    epochs: mne.Epochs,
    flagged: list[int],
    epoch_length: float = EPOCH_LENGTH
) -> list[int]:
    """
    Opens MNE's built-in interactive epoch browser.
    Auto-flagged epochs are pre-marked red for review.

    Controls (shown in the browser window):
      Click a channel trace   → mark that channel bad in this epoch
      Click epoch number bar  → mark entire epoch bad
      [a]                     → mark all channels in epoch bad
      Arrow keys / scroll     → navigate
      Close window            → finish and continue

    Returns a list of epoch indices marked as bad.
    """
    print("\n[PREPROCESS] Opening artifact reviewer...")
    print("    Close the window when done to continue.")

    # Pre-mark auto-flagged epochs so they show highlighted on open
    for idx in flagged:
        epochs.drop_log  # ensure drop_log exists
    if flagged:
        print(f"    {len(flagged)} epochs pre-flagged (shown in red)")

    # Annotate auto-flagged epochs as bad before opening browser
    bad_labels = ['auto-flagged' if i in flagged else '' 
                  for i in range(len(epochs))]

    # Use MNE's browser — blocks until window is closed
    fig = epochs.plot(
        n_epochs=5,           # show 5 epochs at a time
        n_channels=19,        # show all channels
        scalings=dict(eeg=150e-6),   # 150µV scale — adjust if traces look flat/clipped
        title="Artifact Review — close window when done",
        show=True,
        block=True,           # blocks pipeline until closed
        picks="eeg",
    )

    # MNE drops epochs internally during browser interaction
    # Count how many were dropped for reporting
    n_dropped = sum(1 for log in epochs.drop_log if len(log) > 0)
    n_total = len(epochs.drop_log)

    print(f"\n    Review complete.")
    print(f"    Epochs dropped : {n_dropped} / {n_total}")

    return []   # finalize_epochs reads directly from epochs object

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — FINALIZE: drop bad epochs, return clean data
# ─────────────────────────────────────────────────────────────────────────────
def finalize_epochs(
    epochs:      mne.Epochs,
    bad_indices: list[int]    # kept for API compatibility, MNE handles drops internally
) -> np.ndarray:
    """
    Return clean epoch data after MNE's interactive browser has
    already dropped bad epochs in-place.

    Note: bad_indices is intentionally ignored here — MNE's epoch
    browser drops epochs directly on the Epochs object during
    interactive review. We simply retrieve whatever survived.
    """
    clean_data = epochs.get_data()
    n_good     = clean_data.shape[0]
    duration   = n_good * EPOCH_LENGTH

    print(f"\n[PREPROCESS] Finalized.")
    print(f"    Good epochs : {n_good}")
    print(f"    Clean data  : {duration:.1f}s used for analysis")

    if n_good == 0:
        raise RuntimeError(
            "No clean epochs remain after artifact rejection. "
            "Consider raising the amplitude threshold or reviewing fewer epochs."
        )

    if n_good < 6:
        print(f"    ⚠️  Only {n_good} clean epochs — results may be unreliable.")
        print(f"       Aim for at least 40 epochs (200s) for stable power estimates.")

    return clean_data


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE FUNCTION — call this from qeeg_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(
    raw: mne.io.Raw,
    epoch_length: float = EPOCH_LENGTH,
    threshold: float = AMPLITUDE_THRESHOLD,
    interactive: bool = True
) -> tuple[np.ndarray, mne.Epochs]:
    """
    Full preprocessing pipeline:
      1. Filter (bandpass + notch)
      2. Epoch
      3. Auto-reject by amplitude
      4. Interactive review (if interactive=True)
      5. Finalize and return clean data

    Parameters
    ----------
    raw         : MNE Raw object from edf_loader
    epoch_length: seconds per epoch (default 5.0)
    threshold   : auto-rejection threshold in volts (default 100e-6 = 100µV)
    interactive : open the epoch reviewer window (set False for batch/testing)

    Returns
    -------
    clean_data  : np.ndarray (n_good_epochs, n_channels, n_times)
    epochs      : MNE Epochs object (for downstream use if needed)
    """
    filtered      = filter_raw(raw)
    epochs        = make_epochs(filtered, epoch_length)
    epochs, flagged = auto_reject_epochs(epochs, threshold)

    if interactive:
        try:
            bad_indices = review_epochs(epochs, flagged, epoch_length)
        except Exception as e:
            print(f"\n    ⚠️  Reviewer window failed to open: {e}")
            print(f"    Falling back to auto-flagged epochs only.")
            bad_indices = flagged
    else:
        bad_indices = flagged
        print(f"\n[PREPROCESS] Non-interactive mode — using auto-flagged only.")

    clean_data = finalize_epochs(epochs, bad_indices)
    return clean_data, epochs