"""
preprocessor.py
Bandpass + notch filtering, epoch creation, amplitude-based auto-rejection,
and interactive artifact review. Returns a clean array of good epochs
ready for band power extraction.
"""

import mne
import numpy as np

from artifact_detection import (
    ArtifactDetectionConfig,
    ArtifactDetectionResult,
    apply_artifact_annotations,
    detect_artifacts,
)


BANDPASS_LOW = 1.0
BANDPASS_HIGH = 40.0
NOTCH_FREQ = 60.0
EPOCH_LENGTH = 5.0
AMPLITUDE_THRESHOLD = 300e-6  # 300 uV fallback threshold for linked-ear recordings


def filter_raw(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Apply bandpass (1-40 Hz) and notch (60 Hz) filters.
    Returns a filtered copy; original is not modified.
    """
    print("\n[PREPROCESS] Filtering...")
    raw = raw.copy()
    raw.filter(
        l_freq=BANDPASS_LOW,
        h_freq=BANDPASS_HIGH,
        method="fir",
        verbose=False,
    )
    raw.notch_filter(freqs=NOTCH_FREQ, verbose=False)
    print(f"    [OK] Bandpass : {BANDPASS_LOW}-{BANDPASS_HIGH} Hz")
    print(f"    [OK] Notch    : {NOTCH_FREQ} Hz")
    return raw


def make_epochs(raw: mne.io.Raw, epoch_length: float = EPOCH_LENGTH) -> mne.Epochs:
    """
    Slice filtered continuous data into fixed-length epochs.
    Uses a synthetic regular event track; no trigger channel needed.
    """
    print(f"\n[PREPROCESS] Creating {epoch_length}s epochs...")

    sfreq = raw.info["sfreq"]
    step = int(epoch_length * sfreq)
    n_times = len(raw.times)
    n_epochs = n_times // step

    event_samples = np.arange(n_epochs) * step
    events = np.column_stack([
        event_samples,
        np.zeros(n_epochs, dtype=int),
        np.ones(n_epochs, dtype=int),
    ])

    epochs = mne.Epochs(
        raw,
        events,
        event_id=1,
        tmin=0.0,
        tmax=epoch_length - (1.0 / sfreq),
        baseline=None,
        preload=True,
        verbose=False,
    )

    print(f"    Total epochs created : {len(epochs)}")
    return epochs


def auto_reject_epochs(
    epochs: mne.Epochs,
    threshold: float = AMPLITUDE_THRESHOLD,
) -> tuple[mne.Epochs, list[int]]:
    """
    Automatic artifact rejection using autoreject when available.

    Falls back to a simple peak-to-peak amplitude threshold otherwise.
    Returns the cleaned epochs object and list of flagged indices.
    """
    try:
        from autoreject import AutoReject

        print("\n[PREPROCESS] Auto-rejection via Autoreject (data-adaptive)...")
        print("    Learning thresholds from data; this may take 30-60 seconds...")

        ar = AutoReject(
            n_interpolate=[1, 2, 4],
            random_state=42,
            verbose=False,
        )

        epochs_clean = ar.fit_transform(epochs)

        n_original = len(epochs.drop_log)
        n_kept = len(epochs_clean)
        n_dropped = n_original - n_kept
        pct = 100 * n_dropped / n_original if n_original else 0

        print(f"    Epochs kept    : {n_kept} / {n_original}")
        print(f"    Epochs dropped : {n_dropped} ({pct:.0f}%)")

        if pct > 60:
            print("    [WARN] Over 60% rejected; recording quality may be poor.")
            print("       Check electrode impedances and patient movement.")

        flagged = [i for i, log in enumerate(epochs_clean.drop_log) if len(log) > 0]
        return epochs_clean, flagged

    except ImportError:
        print(
            f"\n[PREPROCESS] Autoreject not installed; "
            f"falling back to {threshold * 1e6:.0f} uV threshold."
        )
        print("    Install with: pip install autoreject")
        return _threshold_reject(epochs, threshold_uv=threshold)


def _threshold_reject(
    epochs: mne.Epochs,
    threshold_uv: float = AMPLITUDE_THRESHOLD,
) -> tuple[mne.Epochs, list[int]]:
    """
    Fallback: simple peak-to-peak amplitude rejection.
    """
    print(f"\n[PREPROCESS] Auto-rejection at +/-{threshold_uv * 1e6:.0f} uV (fallback)...")

    data = epochs.get_data()
    p2p = data.max(axis=2) - data.min(axis=2)
    bad_mask = (p2p > threshold_uv).any(axis=1)
    flagged = np.where(bad_mask)[0].tolist()

    print(f"    Epochs flagged : {len(flagged)} / {len(epochs)}")

    if flagged:
        epochs.drop(flagged, verbose=False)

    return epochs, flagged


def review_epochs(
    epochs: mne.Epochs,
    flagged: list[int],
    epoch_length: float = EPOCH_LENGTH,
) -> list[int]:
    """
    Open MNE's interactive epoch browser.
    Auto-flagged epochs are pre-marked in the reporting only.
    """
    print("\n[PREPROCESS] Opening artifact reviewer...")
    print("    Close the window when done to continue.")

    if flagged:
        print(f"    {len(flagged)} epochs pre-flagged for review")

    epochs.plot(
        n_epochs=5,
        n_channels=19,
        scalings=dict(eeg=150e-6),
        title="Artifact Review - close window when done",
        show=True,
        block=True,
        picks="eeg",
    )

    n_dropped = sum(1 for log in epochs.drop_log if len(log) > 0)
    n_total = len(epochs.drop_log)

    print("\n    Review complete.")
    print(f"    Epochs dropped : {n_dropped} / {n_total}")
    return []


def finalize_epochs(
    epochs: mne.Epochs,
    bad_indices: list[int],
) -> np.ndarray:
    """
    Return clean epoch data after bad epochs have already been dropped.
    """
    clean_data = epochs.get_data()
    n_good = clean_data.shape[0]
    duration = n_good * EPOCH_LENGTH

    print("\n[PREPROCESS] Finalized.")
    print(f"    Good epochs : {n_good}")
    print(f"    Clean data  : {duration:.1f}s used for analysis")

    if n_good == 0:
        raise RuntimeError(
            "No clean epochs remain after artifact rejection. "
            "Consider raising the amplitude threshold or reviewing fewer epochs."
        )

    if n_good < 6:
        print(f"    [WARN] Only {n_good} clean epochs; results may be unreliable.")
        print("       Aim for at least 40 epochs (200s) for stable power estimates.")

    return clean_data


def preprocess(
    raw: mne.io.Raw,
    epoch_length: float = EPOCH_LENGTH,
    threshold: float = AMPLITUDE_THRESHOLD,
    interactive: bool = True,
    detect_artifacts_first: bool = False,
    artifact_config: ArtifactDetectionConfig | None = None,
    artifact_visualize: bool = False,
) -> tuple[np.ndarray, mne.Epochs, ArtifactDetectionResult | None]:
    """
    Full preprocessing pipeline:
      1. Filter
      2. Epoch
      3. Optional artifact detection on raw windows
      4. Auto-reject
      5. Interactive review (optional)
      6. Finalize and return clean data
    """
    filtered = filter_raw(raw)
    artifact_result = None

    if detect_artifacts_first:
        artifact_result = detect_artifacts(filtered, config=artifact_config)
        artifact_view = apply_artifact_annotations(filtered, artifact_result)
        filtered = artifact_view
        print("\n[PREPROCESS] Artifact detection summary:")
        print(f"    Bad windows  : {int(artifact_result.bad_windows.sum())} / {len(artifact_result.bad_windows)}")
        print(f"    Bad channels : {artifact_result.bad_channels or 'None'}")
        if artifact_visualize:
            try:
                artifact_view.plot(
                    start=0.0,
                    duration=20.0,
                    n_channels=min(len(artifact_result.ch_names), 20),
                    scalings="auto",
                    block=True,
                    title="Artifact Detection View",
                )
            except Exception as e:
                print(f"    [WARN] Artifact viewer failed to open: {e}")

    epochs = make_epochs(filtered, epoch_length)
    epochs, flagged = auto_reject_epochs(epochs, threshold=threshold)

    if interactive:
        try:
            bad_indices = review_epochs(epochs, flagged, epoch_length)
        except Exception as e:
            print(f"\n    [WARN] Reviewer window failed to open: {e}")
            print("    Falling back to auto-flagged epochs only.")
            bad_indices = flagged
    else:
        bad_indices = flagged
        print("\n[PREPROCESS] Non-interactive mode; using auto-flagged only.")

    clean_data = finalize_epochs(epochs, bad_indices)
    return clean_data, epochs, artifact_result
