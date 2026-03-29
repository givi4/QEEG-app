"""
eeg_cleaner.py
Standalone EEG cleaning app with Streamlit UI.
Loads raw EDF, filters, runs Autoreject, applies ICA with semi-automatic
component selection, and exports a clean EDF file.

Run with:
    streamlit run eeg_cleaner.py
"""

import streamlit as st
import mne
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import os
import io
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if not (PROJECT_ROOT / "edf_loader.py").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from edf_loader import load_edf

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EEG Cleaner",
    page_icon="🧠",
    layout="wide"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stButton>button {
        background-color: #00d4aa;
        color: #0f1117;
        font-weight: bold;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
    }
    .stButton>button:hover { background-color: #00b894; }
    .status-box {
        background-color: #1a1a2e;
        border-left: 4px solid #00d4aa;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        color: #e0e0e0;
        font-family: monospace;
        font-size: 0.85rem;
        white-space: pre-wrap;
    }
    .warn-box {
        background-color: #1a1a2e;
        border-left: 4px solid #ff8800;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        color: #e0e0e0;
        font-size: 0.85rem;
    }
    h1, h2, h3 { color: #e0e0e0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "stage":           "upload",   # upload → filter → autoreject → ica → export
        "raw":             None,
        "raw_filtered":    None,
        "epochs":          None,
        "epochs_clean":    None,
        "raw_clean":       None,
        "ica":             None,
        "ica_n_components": None,
        "ica_auto_excludes": None,
        "eog_scores":      None,
        "ica_excludes":    [],
        "edf_filename":    None,
        "uploaded_signature": None,
        "log":             [],
        "n_original":      0.0,
        "n_kept":          0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


def log(msg: str):
    st.session_state.log.append(msg)


def reset_keys(updates: dict):
    for key, value in updates.items():
        st.session_state[key] = value.copy() if isinstance(value, list) else value


def reset_downstream_state(from_stage: str):
    resets = {
        "upload": {
            "raw_filtered": None,
            "epochs": None,
            "epochs_clean": None,
            "raw_clean": None,
            "ica": None,
            "ica_n_components": None,
            "ica_auto_excludes": None,
            "eog_scores": None,
            "ica_excludes": [],
            "n_kept": 0,
        },
        "filter": {
            "epochs": None,
            "epochs_clean": None,
            "raw_clean": None,
            "ica": None,
            "ica_n_components": None,
            "ica_auto_excludes": None,
            "eog_scores": None,
            "ica_excludes": [],
            "n_kept": 0,
        },
        "autoreject": {
            "raw_clean": None,
            "ica": None,
            "ica_n_components": None,
            "ica_auto_excludes": None,
            "eog_scores": None,
            "ica_excludes": [],
        },
        "ica": {
            "raw_clean": None,
            "ica": None,
            "ica_n_components": None,
            "ica_auto_excludes": None,
            "eog_scores": None,
            "ica_excludes": [],
        },
    }
    reset_keys(resets[from_stage])


def raw_duration_seconds(raw):
    return raw.n_times / raw.info["sfreq"]


def find_channels_case_insensitive(ch_names, desired_names):
    lookup = {name.upper(): name for name in ch_names}
    return [lookup[name.upper()] for name in desired_names if name.upper() in lookup]


def prepare_export_raw_for_pipeline(raw):
    export_raw = raw.copy()
    rename_map = {ch: f"{ch}-A1" for ch in export_raw.ch_names}
    export_raw.rename_channels(rename_map)

    bridge_data = np.zeros((1, export_raw.n_times), dtype=export_raw.get_data().dtype)
    bridge_info = mne.create_info(["A1-A2"], export_raw.info["sfreq"], ch_types=["misc"])
    bridge_raw = mne.io.RawArray(bridge_data, bridge_info, verbose=False)

    export_raw.add_channels([bridge_raw], force_update_info=True)
    return export_raw


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — render log
# ─────────────────────────────────────────────────────────────────────────────
def render_log():
    if st.session_state.log:
        log_text = "\n".join(st.session_state.log)
        st.markdown(f'<div class="status-box">{log_text}</div>',
                    unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — figure to streamlit image
# ─────────────────────────────────────────────────────────────────────────────
def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf


def render_psd_comparison(raw, raw_f):
    st.markdown("#### Power Spectral Density - Before vs After")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4),
                             facecolor="#0f1117")
    for ax in axes:
        ax.set_facecolor("#0f1117")

    raw.compute_psd(fmax=80).plot(
        axes=axes[0], show=False, picks="eeg"
    )
    axes[0].set_title("Before filtering",
                      color="white", fontsize=11)

    raw_f.compute_psd(fmax=80).plot(
        axes=axes[1], show=False, picks="eeg"
    )
    axes[1].set_title("After filtering",
                      color="white", fontsize=11)

    for ax in axes:
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")

    plt.tight_layout()
    st.image(fig_to_image(fig), use_container_width=True)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
def stage_upload():
    st.title("🧠 EEG Cleaner")
    st.markdown("#### Artifact removal pipeline — Filter → Autoreject → ICA → Export")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload raw EDF file",
        type=["edf", "EDF"],
        help="EDF or EDF+ files from BrainMaster Discovery"
    )

    raw = st.session_state.raw

    if uploaded is not None:
        upload_signature = (uploaded.name, uploaded.size)

        # Save to temp file so MNE can read it
        with tempfile.NamedTemporaryFile(
            suffix=".edf", delete=False
        ) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        with st.spinner("Loading EDF..."):
            try:
                raw = load_edf(
                    tmp_path,
                    verbose=False,
                    rereference="linked_ears",
                )
                reset_downstream_state("upload")
                st.session_state.log = []
                st.session_state.raw = raw
                st.session_state.edf_filename = uploaded.name
                st.session_state.uploaded_signature = upload_signature
                st.session_state.n_original = raw_duration_seconds(raw)

                log(f"✓ Loaded: {uploaded.name}")
                log(f"  Channels  : {len(raw.ch_names)}")
                log(f"  Duration  : {raw_duration_seconds(raw):.1f}s")
                log(f"  Sample rate: {raw.info['sfreq']} Hz")

                os.unlink(tmp_path)
            except Exception as e:
                st.session_state.raw = None
                st.session_state.edf_filename = None
                st.session_state.uploaded_signature = None
                st.error(f"Failed to load EDF: {e}")
                os.unlink(tmp_path)
                return

        raw = st.session_state.raw
        render_log()

        # Channel summary
        col1, col2, col3 = st.columns(3)
        col1.metric("Channels", len(raw.ch_names))
        col2.metric("Duration", f"{raw_duration_seconds(raw)/60:.1f} min")
        col3.metric("Sample Rate", f"{int(raw.info['sfreq'])} Hz")

        st.markdown("**Channels loaded:**")
        st.code("  ".join(raw.ch_names))

        if st.button("▶  Continue to Filtering"):
            st.session_state.stage = "filter"
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — FILTER
# ─────────────────────────────────────────────────────────────────────────────
def stage_filter():
    st.title("🧠 EEG Cleaner — Step 1: Filter")
    st.markdown("---")

    raw = st.session_state.raw

    if raw is None:
        st.error("No EDF is loaded. Return to Upload and select a BrainMaster EDF.")
        return

    col1, col2 = st.columns(2)
    with col1:
        bp_low  = st.number_input("Bandpass low (Hz)",  value=1.0,  step=0.5)
        bp_high = st.number_input("Bandpass high (Hz)", value=40.0, step=1.0)
    with col2:
        notch   = st.number_input("Notch filter (Hz)",  value=60.0, step=10.0)

    st.markdown(
        '<div class="warn-box">⚠️  Filtering may take 10–20 seconds '
        'on a 5-minute recording.</div>',
        unsafe_allow_html=True
    )

    if st.button("▶  Apply Filters"):
        nyquist = raw.info["sfreq"] / 2.0
        if bp_low >= bp_high:
            st.error("Bandpass low must be smaller than bandpass high.")
            return
        if bp_high >= nyquist:
            st.error(f"Bandpass high must be below the Nyquist frequency ({nyquist:.1f} Hz).")
            return

        with st.spinner("Filtering..."):
            raw_f = raw.copy()
            raw_f.filter(
                l_freq=bp_low, h_freq=bp_high,
                method="fir", verbose=False
            )
            raw_f.notch_filter(freqs=notch, verbose=False)
            reset_downstream_state("filter")
            st.session_state.raw_filtered = raw_f

            log(f"✓ Bandpass  : {bp_low}–{bp_high} Hz")
            log(f"✓ Notch     : {notch} Hz")

        render_log()

        # Plot PSD before/after
        st.markdown("#### Power Spectral Density — Before vs After")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4),
                                 facecolor="#0f1117")
        for ax in axes:
            ax.set_facecolor("#0f1117")

        raw.compute_psd(fmax=80).plot(
            axes=axes[0], show=False, picks="eeg"
        )
        axes[0].set_title("Before filtering",
                           color="white", fontsize=11)

        raw_f.compute_psd(fmax=80).plot(
            axes=axes[1], show=False, picks="eeg"
        )
        axes[1].set_title("After filtering",
                           color="white", fontsize=11)

        for ax in axes:
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")

        plt.tight_layout()
        st.image(fig_to_image(fig), use_container_width=True)
        plt.close(fig)

        if st.button("▶  Continue to Autoreject"):
            st.session_state.stage = "autoreject"
            st.rerun()

    raw_f = st.session_state.raw_filtered
    if raw_f is not None:
        render_log()
        render_psd_comparison(raw, raw_f)

        if st.button("Next: Autoreject"):
            st.session_state.stage = "autoreject"
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — AUTOREJECT
# ─────────────────────────────────────────────────────────────────────────────
def stage_autoreject():
    st.title("🧠 EEG Cleaner — Step 2: Autoreject")
    st.markdown("---")

    raw_f = st.session_state.raw_filtered

    if raw_f is None:
        st.error("Filtering has not been run yet.")
        return

    epoch_len = st.slider(
        "Epoch length (seconds)", min_value=2, max_value=10, value=5
    )

    st.markdown(
        '<div class="warn-box">⚠️  Autoreject learns thresholds from your data. '
        'This typically takes 30–90 seconds.</div>',
        unsafe_allow_html=True
    )

    if st.button("▶  Run Autoreject"):
        with st.spinner("Creating epochs..."):
            sfreq  = raw_f.info["sfreq"]
            step   = int(epoch_len * sfreq)
            n_times = len(raw_f.times)
            n_epochs = n_times // step

            if n_epochs < 2:
                st.error(
                    f"This recording only contains {n_times / sfreq:.1f}s of filtered data, "
                    f"which is not enough for {epoch_len}s Autoreject epochs."
                )
                return

            event_samples = np.arange(n_epochs) * step
            events = np.column_stack([
                event_samples,
                np.zeros(n_epochs, dtype=int),
                np.ones(n_epochs, dtype=int)
            ])

            epochs = mne.Epochs(
                raw_f, events, event_id=1,
                tmin=0.0,
                tmax=epoch_len - (1.0 / sfreq),
                baseline=None, preload=True, verbose=False
            )
            log(f"✓ Epochs created : {len(epochs)}")

        with st.spinner("Running Autoreject — learning thresholds..."):
            try:
                from autoreject import AutoReject
                ar = AutoReject(
                    n_interpolate=[1, 2, 4],
                    random_state=42,
                    verbose=False
                )
                epochs_clean = ar.fit_transform(epochs)

                n_orig    = len(epochs.drop_log)
                n_kept    = len(epochs_clean)
                n_dropped = n_orig - n_kept
                pct       = 100 * n_dropped / n_orig

                reset_downstream_state("autoreject")
                st.session_state.epochs       = epochs
                st.session_state.epochs_clean = epochs_clean
                st.session_state.n_kept       = n_kept

                log(f"✓ Autoreject complete")
                log(f"  Epochs kept    : {n_kept} / {n_orig}")
                log(f"  Epochs dropped : {n_dropped} ({pct:.0f}%)")
                log(f"  Clean data     : {n_kept * epoch_len:.0f}s")

                if pct > 60:
                    st.warning(
                        f"⚠️  {pct:.0f}% of epochs rejected. "
                        "Recording quality may be poor — "
                        "check electrode impedances."
                    )

                st.session_state.stage = "ica"
                st.rerun()

            except ImportError:
                st.error(
                    "Autoreject not installed. "
                    "Run: pip install autoreject"
                )
                return
            except Exception as e:
                st.error(f"Autoreject failed: {e}")
                return

        render_log()

        col1, col2, col3 = st.columns(3)
        col1.metric("Original epochs", n_orig)
        col2.metric("Kept", n_kept)
        col3.metric("Rejected", f"{n_dropped} ({pct:.0f}%)")

        if st.button("▶  Continue to ICA"):
            st.session_state.stage = "ica"
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — ICA


def stage_ica():
    st.title("🧠 EEG Cleaner — Step 3: ICA")
    st.markdown("---")

    epochs_clean = st.session_state.epochs_clean

    if epochs_clean is None:
        st.error("Autoreject has not been run yet.")
        return

    n_eeg_channels = len(mne.pick_types(epochs_clean.info, eeg=True, exclude="bads"))
    if n_eeg_channels < 2:
        st.error("ICA requires at least two EEG channels.")
        return

    default_components = min(15, n_eeg_channels)
    n_components = st.slider(
        "Number of ICA components",
        min_value=2, max_value=n_eeg_channels, value=default_components,
        help="Typically 15 for 19-channel BrainMaster data. More = finer decomposition."
    )

    st.markdown(
        '<div class="warn-box">⚠️  ICA fitting takes 20–40 seconds. '
        'After fitting, you will see detected artifact components '
        'and confirm which to remove.</div>',
        unsafe_allow_html=True
    )

    # ── Fit ICA ───────────────────────────────────────────────────────────────
    if st.button("▶  Fit ICA") and st.session_state.ica is None:
        with st.spinner("Fitting ICA..."):
            ica = mne.preprocessing.ICA(
                n_components=n_components,
                method="fastica",
                random_state=42,
                max_iter="auto",
                verbose=False
            )
            ica.fit(epochs_clean, verbose=False)
            st.session_state.ica = ica
            log(f"✓ ICA fitted : {n_components} components")

    if st.session_state.ica is None:
        render_log()
        return

    ica = st.session_state.ica

    # ── Auto-detect artifact components ───────────────────────────────────────
    st.markdown("#### Auto-detected artifact components")
    st.markdown(
        "ICA used FP1 and FP2 as proxy EOG channels to detect "
        "eye blink and movement components."
    )

    if st.session_state.ica_auto_excludes is None:
        with st.spinner("Detecting artifact components..."):
            # Use FP1/FP2 as EOG proxy — standard for no dedicated EOG channel
            eog_indices, eog_scores = ica.find_bads_eog(
                epochs_clean,
                ch_name=find_channels_case_insensitive(
                    epochs_clean.ch_names, ["FP1", "FP2"]
                ),
                verbose=False
            )
            st.session_state.ica_auto_excludes = eog_indices
            st.session_state.eog_scores        = eog_scores
            log(f"✓ Auto-detected EOG components: {eog_indices}")

    auto_excludes = st.session_state.ica_auto_excludes
    eog_scores    = st.session_state.eog_scores

    if auto_excludes:
        st.markdown(
            f'<div class="status-box">Auto-detected {len(auto_excludes)} '
            f'likely artifact component(s): '
            f'{", ".join([f"ICA{i:03d}" for i in auto_excludes])}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="warn-box">No artifact components '
            'auto-detected. Review topomaps below and select '
            'manually if needed.</div>',
            unsafe_allow_html=True
        )

    # ── Plot component topomaps ───────────────────────────────────────────────
    st.markdown("#### Component topomaps — review before confirming removal")

    n_comp   = ica.n_components_
    n_cols   = 5
    n_rows   = int(np.ceil(n_comp / n_cols))
    fig_topo, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.5, n_rows * 2.8),
        facecolor="#0f1117"
    )
    axes_flat = axes.flatten()

    ica.plot_components(
        picks=range(n_comp),
        axes=axes_flat[:n_comp],
        show=False,
        colorbar=False,
    )

    # Highlight auto-detected components
    for idx in auto_excludes:
        if idx < len(axes_flat):
            axes_flat[idx].set_facecolor("#3a0000")
            axes_flat[idx].set_title(
                f"ICA{idx:03d} ⚠️",
                color="#ff4444", fontsize=9
            )

    # Hide unused axes
    for ax in axes_flat[n_comp:]:
        ax.set_visible(False)

    for ax in axes_flat[:n_comp]:
        ax.set_facecolor("#0f1117")

    fig_topo.patch.set_facecolor("#0f1117")
    plt.tight_layout(pad=0.5)
    st.image(fig_to_image(fig_topo), use_container_width=True)
    plt.close(fig_topo)

    # ── Component time series for auto-detected ───────────────────────────────
    if auto_excludes:
        st.markdown("#### Time series of auto-detected components")
        fig_ts, axes_ts = plt.subplots(
            len(auto_excludes), 1,
            figsize=(14, 2.5 * len(auto_excludes)),
            facecolor="#0f1117"
        )
        if len(auto_excludes) == 1:
            axes_ts = [axes_ts]

        sources = ica.get_sources(epochs_clean)
        src_data = sources.get_data().mean(axis=0)  # mean across epochs

        for ax, idx in zip(axes_ts, auto_excludes):
            ax.set_facecolor("#0f1117")
            ax.plot(src_data[idx], color="#ff4444",
                    linewidth=0.8, alpha=0.9)
            ax.set_title(f"ICA{idx:03d}",
                         color="white", fontsize=10)
            ax.tick_params(colors="#aaaaaa")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444444")

        fig_ts.patch.set_facecolor("#0f1117")
        plt.tight_layout()
        st.image(fig_to_image(fig_ts), use_container_width=True)
        plt.close(fig_ts)

    # ── Component selection ───────────────────────────────────────────────────
    st.markdown("#### Select components to remove")
    st.markdown(
        "Auto-detected components are pre-selected. "
        "Add or remove any based on your review of the topomaps above."
    )

    all_components = [f"ICA{i:03d}" for i in range(n_comp)]
    default_selected = [f"ICA{i:03d}" for i in auto_excludes]

    selected = st.multiselect(
        "Components to exclude (remove from signal):",
        options=all_components,
        default=default_selected,
        help="Select components that look like artifacts — "
             "eye blinks show as frontal dipoles, "
             "muscle as high-frequency peripheral components."
    )

    exclude_indices = [int(s.replace("ICA", "")) for s in selected]

    if selected:
        st.markdown(
            f'<div class="status-box">Will remove {len(selected)} '
            f'component(s): {", ".join(selected)}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="warn-box">No components selected — '
            'ICA will not change the signal.</div>',
            unsafe_allow_html=True
        )

    # ── Apply ICA ─────────────────────────────────────────────────────────────
    if st.button("✓  Confirm and Apply ICA"):
        with st.spinner("Applying ICA and reconstructing signal..."):
            ica.exclude = exclude_indices

            # Apply ICA to clean epochs
            epochs_ica = epochs_clean.copy()
            ica.apply(epochs_ica, verbose=False)

            # Reconstruct continuous raw from clean epochs
            epoch_data = epochs_ica.get_data()
            continuous_data = np.concatenate(epoch_data, axis=1)
            raw_clean = mne.io.RawArray(
                continuous_data,
                epochs_ica.info.copy(),
                verbose=False
            )

            st.session_state.raw_clean    = raw_clean
            st.session_state.ica_excludes = exclude_indices

            n_removed = len(exclude_indices)
            n_epochs  = len(epochs_ica)
            duration  = raw_duration_seconds(raw_clean)

            log(f"✓ ICA applied : {n_removed} component(s) removed")
            log(f"✓ Reconstructed : {duration:.1f}s clean signal")
            log(f"  ({n_epochs} epochs × "
                f"{epochs_ica.times[-1]:.1f}s)")

        st.session_state.stage = "export"
        st.rerun()





# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — EXPORT
# ─────────────────────────────────────────────────────────────────────────────
def stage_export():
    st.title("🧠 EEG Cleaner — Step 4: Export")
    st.markdown("---")

    raw_clean     = st.session_state.raw_clean
    orig_filename = st.session_state.edf_filename or "recording"
    stem          = Path(orig_filename).stem
    out_filename  = f"{stem}_clean.edf"

    # ── Summary ───────────────────────────────────────────────────────────────
    st.markdown("#### Cleaning summary")

    orig_dur  = st.session_state.n_original
    clean_dur = raw_duration_seconds(raw_clean)
    removed   = orig_dur - clean_dur
    pct_kept  = 100 * clean_dur / orig_dur if orig_dur > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Original duration",  f"{orig_dur:.1f}s")
    col2.metric("Clean duration",     f"{clean_dur:.1f}s")
    col3.metric("Removed",            f"{removed:.1f}s")
    col4.metric("Data retained",      f"{pct_kept:.0f}%")

    st.markdown(
        f'<div class="status-box">'
        f'ICA components removed: '
        f'{len(st.session_state.ica_excludes)}<br>'
        f'Output file: {out_filename}'
        f'</div>',
        unsafe_allow_html=True
    )

    render_log()

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("#### Export clean EDF")

    if st.button("💾  Export Clean EDF"):
        with st.spinner("Writing EDF file..."):
            try:
                # Write to temp file then offer download
                tmp_out = tempfile.NamedTemporaryFile(
                    suffix=".edf", delete=False
                )
                tmp_out.close()

                raw_export = prepare_export_raw_for_pipeline(raw_clean)

                mne.export.export_raw(
                    tmp_out.name,
                    raw_export,
                    fmt="edf",
                    overwrite=True,
                    verbose=False
                )

                with open(tmp_out.name, "rb") as f:
                    edf_bytes = f.read()

                os.unlink(tmp_out.name)
                log(f"✓ Exported: {out_filename}")

                st.success(f"✓ Clean EDF ready: {out_filename}")
                st.download_button(
                    label="⬇️  Download clean EDF",
                    data=edf_bytes,
                    file_name=out_filename,
                    mime="application/octet-stream"
                )

            except Exception as e:
                st.error(f"Export failed: {e}")
                st.markdown(
                    '<div class="warn-box">'
                    'EDF export requires MNE ≥ 1.2 and edfio.<br>'
                    'Install with: <code>pip install edfio</code>'
                    '</div>',
                    unsafe_allow_html=True
                )

    # ── Restart ───────────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🔄  Clean another file"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PROGRESS SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    stages = {
        "upload":     "① Upload EDF",
        "filter":     "② Filter",
        "autoreject": "③ Autoreject",
        "ica":        "④ ICA",
        "export":     "⑤ Export",
    }
    current = st.session_state.stage
    order   = list(stages.keys())
    current_idx = order.index(current)

    st.sidebar.markdown("### Progress")
    for i, (key, label) in enumerate(stages.items()):
        if i < current_idx:
            st.sidebar.markdown(f"✅ {label}")
        elif i == current_idx:
            st.sidebar.markdown(f"**▶ {label}**")
        else:
            st.sidebar.markdown(f"○ {label}")

    if st.session_state.edf_filename:
        st.sidebar.markdown("---")
        st.sidebar.markdown(
            f"**File:** {st.session_state.edf_filename}"
        )

    if st.session_state.log:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Log")
        for entry in st.session_state.log[-8:]:
            st.sidebar.markdown(
                f'<span style="font-size:0.75rem;'
                f'font-family:monospace;color:#aaaaaa">'
                f'{entry}</span>',
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────
render_sidebar()

stage = st.session_state.stage
if stage == "upload":
    stage_upload()
elif stage == "filter":
    stage_filter()
elif stage == "autoreject":
    stage_autoreject()
elif stage == "ica":
    stage_ica()
elif stage == "export":
    stage_export()
