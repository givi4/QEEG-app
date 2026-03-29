"""
qeeg_launcher.py
Streamlit launcher for the regular QEEG analysis pipeline.
"""

import os
import tempfile
from pathlib import Path

import streamlit as st

from artifact_detection import ArtifactDetectionConfig
from qeeg_pipeline import run_pipeline


st.set_page_config(
    page_title="QEEG Launcher",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
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
    }
    .warn-box {
        background-color: #1a1a2e;
        border-left: 4px solid #ff8800;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        color: #e0e0e0;
    }
    h1, h2, h3 { color: #e0e0e0; }
</style>
""",
    unsafe_allow_html=True,
)


def init_state():
    defaults = {
        "result": None,
        "uploaded_name": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_result(result: dict):
    metadata = result["metadata"]
    report_path = result["report_path"]
    artifact_summary_path = result["artifact_summary_path"]
    overview_path = result["topomap_paths"].get("_overview")

    st.markdown("#### Analysis Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("File", Path(metadata.get("edf_file", "-")).name)
    col2.metric("Epochs", str(metadata.get("n_epochs", "-")))
    col3.metric("Recorded Ref", metadata.get("recording_reference", "unknown"))
    col4.metric("Analysis Ref", metadata.get("analysis_reference", "unknown"))

    st.markdown(
        f'<div class="status-box">Output folder: {result["output_dir"]}</div>',
        unsafe_allow_html=True,
    )

    if os.path.exists(report_path):
        with open(report_path, "rb") as handle:
            st.download_button(
                label="Download PDF Report",
                data=handle.read(),
                file_name=Path(report_path).name,
                mime="application/pdf",
            )

    if overview_path and os.path.exists(overview_path):
        st.markdown("#### Topomaps")
        st.image(overview_path, use_container_width=True)

    if artifact_summary_path and os.path.exists(artifact_summary_path):
        st.markdown("#### Artifact Summary")
        st.image(artifact_summary_path, use_container_width=True)


init_state()

st.title("🧠 QEEG Launcher")
st.markdown("#### Regular analysis pipeline with EDF upload and one-click run")
st.markdown("---")

uploaded = st.file_uploader(
    "Upload EDF file for QEEG analysis",
    type=["edf", "EDF"],
)

col1, col2 = st.columns(2)
with col1:
    reference_mode = st.selectbox(
        "Reference mode",
        options=["linked_ears", "average", "as_recorded", "auto"],
        index=0,
    )
with col2:
    enable_artifact_detection = st.checkbox("Enable artifact detection", value=True)

interactive_review = st.checkbox(
    "Open interactive epoch review window",
    value=False,
    help="Leave this off for the Streamlit workflow unless you want the MNE reviewer window to open.",
)

if st.button("Run QEEG Pipeline"):
    if uploaded is None:
        st.error("Please upload an EDF file first.")
    else:
        with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        output_dir = str(Path("qeeg_output") / Path(uploaded.name).stem)
        st.session_state.uploaded_name = uploaded.name

        with st.spinner("Running QEEG pipeline..."):
            try:
                result = run_pipeline(
                    tmp_path,
                    output_dir=output_dir,
                    reference_mode=reference_mode,
                    interactive=interactive_review,
                    enable_artifact_detection=enable_artifact_detection,
                    artifact_visualize=False,
                    artifact_config=ArtifactDetectionConfig(window_length_s=2.0),
                )
                result["metadata"]["edf_file"] = uploaded.name
                st.session_state.result = result
                st.success("Pipeline complete.")
            except Exception as exc:
                st.session_state.result = None
                st.error(f"Pipeline failed: {exc}")
            finally:
                os.unlink(tmp_path)

if st.session_state.result is not None:
    render_result(st.session_state.result)
