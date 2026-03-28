"""
report.py
Generate a PDF report with metadata, optional topomaps, band power,
and a plain-language summary.
"""

import os
import re
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from band_power import FREQ_BANDS


PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN

C_DARK = colors.HexColor("#0f1117")
C_ACCENT = colors.HexColor("#00d4aa")
C_LIGHT_GRAY = colors.HexColor("#f4f4f6")
C_MID_GRAY = colors.HexColor("#cccccc")
C_TEXT = colors.HexColor("#1a1a2e")
C_FLAG = colors.HexColor("#d32f2f")
C_HEADER_BG = colors.HexColor("#1a1a2e")
C_HEADER_TEXT = colors.white


def _styles():
    _ = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            fontSize=22,
            textColor=C_HEADER_TEXT,
            fontName="Helvetica-Bold",
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            fontSize=11,
            textColor=C_ACCENT,
            fontName="Helvetica",
            alignment=TA_CENTER,
            spaceAfter=2,
        ),
        "section": ParagraphStyle(
            "section",
            fontSize=12,
            textColor=C_DARK,
            fontName="Helvetica-Bold",
            spaceBefore=10,
            spaceAfter=4,
            borderPad=2,
        ),
        "body": ParagraphStyle(
            "body",
            fontSize=9,
            textColor=C_TEXT,
            fontName="Helvetica",
            leading=14,
            alignment=TA_JUSTIFY,
        ),
        "meta_key": ParagraphStyle(
            "meta_key",
            fontSize=9,
            textColor=C_MID_GRAY,
            fontName="Helvetica-Bold",
        ),
        "meta_val": ParagraphStyle(
            "meta_val",
            fontSize=9,
            textColor=C_TEXT,
            fontName="Helvetica",
        ),
        "flag": ParagraphStyle(
            "flag",
            fontSize=8,
            textColor=C_FLAG,
            fontName="Helvetica-Bold",
        ),
        "small": ParagraphStyle(
            "small",
            fontSize=7,
            textColor=C_MID_GRAY,
            fontName="Helvetica",
            alignment=TA_CENTER,
        ),
    }


def _extract_reference_metadata(raw) -> tuple[str, str]:
    description = raw.info.get("description") or ""
    recording_reference = "unknown"
    analysis_reference = "unknown"

    recording_match = re.search(r"recording_reference=([^;]+)", description)
    analysis_match = re.search(r"analysis_reference=([^;]+)", description)

    if recording_match:
        recording_reference = recording_match.group(1)
    if analysis_match:
        analysis_reference = analysis_match.group(1)

    return recording_reference, analysis_reference


def generate_summary(zscores: dict | None, band_power: dict, metadata: dict) -> str:
    if zscores is None:
        return (
            "Normative z-scores were not computed for this recording, so this report "
            "includes band power results without a topographic normative comparison."
        )

    name = metadata.get("patient_name", "The patient")
    bands = list(FREQ_BANDS.keys())
    ch_names = band_power["ch_names"]

    elevated = {b: [] for b in bands}
    suppressed = {b: [] for b in bands}

    for band in bands:
        for ch in ch_names:
            z = zscores[band].get(ch, 0.0)
            if z >= 2.0:
                elevated[band].append((ch, z))
            elif z <= -2.0:
                suppressed[band].append((ch, z))

    for band in bands:
        elevated[band].sort(key=lambda x: -x[1])
        suppressed[band].sort(key=lambda x: x[1])

    paragraphs = []
    date_str = metadata.get("date", datetime.now().strftime("%B %d, %Y"))
    paragraphs.append(
        f"{name}'s QEEG recording ({date_str}) was analyzed using relative "
        f"band power compared to a normative reference database. "
        f"The following findings were noted based on Z-score deviations "
        f"exceeding +/-2.0 standard deviations from the normative mean."
    )

    findings = []
    for band in bands:
        fmin, fmax = FREQ_BANDS[band]
        elev = elevated[band]
        supp = suppressed[band]

        if not elev and not supp:
            findings.append(
                f"{band} activity ({fmin}-{fmax} Hz) was within normal limits "
                f"across all channels."
            )
        else:
            parts = []
            if elev:
                ch_list = ", ".join(ch for ch, _ in elev[:5])
                extra = f" and {len(elev)-5} others" if len(elev) > 5 else ""
                max_z = elev[0][1]
                parts.append(
                    f"elevated {band} power ({fmin}-{fmax} Hz) at "
                    f"{ch_list}{extra} (max Z={max_z:+.1f})"
                )
            if supp:
                ch_list = ", ".join(ch for ch, _ in supp[:5])
                extra = f" and {len(supp)-5} others" if len(supp) > 5 else ""
                min_z = supp[0][1]
                parts.append(
                    f"suppressed {band} power ({fmin}-{fmax} Hz) at "
                    f"{ch_list}{extra} (min Z={min_z:+.1f})"
                )
            findings.append("There was " + " and ".join(parts) + ".")

    paragraphs.append(" ".join(findings))
    paragraphs.append(
        "Note: This report is generated by a non-validated research tool "
        "for clinical reference only. Findings should be interpreted in the "
        "context of the patient's history, symptoms, and other clinical data. "
        "This report does not constitute a medical diagnosis."
    )

    return "\n\n".join(paragraphs)


def generate_report(
    metadata: dict,
    band_power: dict,
    zscores: dict | None,
    topomap_paths: dict,
    artifact_result=None,
    artifact_summary_path: str | None = None,
    output_path: str = "qeeg_output/report.pdf",
) -> str:
    print(f"\n[REPORT] Building PDF -> {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
    )
    S = _styles()
    story = []

    header_table = Table(
        [[
            Paragraph("QEEG Analysis Report", S["title"]),
            Paragraph("Quantitative EEG - Research Use Only", S["subtitle"]),
        ]],
        colWidths=[CONTENT_W],
    )
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_HEADER_BG),
        ("ROUNDEDCORNERS", [4]),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LEFTPADDING", (0, 0), (-1, -1), 16),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph("Session Information", S["section"]))
    story.append(HRFlowable(width=CONTENT_W, thickness=1, color=C_ACCENT))
    story.append(Spacer(1, 3 * mm))

    meta_rows = [
        ["Patient Name", metadata.get("patient_name", "-"), "Patient ID", metadata.get("patient_id", "-")],
        ["Date", metadata.get("date", datetime.now().strftime("%Y-%m-%d")), "Clinician", metadata.get("clinician", "-")],
        ["EDF File", Path(metadata.get("edf_file", "-")).name, "Clean Epochs", str(metadata.get("n_epochs", "-"))],
        ["Recorded Ref", metadata.get("recording_reference", "unknown"), "Analysis Ref", metadata.get("analysis_reference", "unknown")],
        ["Duration", f"{metadata.get('duration_s', 0)/60:.1f} min", "Notes", metadata.get("notes", "-")],
    ]

    meta_table_data = []
    for row in meta_rows:
        meta_table_data.append([
            Paragraph(row[0], S["meta_key"]),
            Paragraph(str(row[1]), S["meta_val"]),
            Paragraph(row[2], S["meta_key"]),
            Paragraph(str(row[3]), S["meta_val"]),
        ])

    meta_table = Table(meta_table_data, colWidths=[35 * mm, 55 * mm, 35 * mm, 55 * mm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_LIGHT_GRAY),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, C_LIGHT_GRAY]),
        ("GRID", (0, 0), (-1, -1), 0.4, C_MID_GRAY),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 6 * mm))

    if artifact_result is not None:
        story.append(Paragraph("Artifact Summary", S["section"]))
        story.append(HRFlowable(width=CONTENT_W, thickness=1, color=C_ACCENT))
        story.append(Spacer(1, 3 * mm))

        bad_window_count = int(artifact_result.bad_windows.sum())
        total_windows = len(artifact_result.bad_windows)
        bad_channel_count = len(artifact_result.bad_channels)
        bad_channel_text = ", ".join(artifact_result.bad_channels) if artifact_result.bad_channels else "None"

        artifact_rows = [
            ["Window Length", f"{artifact_result.window_duration_s:.1f} s", "Bad Windows", f"{bad_window_count} / {total_windows}"],
            ["Bad Channels", str(bad_channel_count), "Channel List", bad_channel_text],
            ["Window Threshold", f"{artifact_result.config.bad_window_channel_fraction:.2f}", "Score Threshold", str(artifact_result.config.score_threshold)],
        ]

        artifact_table_data = []
        for row in artifact_rows:
            artifact_table_data.append([
                Paragraph(row[0], S["meta_key"]),
                Paragraph(str(row[1]), S["meta_val"]),
                Paragraph(row[2], S["meta_key"]),
                Paragraph(str(row[3]), S["meta_val"]),
            ])

        artifact_table = Table(
            artifact_table_data,
            colWidths=[35 * mm, 55 * mm, 35 * mm, 55 * mm]
        )
        artifact_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), C_LIGHT_GRAY),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, C_LIGHT_GRAY]),
            ("GRID", (0, 0), (-1, -1), 0.4, C_MID_GRAY),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(artifact_table)
        story.append(Spacer(1, 3 * mm))

        story.append(Paragraph(
            "Bad windows are detected on short raw EEG segments before FFT. "
            "Shaded regions in the interactive MNE view correspond to annotated bad_artifact segments.",
            S["body"],
        ))
        story.append(Spacer(1, 3 * mm))

        if artifact_summary_path and os.path.exists(artifact_summary_path):
            img = Image(artifact_summary_path, width=CONTENT_W, height=CONTENT_W * 0.60)
            story.append(img)
            story.append(Spacer(1, 4 * mm))
        else:
            story.append(Paragraph("Artifact summary image not available.", S["small"]))
            story.append(Spacer(1, 4 * mm))

    story.append(Paragraph("Z-Score Topographic Maps", S["section"]))
    story.append(HRFlowable(width=CONTENT_W, thickness=1, color=C_ACCENT))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "Red = elevated vs. normative (Z > +2). "
        "Blue = suppressed vs. normative (Z < -2). "
        "Scale: -3 to +3 SD.",
        S["small"],
    ))
    story.append(Spacer(1, 3 * mm))

    bands = [b for b in FREQ_BANDS.keys() if b in topomap_paths]
    topo_img_w = (CONTENT_W - 8 * mm) / 3

    if bands:
        for row_start in range(0, len(bands), 3):
            row_bands = bands[row_start:row_start + 3]
            row_imgs = []
            for band in row_bands:
                path = topomap_paths.get(band)
                if path and os.path.exists(path):
                    row_imgs.append(Image(path, width=topo_img_w, height=topo_img_w))
                else:
                    row_imgs.append(Paragraph(f"{band}\n(image missing)", S["small"]))

            while len(row_imgs) < 3:
                row_imgs.append(Spacer(1, 1))

            topo_table = Table([row_imgs], colWidths=[topo_img_w] * 3)
            topo_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ]))
            story.append(topo_table)
            story.append(Spacer(1, 2 * mm))
    else:
        story.append(Paragraph(
            "Topographic maps were skipped because no normative z-score comparison was available.",
            S["body"],
        ))

    story.append(Spacer(1, 4 * mm))

    story.append(KeepTogether([
        Paragraph("Band Power Summary", S["section"]),
        HRFlowable(width=CONTENT_W, thickness=1, color=C_ACCENT),
        Spacer(1, 3 * mm),
    ]))

    bands_list = list(FREQ_BANDS.keys())
    ch_names = band_power["ch_names"]

    def _power_row(ch):
        row = [Paragraph(ch, S["meta_key"])]
        for band in bands_list:
            rel = band_power["relative"][band].get(ch, 0.0)
            z = float("nan") if zscores is None else zscores[band].get(ch, 0.0)
            is_flagged = zscores is not None and abs(z) >= 2.0
            style = S["flag"] if is_flagged else S["body"]
            z_text = "n/a" if zscores is None else f"{z:+.1f}"
            row.append(Paragraph(f"{rel:.3f}\nZ:{z_text}", style))
        return row

    hdr = [Paragraph("Ch", S["meta_key"])]
    for b in bands_list:
        fmin, fmax = FREQ_BANDS[b]
        hdr.append(Paragraph(f"{b}\n{fmin}-{fmax}Hz", S["meta_key"]))

    power_data = [hdr]
    for ch in ch_names:
        power_data.append(_power_row(ch))

    ch_col_w = 14 * mm
    band_col_w = (CONTENT_W - ch_col_w) / len(bands_list)

    power_table = Table(
        power_data,
        colWidths=[ch_col_w] + [band_col_w] * len(bands_list),
        repeatRows=1,
    )
    power_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, C_LIGHT_GRAY]),
        ("GRID", (0, 0), (-1, -1), 0.3, C_MID_GRAY),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(power_table)
    story.append(Spacer(1, 6 * mm))

    story.append(KeepTogether([
        Paragraph("Clinical Summary", S["section"]),
        HRFlowable(width=CONTENT_W, thickness=1, color=C_ACCENT),
        Spacer(1, 3 * mm),
    ]))

    summary_text = generate_summary(zscores, band_power, metadata)
    for para in summary_text.split("\n\n"):
        story.append(Paragraph(para, S["body"]))
        story.append(Spacer(1, 3 * mm))

    story.append(Spacer(1, 4 * mm))
    story.append(HRFlowable(width=CONTENT_W, thickness=0.5, color=C_MID_GRAY))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        f"Generated by QEEG Analysis Tool | "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')} | "
        f"Normative comparison uses literature-derived adult reference data; "
        f"not validated for clinical diagnosis.",
        S["small"],
    ))

    doc.build(story)
    print(f"    [OK] PDF saved: {output_path}")
    return output_path


def default_metadata(raw, n_epochs: int, edf_path: str) -> dict:
    recording_reference, analysis_reference = _extract_reference_metadata(raw)
    return {
        "patient_name": "Test Patient",
        "patient_id": "001",
        "patient_age": 35,
        "eyes_condition": "closed",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "clinician": "-",
        "notes": "Eyes closed, resting state",
        "n_epochs": n_epochs,
        "duration_s": n_epochs * 5.0,
        "edf_file": edf_path,
        "recording_reference": recording_reference,
        "analysis_reference": analysis_reference,
    }
