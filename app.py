"""
HairVision-AI — Streamlit MVP (v1).

UI-only layer. Calls HairAnalysisPipeline.analyze(...) and renders the report.
Does not modify or re-implement analytical modules.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image

from analysis.pipeline import HairAnalysisPipeline, HairAnalysisReport

ACCEPTED_TYPES = ["jpg", "jpeg", "png"]
DEFICIT_COLOR = (255, 95, 55)
DEFICIT_ALPHA = 0.45


def _load_upload(uploaded) -> Image.Image | None:
    """Decode an uploaded file to RGB PIL Image."""
    if uploaded is None:
        return None
    return Image.open(uploaded).convert("RGB")


def _format_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{float(value):.1f}%"


def _format_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def _compose_deficit_annotation(
    image: Image.Image,
    deficit_mask: np.ndarray | None,
) -> Image.Image:
    """
    Present the pipeline deficit mask on the source image.

    Uses only masks already produced by HairAnalysisPipeline. No analytical
    recomputation — alpha-blend presentation for Section 1.
    When the pipeline upscaled the analysis image, the mask is resized to the
    original upload so the overlay still aligns.
    """
    canvas = image.convert("RGB")
    if deficit_mask is None:
        return canvas

    mask = np.asarray(deficit_mask).astype(bool)
    if mask.shape[:2] != (canvas.height, canvas.width):
        mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        mask_img = mask_img.resize(canvas.size, Image.Resampling.NEAREST)
        mask = np.asarray(mask_img) > 0
    if not np.any(mask):
        return canvas

    base = np.asarray(canvas, dtype=np.float32)
    out = base.copy()
    color = np.array(DEFICIT_COLOR, dtype=np.float32)
    out[mask] = (1.0 - DEFICIT_ALPHA) * base[mask] + DEFICIT_ALPHA * color
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def _run_pipeline(
    front_image: Image.Image | None,
    crown_image: Image.Image | None,
) -> HairAnalysisReport:
    """Invoke the frozen clinical pipeline only."""
    with HairAnalysisPipeline() as pipeline:
        return pipeline.analyze(
            front_image=front_image,
            crown_image=crown_image,
        )


def _render_annotated_images(
    report: HairAnalysisReport,
    front_image: Image.Image | None,
    crown_image: Image.Image | None,
) -> None:
    st.subheader("1. Annotated Images")

    front_ann = None
    crown_ann = None

    if front_image is not None and report.front_deficit_result is not None:
        front_ann = _compose_deficit_annotation(
            front_image,
            report.front_deficit_result.deficit_mask,
        )
    elif front_image is not None and report.hair_deficit_result is not None:
        # front-only: primary deficit is the front result
        if report.analysis_mode and report.analysis_mode.value == "front_only":
            front_ann = _compose_deficit_annotation(
                front_image,
                report.hair_deficit_result.deficit_mask,
            )

    if crown_image is not None and report.crown_deficit_result is not None:
        crown_ann = _compose_deficit_annotation(
            crown_image,
            report.crown_deficit_result.deficit_mask,
        )
    elif crown_image is not None and report.hair_deficit_result is not None:
        if report.analysis_mode and report.analysis_mode.value == "crown_only":
            crown_ann = _compose_deficit_annotation(
                crown_image,
                report.hair_deficit_result.deficit_mask,
            )

    if front_ann is None and crown_ann is None:
        st.info("No annotated views available for this run.")
        return

    if front_ann is not None and crown_ann is not None:
        col_f, col_c = st.columns(2)
        with col_f:
            st.caption("Front")
            st.image(front_ann, use_container_width=True)
        with col_c:
            st.caption("Crown")
            st.image(crown_ann, use_container_width=True)
    elif front_ann is not None:
        st.caption("Front")
        st.image(front_ann, use_container_width=True)
    else:
        st.caption("Crown")
        st.image(crown_ann, use_container_width=True)


def _render_clinical_results(report: HairAnalysisReport) -> None:
    st.subheader("2. Clinical Results")

    metrics = report.hair_loss_metrics
    norwood = report.norwood_result
    rows: list[tuple[str, str]] = []

    if report.analysis_mode is not None:
        rows.append(("Analysis Mode", report.analysis_mode.value))

    if metrics is not None:
        if metrics.front_loss_percentage is not None:
            rows.append(("Front Loss %", _format_pct(metrics.front_loss_percentage)))
        if metrics.left_temple_loss_percentage is not None:
            rows.append(
                ("Left Temple %", _format_pct(metrics.left_temple_loss_percentage))
            )
        if metrics.right_temple_loss_percentage is not None:
            rows.append(
                ("Right Temple %", _format_pct(metrics.right_temple_loss_percentage))
            )
        if metrics.crown_loss_percentage is not None:
            rows.append(("Crown Loss %", _format_pct(metrics.crown_loss_percentage)))
        if metrics.overall_hair_loss_percentage is not None:
            rows.append(
                ("Overall Hair Loss %", _format_pct(metrics.overall_hair_loss_percentage))
            )
        if metrics.overall_hair_coverage_percentage is not None:
            rows.append(
                (
                    "Overall Coverage %",
                    _format_pct(metrics.overall_hair_coverage_percentage),
                )
            )
        if metrics.largest_deficit_zone is not None:
            zone_label = str(metrics.largest_deficit_zone)
            if metrics.largest_deficit_percentage is not None:
                zone_label = (
                    f"{zone_label} ({_format_pct(metrics.largest_deficit_percentage)})"
                )
            rows.append(("Largest Deficit Zone", zone_label))

    if norwood is not None:
        rows.append(("Norwood Stage", norwood.stage.value))
        rows.append(("Confidence", f"{norwood.confidence:.1f}"))
        rows.append(("Rule ID", norwood.rule_id))

    if not rows:
        st.info("No clinical metrics available.")
        return

    st.table({"Metric": [r[0] for r in rows], "Value": [r[1] for r in rows]})


def _render_interpretation(report: HairAnalysisReport) -> None:
    st.subheader("3. Clinical Interpretation")

    norwood = report.norwood_result
    if norwood is None:
        st.info("No clinical interpretation available.")
        return

    with st.container():
        st.markdown("**Explanation**")
        st.write(norwood.explanation)

        st.markdown("**Recommendation**")
        st.write(norwood.recommendation)

        st.markdown("**Limitations**")
        if norwood.limitations:
            for item in norwood.limitations:
                st.write(f"- {item}")
        else:
            st.write("—")

        if report.warnings:
            st.markdown("**Warnings**")
            for warning in report.warnings:
                st.warning(warning)


def _render_debug(report: HairAnalysisReport) -> None:
    with st.expander("4. Debug Information", expanded=False):
        st.write(f"Processing time: {report.processing_time:.4f} s")
        st.write("Pipeline metadata:")
        st.json(report.metadata or {})
        if report.errors:
            st.write("Errors:")
            for err in report.errors:
                st.write(f"- {err}")
        else:
            st.write("Errors: none")
        st.write("Raw report.to_dict():")
        st.code(
            json.dumps(report.to_dict(), indent=2, default=str),
            language="json",
        )


def main() -> None:
    st.set_page_config(
        page_title="HairVision-AI",
        page_icon=None,
        layout="centered",
    )

    st.title("HairVision-AI")
    st.subheader("AI Powered Hair Loss Analysis")
    st.write(
        "Upload a front image, a crown image, or both to analyze hair loss "
        "and estimate the Norwood stage."
    )

    st.divider()

    col_front, col_crown = st.columns(2)
    with col_front:
        st.markdown("**Front Image**")
        front_file = st.file_uploader(
            "Front (optional)",
            type=ACCEPTED_TYPES,
            key="front_upload",
            help="Accepted: JPG, JPEG, PNG",
        )
        if front_file is not None:
            st.image(_load_upload(front_file), caption="Front preview", use_container_width=True)

    with col_crown:
        st.markdown("**Crown Image**")
        crown_file = st.file_uploader(
            "Crown (optional)",
            type=ACCEPTED_TYPES,
            key="crown_upload",
            help="Accepted: JPG, JPEG, PNG",
        )
        if crown_file is not None:
            st.image(_load_upload(crown_file), caption="Crown preview", use_container_width=True)

    st.divider()
    analyze_clicked = st.button("Analyze Hair", type="primary", use_container_width=True)

    if not analyze_clicked:
        return

    if front_file is None and crown_file is None:
        st.warning(
            "Please upload at least one image (front, crown, or both) before analyzing."
        )
        return

    try:
        front_image = _load_upload(front_file)
        crown_image = _load_upload(crown_file)
    except Exception as exc:  # noqa: BLE001 — UI boundary
        st.error(f"Could not read uploaded image: {exc}")
        return

    try:
        with st.spinner("Analyzing images..."):
            report = _run_pipeline(front_image, crown_image)
    except Exception as exc:  # noqa: BLE001 — never crash the UI
        st.error(f"Pipeline failed: {exc}")
        return

    if not report.success:
        message = "; ".join(report.errors) if report.errors else "Analysis failed."
        st.error(message)
        if report.warnings:
            for warning in report.warnings:
                st.warning(warning)
        _render_debug(report)
        return

    st.success("Analysis complete.")
    if report.warnings:
        for warning in report.warnings:
            st.warning(warning)
    _render_annotated_images(report, front_image, crown_image)
    _render_clinical_results(report)
    _render_interpretation(report)
    _render_debug(report)


if __name__ == "__main__":
    main()
