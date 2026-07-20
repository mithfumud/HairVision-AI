"""Explainable overlays from segmentation masks and measured features."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# BGR/RGB overlay colors (RGB) and alpha in [0, 1].
HAIR_COLOR: tuple[int, int, int] = (40, 180, 60)
HAIR_ALPHA: float = 0.40

SCALP_COLOR: tuple[int, int, int] = (255, 140, 0)
SCALP_ALPHA: float = 0.40

HEAD_BOUNDARY_COLOR: tuple[int, int, int] = (255, 255, 255)
HEAD_BOUNDARY_THICKNESS: int = 2

PANEL_BG_COLOR: tuple[int, int, int, int] = (0, 0, 0, 160)
PANEL_TEXT_COLOR: tuple[int, int, int] = (255, 255, 255)
PANEL_MARGIN: int = 12
PANEL_PADDING: int = 10
PANEL_LINE_HEIGHT: int = 18

OverlayFn = Callable[
    [Image.Image, dict[str, Any], dict[str, Any]],
    Image.Image,
]


def _as_bool_mask(mask: Any, name: str) -> np.ndarray:
    """Coerce a mask to a 2D boolean array."""
    if mask is None:
        raise KeyError(f"segmentation is missing required key: {name}")
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    return arr.astype(bool)


def _blend_mask(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> Image.Image:
    """Alpha-blend a solid color onto pixels where mask is True."""
    if not np.any(mask):
        return image

    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    out = base.copy()
    color_arr = np.array(color, dtype=np.float32)
    out[mask] = (1.0 - alpha) * base[mask] + alpha * color_arr
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def _format_percent(value: Any) -> str:
    """Format a ratio in [0, 1] as a percentage with one decimal place."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100.0:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


class Visualizer:
    """
    Build annotated RGB visualizations from masks and features.

    Overlay steps are registered in ``self._overlays`` so hairline, temple,
    crown, and stage drawings can be appended later without changing
    ``visualize``.
    """

    def __init__(self) -> None:
        self._overlays: list[OverlayFn] = [
            self._overlay_hair,
            self._overlay_scalp,
            self._draw_head_boundary,
            self._draw_info_panel,
        ]

    def add_overlay(self, overlay_fn: OverlayFn) -> None:
        """Append a custom overlay step (inserted before the info panel)."""
        # Keep the info panel last when possible.
        if self._overlays and self._overlays[-1] == self._draw_info_panel:
            self._overlays.insert(-1, overlay_fn)
        else:
            self._overlays.append(overlay_fn)

    def visualize(
        self,
        image: Image.Image,
        segmentation: dict,
        features: dict,
    ) -> Image.Image:
        """Return an RGB image with segmentation overlays and a metrics panel."""
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")
        if not isinstance(segmentation, dict):
            raise TypeError("segmentation must be a dict")
        if not isinstance(features, dict):
            raise TypeError("features must be a dict")

        canvas = image.convert("RGB")
        for overlay in self._overlays:
            canvas = overlay(canvas, segmentation, features)
        return canvas

    def _overlay_hair(
        self,
        image: Image.Image,
        segmentation: dict,
        features: dict,
    ) -> Image.Image:
        """Tint the hair mask green."""
        del features  # reserved for future density-aware styling
        hair = _as_bool_mask(segmentation.get("hair_mask"), "hair_mask")
        return _blend_mask(image, hair, HAIR_COLOR, HAIR_ALPHA)

    def _overlay_scalp(
        self,
        image: Image.Image,
        segmentation: dict,
        features: dict,
    ) -> Image.Image:
        """Tint exposed scalp (skin within head) orange."""
        del features
        skin = _as_bool_mask(segmentation.get("skin_mask"), "skin_mask")
        head = _as_bool_mask(segmentation.get("head_mask"), "head_mask")
        if skin.shape != head.shape:
            raise ValueError("skin_mask and head_mask must share the same shape")
        scalp = skin & head
        return _blend_mask(image, scalp, SCALP_COLOR, SCALP_ALPHA)

    def _draw_head_boundary(
        self,
        image: Image.Image,
        segmentation: dict,
        features: dict,
    ) -> Image.Image:
        """Draw the outer head contour in white using OpenCV."""
        del features
        head = _as_bool_mask(segmentation.get("head_mask"), "head_mask")
        if not np.any(head):
            return image

        mask_u8 = (head.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return image

        rgb = np.asarray(image).copy()
        cv2.drawContours(
            rgb,
            contours,
            contourIdx=-1,
            color=HEAD_BOUNDARY_COLOR,
            thickness=HEAD_BOUNDARY_THICKNESS,
        )
        return Image.fromarray(rgb, mode="RGB")

    def _draw_info_panel(
        self,
        image: Image.Image,
        segmentation: dict,
        features: dict,
    ) -> Image.Image:
        """Draw a small PIL text panel with key measured features."""
        del segmentation
        lines = [
            f"Hair Density: {_format_percent(features.get('hair_density'))}",
            f"Scalp Exposure: {_format_percent(features.get('scalp_exposure'))}",
            f"Hair Percentage: {_format_percent(features.get('hair_percentage'))}",
            f"Scalp Percentage: {_format_percent(features.get('scalp_percentage'))}",
        ]

        font = ImageFont.load_default()
        # Estimate panel size from text metrics.
        text_widths: list[int] = []
        for line in lines:
            bbox = font.getbbox(line)
            text_widths.append(bbox[2] - bbox[0])
        panel_w = max(text_widths) + 2 * PANEL_PADDING
        panel_h = len(lines) * PANEL_LINE_HEIGHT + 2 * PANEL_PADDING

        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        x0, y0 = PANEL_MARGIN, PANEL_MARGIN
        draw.rectangle(
            (x0, y0, x0 + panel_w, y0 + panel_h),
            fill=PANEL_BG_COLOR,
        )

        y = y0 + PANEL_PADDING
        for line in lines:
            draw.text(
                (x0 + PANEL_PADDING, y),
                line,
                fill=PANEL_TEXT_COLOR + (255,),
                font=font,
            )
            y += PANEL_LINE_HEIGHT

        return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
