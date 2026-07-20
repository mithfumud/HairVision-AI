"""Image quality validation for hair-analysis inputs."""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any, Literal

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

ImageType = Literal["front", "crown"]

# Recommended working resolution (soft warning + auto-upscale if below).
RECOMMENDED_MIN_WIDTH = 512
RECOMMENDED_MIN_HEIGHT = 512
# Absolute floor — below this, inference is not practical.
HARD_MIN_WIDTH = 64
HARD_MIN_HEIGHT = 64

BLUR_WARN_THRESHOLD = 100.0
BRIGHTNESS_LOW = 40.0
BRIGHTNESS_HIGH = 220.0
CONTRAST_LOW = 25.0

# Back-compat aliases (recommended size; no longer a hard gate).
MIN_WIDTH = RECOMMENDED_MIN_WIDTH
MIN_HEIGHT = RECOMMENDED_MIN_HEIGHT

_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "assets" / "models"
_DEFAULT_MODEL_PATH = _DEFAULT_MODEL_DIR / "face_landmarker.task"

_MSG_RESOLUTION_LOW = (
    "Image resolution is lower than recommended. Results may be less accurate."
)
_MSG_UPSCALED = (
    "Image was automatically upscaled to the recommended resolution for analysis."
)


def _ensure_face_landmarker_model(model_path: Path) -> Path:
    """Download the Face Landmarker asset if it is not already present."""
    if model_path.is_file():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(_FACE_LANDMARKER_URL, model_path)
    return model_path


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to an OpenCV BGR array."""
    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _blur_score(gray: np.ndarray) -> float:
    """Variance of Laplacian — higher means sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _mean_brightness(gray: np.ndarray) -> float:
    """Average grayscale intensity in [0, 255]."""
    return float(np.mean(gray))


def upscale_to_recommended(image: Image.Image) -> tuple[Image.Image, bool]:
    """
    Upscale so both sides meet the recommended minimum, preserving aspect ratio.

    Returns ``(image, upscaled)``. Never downscales.
    """
    width, height = image.size
    if width <= 0 or height <= 0:
        return image, False
    if width >= RECOMMENDED_MIN_WIDTH and height >= RECOMMENDED_MIN_HEIGHT:
        return image, False

    scale = max(
        RECOMMENDED_MIN_WIDTH / float(width),
        RECOMMENDED_MIN_HEIGHT / float(height),
    )
    if scale <= 1.0:
        return image, False

    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return resized, True


class ImageQualityChecker:
    """Validate whether an image is suitable for front or crown hair analysis."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        path = Path(model_path) if model_path is not None else _DEFAULT_MODEL_PATH
        self._model_path = _ensure_face_landmarker_model(path)
        self._landmarker: FaceLandmarker | None = None

    def _get_landmarker(self) -> FaceLandmarker:
        """Create the Face Landmarker once and reuse it."""
        if self._landmarker is None:
            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self._model_path)),
                running_mode=RunningMode.IMAGE,
                num_faces=1,
            )
            self._landmarker = FaceLandmarker.create_from_options(options)
        return self._landmarker

    def _face_detected(self, image_rgb: np.ndarray) -> bool:
        """Return True if MediaPipe Face Landmarker finds at least one face."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self._get_landmarker().detect(mp_image)
        return bool(result.face_landmarks)

    def detect_landmarks(self, image: Image.Image) -> np.ndarray:
        """
        Detect face landmarks and return an (N, 3) array of normalized x, y, z.

        Reuses the Face Landmarker loaded by this checker. Raises ValueError
        when no face is found.
        """
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")

        rgb = np.asarray(image.convert("RGB"))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._get_landmarker().detect(mp_image)
        if not result.face_landmarks:
            raise ValueError("no face landmarks detected")

        face = result.face_landmarks[0]
        return np.asarray(
            [[lm.x, lm.y, lm.z] for lm in face],
            dtype=np.float64,
        )

    def validate(self, image: Image.Image, image_type: str) -> dict[str, Any]:
        """
        Run quality checks for a front or crown photo.

        Hard failures (``valid=False``, listed in ``errors``) stop analysis:
        undecodable / empty image, resolution below the practical floor, or
        no face on a front image.

        Soft issues (``warnings``) never set ``valid=False``: recommended
        resolution shortfall, blur, brightness, contrast. When resolution is
        below recommended but still usable, ``prepared_image`` is an
        aspect-preserving LANCZOS upscale to at least 512×512.
        """
        if image_type not in ("front", "crown"):
            raise ValueError("image_type must be 'front' or 'crown'")
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")

        try:
            working = image.convert("RGB")
        except Exception as exc:  # noqa: BLE001
            return {
                "valid": False,
                "image_type": image_type,
                "errors": [f"image_corrupted ({exc})"],
                "warnings": [],
                "prepared_image": image,
                "upscaled": False,
                "metrics": {
                    "blur_score": 0.0,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "resolution": [0, 0],
                    "prepared_resolution": [0, 0],
                },
            }

        width, height = working.size
        errors: list[str] = []
        warnings: list[str] = []

        if width <= 0 or height <= 0:
            errors.append("image_empty (could not read pixel dimensions)")
            return {
                "valid": False,
                "image_type": image_type,
                "errors": errors,
                "warnings": warnings,
                "prepared_image": working,
                "upscaled": False,
                "metrics": {
                    "blur_score": 0.0,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "resolution": [width, height],
                    "prepared_resolution": [width, height],
                },
            }

        if width < HARD_MIN_WIDTH or height < HARD_MIN_HEIGHT:
            errors.append(
                f"resolution_unusable ({width}x{height}; "
                f"practical minimum is {HARD_MIN_WIDTH}x{HARD_MIN_HEIGHT})"
            )

        bgr = _pil_to_bgr(working)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blur = _blur_score(gray)
        brightness = _mean_brightness(gray)
        contrast = float(gray.std())

        below_recommended = (
            width < RECOMMENDED_MIN_WIDTH or height < RECOMMENDED_MIN_HEIGHT
        )
        if below_recommended and not errors:
            warnings.append(
                f"{_MSG_RESOLUTION_LOW} "
                f"(got {width}x{height}; recommended "
                f"{RECOMMENDED_MIN_WIDTH}x{RECOMMENDED_MIN_HEIGHT}+)."
            )

        if blur < BLUR_WARN_THRESHOLD:
            warnings.append(
                f"image_may_be_blurry (blur_score={blur:.1f}; "
                f"threshold={BLUR_WARN_THRESHOLD})"
            )

        if brightness < BRIGHTNESS_LOW:
            warnings.append(
                f"image_too_dark (brightness={brightness:.1f}; "
                f"threshold={BRIGHTNESS_LOW})"
            )
        elif brightness > BRIGHTNESS_HIGH:
            warnings.append(
                f"image_too_bright (brightness={brightness:.1f}; "
                f"threshold={BRIGHTNESS_HIGH})"
            )

        if contrast < CONTRAST_LOW:
            warnings.append("low_contrast")

        if image_type == "front":
            rgb = np.asarray(working)
            if not self._face_detected(rgb):
                errors.append("no_face_detected (required for front images)")

        prepared = working
        upscaled = False
        if not errors and below_recommended:
            prepared, upscaled = upscale_to_recommended(working)
            if upscaled:
                warnings.append(
                    f"{_MSG_UPSCALED} "
                    f"({width}x{height} → {prepared.size[0]}x{prepared.size[1]})."
                )

        prep_w, prep_h = prepared.size
        return {
            "valid": len(errors) == 0,
            "image_type": image_type,
            "errors": errors,
            "warnings": warnings,
            "prepared_image": prepared,
            "upscaled": upscaled,
            "metrics": {
                "blur_score": round(blur, 2),
                "brightness": round(brightness, 2),
                "contrast": round(contrast, 2),
                "resolution": [width, height],
                "prepared_resolution": [prep_w, prep_h],
            },
        }

    def close(self) -> None:
        """Release the Face Landmarker if it was created."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> ImageQualityChecker:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
