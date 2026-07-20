"""Semantic segmentation for hair, skin, and head regions."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

MODEL_ID = "jonathandinu/face-parsing"

# Expected CelebAMask-HQ label names used by jonathandinu/face-parsing.
HAIR_LABEL = "hair"
SKIN_LABEL = "skin"
HEAD_LABELS = ("hair", "skin", "l_ear", "r_ear")


class HairSegmenter:
    """SegFormer face-parsing wrapper. Loads once; reusable across calls."""

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = SegformerImageProcessor.from_pretrained(model_id)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

        id2label: dict[int, str] = {
            int(k): v for k, v in self.model.config.id2label.items()
        }
        self.id2label = id2label
        self.label2id = {v: k for k, v in id2label.items()}

    def _label_id(self, name: str) -> int | None:
        """Return class id for a label name, or None if absent."""
        return self.label2id.get(name)

    def _binary_mask(self, labels: np.ndarray, class_id: int | None) -> np.ndarray:
        """Build a uint8 {0, 1} mask for one class; empty if class is missing."""
        if class_id is None:
            return np.zeros(labels.shape, dtype=np.uint8)
        return (labels == class_id).astype(np.uint8)

    @torch.inference_mode()
    def segment(self, image: Image.Image) -> dict[str, Any]:
        """
        Run semantic segmentation on a PIL image.

        Returns a dict with the full label map, binary hair/skin/head masks,
        and a hair softmax probability map — all at the original image size.
        """
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")

        image = image.convert("RGB")
        width, height = image.size

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        logits = self.model(**inputs).logits  # (1, C, h/4, w/4)
        upsampled = F.interpolate(
            logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        probs = F.softmax(upsampled, dim=1)[0]  # (C, H, W)
        labels = upsampled.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int32)

        hair_id = self._label_id(HAIR_LABEL)
        skin_id = self._label_id(SKIN_LABEL)

        if hair_id is None:
            hair_probability = np.zeros((height, width), dtype=np.float32)
        else:
            hair_probability = probs[hair_id].detach().cpu().numpy().astype(np.float32)

        hair_mask = self._binary_mask(labels, hair_id)
        skin_mask = self._binary_mask(labels, skin_id)

        head_mask = np.zeros((height, width), dtype=np.uint8)
        for name in HEAD_LABELS:
            head_mask |= self._binary_mask(labels, self._label_id(name))

        return {
            "segmentation_mask": labels,
            "hair_mask": hair_mask.astype(bool),
            "skin_mask": skin_mask.astype(bool),
            "head_mask": head_mask.astype(bool),
            "hair_probability": hair_probability,
            "image_size": (width, height),
            "id2label": self.id2label,
        }
