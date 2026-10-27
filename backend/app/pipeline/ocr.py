"""Number-plate OCR with PaddleOCR (PP-OCRv5 mobile detection + English recognition).

Why PaddleOCR: highest accuracy among free OCR engines on short, printed alphanumeric text, handles
multi-line text (two-line Indian bike plates) and runs on CPU (~0.2-0.6 s per plate).

Pre-processing before OCR: the plate crop gets a small margin, is upscaled so its height is at least
``min_height`` px (recognition models work best on ~48-64 px high text lines) and gets mild contrast
enhancement. Text lines are ordered top-to-bottom, left-to-right, concatenated and validated by
:mod:`app.pipeline.plate_text`.
"""
from __future__ import annotations

import logging
import os

import cv2
import numpy as np

from app.core.exceptions import ModelLoadError
from app.pipeline.plate_text import normalise
from app.pipeline.types import Box, PlateRead

log = logging.getLogger(__name__)


class PlateReader:
    def __init__(self, lang: str = "en", min_height: int = 96, use_gpu: bool = False):
        # On Windows torch must be imported before paddle, otherwise loading torch's DLLs fails later.
        import torch  # noqa: F401
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        try:
            from paddleocr import PaddleOCR
        except ImportError as e:
            raise ModelLoadError("paddleocr is not installed (pip install paddlepaddle paddleocr)") from e
        try:
            self.ocr = PaddleOCR(
                lang=lang,
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="en_PP-OCRv5_mobile_rec" if lang == "en" else None,
                use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
                # oneDNN (MKL-DNN) crashes with Paddle 3.3 on some CPUs ("ConvertPirAttribute2RuntimeAttribute")
                enable_mkldnn=False,
                device="gpu" if use_gpu else "cpu",
            )
        except Exception as e:
            raise ModelLoadError(f"cannot initialise PaddleOCR: {e}") from e
        self.min_height = min_height
        log.info("PaddleOCR ready (%s)", "GPU" if use_gpu else "CPU")

    def _prepare(self, crop: np.ndarray) -> np.ndarray:
        h, w = crop.shape[:2]
        if h < self.min_height:
            s = self.min_height / max(h, 1)
            crop = cv2.resize(crop, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(lab[..., 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _ocr_once(self, crop: np.ndarray, plate_box: Box) -> PlateRead:
        import torch
        threads = torch.get_num_threads()
        try:
            res = self.ocr.predict(self._prepare(crop))[0]
            texts, scores, boxes = list(res["rec_texts"]), list(res["rec_scores"]), list(res["rec_boxes"])
        except Exception as e:                       # OCR failure must not fail the whole detection
            log.warning("OCR failed: %s", e)
            return PlateRead(box=plate_box)
        finally:
            # Paddle sets the shared OpenMP pool to 1 thread on its first call, which made every later
            # YOLO inference ~2.4x slower on CPU (measured 0.57 s -> 1.35 s per frame); restore it
            torch.set_num_threads(threads)
        lines = [(bx, t, s) for bx, t, s in zip(boxes, texts, scores) if t.strip() and s > 0.3]
        if not lines:
            return PlateRead(box=plate_box)
        # reading order: top line first; on the same line, left to right
        line_h = np.median([bx[3] - bx[1] for bx, _, _ in lines])
        lines.sort(key=lambda l: (round(l[0][1] / max(line_h, 1)), l[0][0]))
        raw = " ".join(t for _, t, _ in lines)
        text, valid = normalise(raw)
        return PlateRead(text=text, raw_text=raw, confidence=float(np.mean([s for _, _, s in lines])),
                         valid_format=valid, box=plate_box)

    def read(self, image: np.ndarray, plate_box: Box, pad: float = 0.08) -> PlateRead:
        """OCR the plate at ``plate_box`` (full-image coordinates) of ``image``.

        If the first read is not a valid Indian number, the mirrored and the 180-degree-rotated crop
        are tried too (mirrored/upside-down plates occur in some datasets and camera setups). A flipped
        read is used only if it is a valid number; otherwise the original read is kept.
        """
        H, W = image.shape[:2]
        x1, y1, x2, y2 = plate_box.expanded(pad, W, H).as_int()
        crop = image[y1:y2, x1:x2]
        if crop.size == 0 or min(crop.shape[:2]) < 6:
            return PlateRead(box=plate_box)
        best = self._ocr_once(crop, plate_box)
        if best.valid_format:
            return best
        for variant in (cv2.flip(crop, 1), cv2.rotate(crop, cv2.ROTATE_180)):
            r = self._ocr_once(variant, plate_box)
            if r.valid_format:       # only a *valid* flipped read replaces the original read
                return r
        return best
