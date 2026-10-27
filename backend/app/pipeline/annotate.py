"""Draw detection results on an image (used for both images and video frames)."""
from __future__ import annotations

import cv2
import numpy as np

from app.pipeline.types import FrameAnalysis, HelmetStatus

# BGR colours
RED, GREEN, GREY, YELLOW, WHITE, BLACK = (40, 40, 230), (60, 180, 60), (150, 150, 150), (0, 215, 255), (255, 255, 255), (0, 0, 0)
STATUS_COLOR = {HelmetStatus.VIOLATION: RED, HelmetStatus.SAFE: GREEN, HelmetStatus.UNKNOWN: GREY}


def _label(img: np.ndarray, text: str, x: int, y: int, color, scale: float) -> None:
    th = max(1, int(scale * 2))
    (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    y = max(tht + 4, y)
    cv2.rectangle(img, (x, y - tht - base - 4), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - base - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE if color != YELLOW else BLACK, th, cv2.LINE_AA)


def draw(image: np.ndarray, analysis: FrameAnalysis, banner: bool = True) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    scale = max(0.45, min(w, h) / 1400)
    thick = max(2, int(min(w, h) / 450))
    for m in analysis.motorcycles:
        c = STATUS_COLOR[m.status]
        x1, y1, x2, y2 = m.box.as_int()
        cv2.rectangle(out, (x1, y1), (x2, y2), c, thick)
        tag = f"{m.status.value}" + (f" #{m.track_id}" if m.track_id is not None else "")
        _label(out, tag, x1, y1, c, scale)
        for hd in m.heads:
            hc = GREEN if hd.label == "helmet" else RED
            a1, b1, a2, b2 = hd.as_int()
            cv2.rectangle(out, (a1, b1), (a2, b2), hc, thick)
            _label(out, f"{'Helmet' if hd.label == 'helmet' else 'No helmet'} {hd.conf:.2f}", a1, b1, hc, scale * 0.8)
        if m.plate and m.plate.box:
            p1, q1, p2, q2 = m.plate.box.as_int()
            cv2.rectangle(out, (p1, q1), (p2, q2), YELLOW, thick)
            below = q2 + int(28 * scale) + 4
            # below the plate if there is room, otherwise above it (never over the plate itself)
            _label(out, m.plate.text or "plate (unreadable)", p1, below if below < h else q1 - 2, YELLOW, scale)
    if banner:
        s = analysis.summary()
        text = f"Motorcycles {s['motorcycles']} | Violations {s['violations']} | Safe {s['safe']} | Unknown {s['unknown']}"
        bh = int(36 * scale) + 10
        cv2.rectangle(out, (0, 0), (w, bh), BLACK, -1)
        cv2.putText(out, text, (10, bh - 10), cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, max(1, int(scale * 2)), cv2.LINE_AA)
    return out
