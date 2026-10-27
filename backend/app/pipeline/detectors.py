"""YOLO model wrappers. Models are loaded once and reused for every request / frame."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.core.device import DeviceInfo
from app.core.exceptions import ModelLoadError
from app.pipeline.types import Box

log = logging.getLogger(__name__)

COCO_PERSON, COCO_MOTORCYCLE = 0, 3


class YoloDetector:
    """Thin wrapper around an Ultralytics YOLO model.

    Args:
        weights: path to a ``.pt`` file. For the COCO model a missing file is downloaded automatically.
        imgsz:   inference size; must match the training size for best accuracy.
        conf:    minimum confidence of returned boxes.
        iou:     NMS IoU threshold (YOLO26 is NMS-free; kept for YOLO11 compatibility).
        device:  result of :func:`app.core.device.select_device`.
        classes: optional list of class ids to keep (e.g. COCO person + motorcycle).
        allow_download: only the generic COCO model may be fetched from the internet.
    """

    def __init__(self, weights: Path, imgsz: int, conf: float, iou: float, device: DeviceInfo,
                 classes: list[int] | None = None, allow_download: bool = False):
        from ultralytics import YOLO
        weights = Path(weights)
        if not weights.exists() and not allow_download:
            raise ModelLoadError(f"model file not found: {weights}")
        weights.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.model = YOLO(str(weights))
        except Exception as e:
            raise ModelLoadError(f"cannot load {weights}: {e}") from e
        self.names: dict[int, str] = self.model.names
        self.imgsz, self.conf, self.iou, self.device, self.classes = imgsz, conf, iou, device, classes
        log.info("loaded %s (%s) on %s", weights.name, ", ".join(self.names.values())[:60], device.name)

    def _kwargs(self, imgsz: int | None = None) -> dict:
        kw = dict(imgsz=imgsz or self.imgsz, conf=self.conf, iou=self.iou, device=self.device.device,
                  classes=self.classes, verbose=False)
        if self.device.half:          # FP16 only on CUDA (passing it on CPU triggers a deprecation warning)
            kw["half"] = True
        return kw

    @staticmethod
    def _to_boxes(result, names: dict[int, str], offset: tuple[float, float] = (0, 0)) -> list[Box]:
        out = []
        if result.boxes is None or len(result.boxes) == 0:
            return out
        xyxy = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)
        ids = result.boxes.id.cpu().numpy().astype(int) if getattr(result.boxes, "id", None) is not None else None
        ox, oy = offset
        for i, ((x1, y1, x2, y2), c, k) in enumerate(zip(xyxy, conf, cls)):
            # plain Python floats: numpy float32 is not JSON serialisable
            b = Box(float(x1) + ox, float(y1) + oy, float(x2) + ox, float(y2) + oy, float(c), names[int(k)])
            if ids is not None:
                b.label = f"{names[k]}#{ids[i]}"
            out.append(b)
        return out

    def predict(self, images: list[np.ndarray], offsets: list[tuple[float, float]] | None = None,
                imgsz: int | None = None) -> list[list[Box]]:
        """Detect objects in a batch of BGR images. ``offsets`` shift crop boxes back to full-image coordinates."""
        if not images:
            return []
        results = self.model.predict(images, **self._kwargs(imgsz))
        offsets = offsets or [(0, 0)] * len(images)
        return [self._to_boxes(r, self.names, o) for r, o in zip(results, offsets)]

    def track(self, frame: np.ndarray, tracker: str = "bytetrack.yaml", imgsz: int | None = None) -> list[Box]:
        """Detect + track (persistent IDs across calls) - used for video. Labels become 'name#id'."""
        result = self.model.track(frame, persist=True, tracker=tracker, **self._kwargs(imgsz))[0]
        return self._to_boxes(result, self.names)

    def reset_tracker(self) -> None:
        """Forget track IDs (call before processing a new video)."""
        predictor = getattr(self.model, "predictor", None)
        if predictor is not None and getattr(predictor, "trackers", None):
            for t in predictor.trackers:
                t.reset()


def split_track_id(label: str) -> tuple[str, int | None]:
    """'motorcycle#12' -> ('motorcycle', 12)."""
    if "#" in label:
        name, tid = label.rsplit("#", 1)
        return name, int(tid)
    return label, None
