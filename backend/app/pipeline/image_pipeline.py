"""The complete per-image pipeline.

    image -> quality check + enhancement -> COCO YOLO (motorcycles, persons) -> helmet model (heads)
          -> assign heads/riders to motorcycles -> SAFE / VIOLATION / UNKNOWN
          -> for each VIOLATION: crop motorcycle (+15%) -> plate model -> best plate -> PaddleOCR

All models are created once in ``__init__`` (loading takes seconds; inference takes milliseconds).
The class is used directly by the CLI, by the video processor (per frame) and by the API.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from app.config import Settings
from app.core.device import select_device
from app.pipeline.association import assign, dedupe_bikes, head_only_riders
from app.pipeline.detectors import COCO_MOTORCYCLE, COCO_PERSON, YoloDetector, split_track_id
from app.pipeline.enhancement import ImageEnhancer
from app.pipeline.ocr import PlateReader
from app.pipeline.types import Box, FrameAnalysis, HelmetStatus, Motorcycle, PlateRead

log = logging.getLogger(__name__)


class HelmetViolationPipeline:
    def __init__(self, settings: Settings, load_ocr: bool = True):
        self.s = settings
        self.device = select_device(settings.device)
        log.info("inference device: %s", self.device.name)
        self.enhancer = ImageEnhancer(settings.enhance_mode)
        self.vehicles = YoloDetector(settings.vehicle_model, settings.vehicle_imgsz, settings.vehicle_conf,
                                     settings.iou, self.device, classes=[COCO_PERSON, COCO_MOTORCYCLE],
                                     allow_download=True)
        self.helmets = YoloDetector(settings.helmet_model, settings.helmet_imgsz, settings.helmet_conf,
                                    settings.iou, self.device)
        self.plates = YoloDetector(settings.plate_model, settings.plate_imgsz, settings.plate_conf,
                                   settings.iou, self.device)
        self.reader = PlateReader(settings.ocr_lang, settings.ocr_min_height,
                                  use_gpu=self.device.device != "cpu") if load_ocr else None

    # -------------------------------------------------------------------------------------------
    def find_plate(self, image: np.ndarray, moto: Motorcycle) -> Box | None:
        """Search the motorcycle crop (same +15% padding as in training) for its plate."""
        H, W = image.shape[:2]
        crop_box = moto.box.expanded(self.s.motorcycle_crop_pad, W, H)
        x1, y1, x2, y2 = crop_box.as_int()
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        plates = self.plates.predict([crop], offsets=[(x1, y1)])[0]
        return max(plates, key=lambda p: p.conf) if plates else None

    def read_plate(self, image: np.ndarray, moto: Motorcycle, enhanced: np.ndarray | None = None) -> PlateRead:
        """Find and read the plate. If the read is not a valid number and an enhanced version of the
        image exists, OCR is retried on the enhanced crop (helps with dark / low-contrast plates)."""
        box = self.find_plate(image, moto)
        if box is None:
            return PlateRead()
        if self.reader is None:
            return PlateRead(box=box)
        read = self.reader.read(image, box, self.s.plate_crop_pad)
        if not read.valid_format and enhanced is not None and enhanced is not image:
            retry = self.reader.read(enhanced, box, self.s.plate_crop_pad)
            if retry.valid_format or (retry.text and not read.text):
                return retry
        return read

    def analyse(self, image: np.ndarray, tracked: bool = False, fast_enhance: bool = False,
                read_plates: bool = True, force: frozenset[str] = frozenset(),
                all_plates: bool = False) -> tuple[FrameAnalysis, np.ndarray]:
        """Run the pipeline on one BGR image. ``force`` = enhancement steps requested by the user.
        Plates are read for violations only, or for every motorcycle when ``all_plates`` is set.

        Returns (analysis, display image) - the display image is the enhanced image when enhancement
        changed something (boxes are drawn on it), otherwise the original.
        """
        t = {}
        t0 = time.perf_counter()
        enhanced, enh = self.enhancer(image, fast=fast_enhance, force=force)
        # user-requested steps (e.g. rain removal) are meant to change what the models see
        use_enhanced = self.s.enhance_for_detection or any(
            step in enh["applied"] for step in (*self.s.detect_on_enhanced_if, *force))
        enh["used_for_detection"] = use_enhanced
        t["enhancement"] = (time.perf_counter() - t0) * 1000
        det_img = enhanced if use_enhanced else image

        h, w = image.shape[:2]
        t0 = time.perf_counter()
        vsz = self.s.vehicle_small_imgsz if max(h, w) <= self.s.vehicle_small_side else self.s.vehicle_imgsz
        vehicles = (self.vehicles.track(det_img, imgsz=vsz) if tracked
                    else self.vehicles.predict([det_img], imgsz=vsz)[0])
        bikes = [b for b in vehicles if split_track_id(b.label)[0] == "motorcycle"]
        bikes = dedupe_bikes(bikes, self.s.bike_dup_iou, self.s.bike_dup_inside)
        persons = [b for b in vehicles if split_track_id(b.label)[0] == "person"]
        heads = self.helmets.predict([det_img])[0]
        t["detection"] = (time.perf_counter() - t0) * 1000

        if not bikes and heads and self.s.head_only_fallback and not tracked:
            motos, unassigned = head_only_riders(persons, heads, w, h), []
        else:
            motos, unassigned = assign(bikes, persons, heads, self.s.head_search_up, self.s.head_search_side)
        for m in motos:
            name, tid = split_track_id(m.box.label)
            m.box.label, m.track_id = name, tid

        t0 = time.perf_counter()
        if read_plates:
            for m in motos:
                if all_plates or m.status == HelmetStatus.VIOLATION:
                    m.plate = self.read_plate(det_img, m, enhanced)
            if all_plates and not any(m.plate and m.plate.box for m in motos):
                # close-up of a plate / rear of a bike: no motorcycle box to search -> search the whole image
                for pb in self.plates.predict([det_img])[0]:
                    pr = self.reader.read(det_img, pb, self.s.plate_crop_pad) if self.reader else PlateRead(box=pb)
                    bike = Box(pb.x1 - pb.w, pb.y1 - 2 * pb.h, pb.x2 + pb.w, pb.y2 + pb.h, pb.conf, "motorcycle")
                    motos.append(Motorcycle(box=bike.expanded(0, w, h), plate=pr, bike_detected=False))
        t["plate_ocr"] = (time.perf_counter() - t0) * 1000

        return FrameAnalysis(w, h, motos, unassigned, enh, t), enhanced
