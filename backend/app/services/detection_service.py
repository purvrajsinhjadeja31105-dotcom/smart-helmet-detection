"""Runs a detection job end-to-end and stores every artefact.

For each run a unique ``run_id`` is created and these files are written:
    outputs/<run_id>/annotated.jpg | annotated.mp4    processed media with boxes
    outputs/<run_id>/enhanced.jpg                     only if enhancement changed the image
    outputs/<run_id>/violation_*.jpg, plate_*.jpg     evidence crops for each violation
    reports/<run_id>.json                             the violation report (also returned to the caller)
The database layer (Phase 3) will store the report JSON; nothing here depends on it.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from app import __version__
from app.config import Settings
from app.pipeline.annotate import draw
from app.pipeline.image_pipeline import HelmetViolationPipeline
from app.pipeline.types import HelmetStatus
from app.pipeline.video_pipeline import VideoProcessor
from app.services.media import load_image, media_kind, save_image

log = logging.getLogger(__name__)


class DetectionService:
    def __init__(self, settings: Settings, pipeline: HelmetViolationPipeline | None = None):
        self.s = settings
        self.pipeline = pipeline or HelmetViolationPipeline(settings)
        self.video = VideoProcessor(self.pipeline, settings)
        # one job at a time on the models: GPU memory and the video tracker state are shared
        self._lock = threading.Lock()

    def _new_run(self) -> tuple[str, Path]:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        out = self.s.output_dir / run_id
        out.mkdir(parents=True, exist_ok=True)
        return run_id, out

    def _base(self, run_id: str, source: Path, kind: str) -> dict:
        return {"run_id": run_id, "type": kind, "source_file": source.name, "options": {},
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "device": self.pipeline.device.name, "app_version": __version__,
                "thresholds": {"helmet_conf": self.s.helmet_conf, "plate_conf": self.s.plate_conf,
                               "vehicle_conf": self.s.vehicle_conf}}

    def _write_report(self, report: dict) -> None:
        path = self.s.report_dir / f"{report['run_id']}.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # -------------------------------------------------------------------------------------------
    def process_image(self, path: Path, force: frozenset[str] = frozenset(), all_plates: bool = False) -> dict:
        """``force``: enhancement steps requested by the user, e.g. frozenset({"rain"}).
        ``all_plates``: read the plate of every motorcycle, not only of violators."""
        img = load_image(path)
        run_id, out = self._new_run()
        t0 = time.perf_counter()
        with self._lock:
            analysis, work = self.pipeline.analyse(img, force=force, all_plates=all_plates)
        save_image(out / "annotated.jpg", draw(work, analysis))
        files = {"annotated": "annotated.jpg"}
        if analysis.enhancement.get("applied"):
            save_image(out / "enhanced.jpg", work)
            files["enhanced"] = "enhanced.jpg"

        violations = []
        H, W = work.shape[:2]
        for i, m in enumerate(analysis.motorcycles):
            if m.status != HelmetStatus.VIOLATION:
                continue
            x1, y1, x2, y2 = m.box.expanded(0.3, W, H).as_int()
            top = max(0, y1 - int(0.8 * m.box.h))                    # include the riders above the bike
            crop_name = f"violation_{i}.jpg"
            save_image(out / crop_name, work[top:y2, x1:x2])
            plate_name = None
            if m.plate and m.plate.box:
                a1, b1, a2, b2 = m.plate.box.expanded(0.08, W, H).as_int()
                if a2 > a1 and b2 > b1:
                    plate_name = f"plate_{i}.jpg"
                    save_image(out / plate_name, work[b1:b2, a1:a2])
            no_helmet = [h.conf for h in m.heads if h.label == "no_helmet"]
            violations.append({
                "violation_id": f"{run_id}-{i}", "motorcycle_index": i, "rider_count": m.rider_count,
                "triple_riding": m.rider_count >= 3, "no_helmet_confidence": round(max(no_helmet), 4),
                "plate_number": m.plate.text if m.plate else "",
                "plate_valid_format": bool(m.plate and m.plate.valid_format),
                "ocr_confidence": round(m.plate.confidence, 4) if m.plate else 0.0,
                "plate_detected": bool(m.plate and m.plate.box),
                "evidence_image": crop_name, "plate_image": plate_name})

        # every plate that was read (with all_plates: also SAFE / UNKNOWN motorcycles)
        plates = []
        for i, m in enumerate(analysis.motorcycles):
            if not (m.plate and m.plate.box):
                continue
            a1, b1, a2, b2 = m.plate.box.expanded(0.08, W, H).as_int()
            name = f"plate_{i}.jpg"
            if a2 > a1 and b2 > b1 and not (out / name).exists():
                save_image(out / name, work[b1:b2, a1:a2])
            plates.append({"motorcycle_index": i, "status": m.status.value, "plate_number": m.plate.text,
                           "plate_valid_format": m.plate.valid_format,
                           "ocr_confidence": round(m.plate.confidence, 4),
                           "plate_confidence": round(m.plate.box.conf, 4),
                           "plate_image": name if (out / name).exists() else None})

        report = self._base(run_id, path, "image")
        report["options"] = {"requested_enhancements": sorted(force), "read_all_plates": all_plates}
        report.update(processing_ms=round((time.perf_counter() - t0) * 1000, 1), files=files,
                      violations=violations, plates=plates, **analysis.to_dict())
        self._write_report(report)
        log.info("image %s: %s", path.name, analysis.summary())
        return report

    def process_video(self, path: Path, progress: Callable[[float], None] | None = None,
                      force: frozenset[str] = frozenset()) -> dict:
        run_id, out = self._new_run()
        t0 = time.perf_counter()
        with self._lock:
            result = self.video.process(path, out, progress, force)
        violations = [{
            "violation_id": f"{run_id}-t{e['track_id']}", "track_id": e["track_id"],
            "start_s": e["start_s"], "end_s": e["end_s"], "rider_count": e["rider_count"],
            "triple_riding": e["triple_riding"],
            "plate_number": (e["plate"] or {}).get("text", ""),
            "plate_valid_format": bool((e["plate"] or {}).get("valid_format")),
            "ocr_confidence": (e["plate"] or {}).get("confidence", 0.0),
            "plate_detected": bool((e["plate"] or {}).get("box")),
            "evidence_image": e["snapshot"], "plate_image": e["plate_snapshot"]}
            for e in result["events"] if e["status"] == "VIOLATION"]
        report = self._base(run_id, path, "video")
        report["options"] = {"requested_enhancements": sorted(force)}
        report.update(processing_ms=round((time.perf_counter() - t0) * 1000, 1),
                      files={"annotated": result.pop("annotated_video")}, violations=violations, **result)
        self._write_report(report)
        log.info("video %s: %s", path.name, report["summary"])
        return report

    def process(self, path: Path, progress: Callable[[float], None] | None = None,
                force: frozenset[str] = frozenset()) -> dict:
        if media_kind(path) == "image":
            return self.process_image(path, force)
        return self.process_video(path, progress, force)
