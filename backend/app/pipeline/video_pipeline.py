"""Video processing: sampled frames + motorcycle tracking + vote aggregation.

Why not just run the image pipeline on every frame and report every violation?
* the same rider appears in dozens of frames -> one violation must be reported once;
* single frames are noisy (a head hidden for one frame, a wrong class for one frame).

So motorcycles are tracked (ByteTrack, persistent IDs) and every analysed frame casts a vote
(VIOLATION / SAFE) for its track. A track is reported only after ``min_votes`` decided frames and gets
the majority status. OCR runs on a few of the track's frames and the most frequent valid plate text
(or the highest-confidence read) is kept.

Speed: only ``target_fps`` frames per second are analysed (default 5); the other frames re-use the
last drawing, so the output video keeps its original frame rate and length.
"""
from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from app.config import Settings
from app.pipeline.annotate import draw
from app.pipeline.image_pipeline import HelmetViolationPipeline
from app.pipeline.types import Box, FrameAnalysis, HelmetStatus, Motorcycle, PlateRead
from app.services.media import make_writer, open_video, save_image, to_browser_mp4

log = logging.getLogger(__name__)
MAX_OCR_PER_TRACK = 4


@dataclass
class TrackState:
    track_id: int
    first_frame: int
    last_frame: int
    votes: Counter = field(default_factory=Counter)
    max_riders: int = 0
    plates: list[PlateRead] = field(default_factory=list)
    best_score: float = -1.0                      # confidence of the best violation snapshot so far
    snapshot: str | None = None
    plate_snapshot: str | None = None

    @property
    def decided(self) -> int:
        return self.votes[HelmetStatus.VIOLATION] + self.votes[HelmetStatus.SAFE]

    def status(self, min_votes: int) -> HelmetStatus:
        if self.decided < min_votes:
            return HelmetStatus.UNKNOWN
        v, s = self.votes[HelmetStatus.VIOLATION], self.votes[HelmetStatus.SAFE]
        return HelmetStatus.VIOLATION if v >= s else HelmetStatus.SAFE

    def best_plate(self) -> PlateRead | None:
        valid = [p for p in self.plates if p.text and p.valid_format]
        if valid:
            text = Counter(p.text for p in valid).most_common(1)[0][0]
            return max((p for p in valid if p.text == text), key=lambda p: p.confidence)
        read = [p for p in self.plates if p.text]
        if read:
            return max(read, key=lambda p: p.confidence)
        boxed = [p for p in self.plates if p.box]
        return boxed[-1] if boxed else None


def _scaled(b: Box, f: float) -> Box:
    return Box(b.x1 * f, b.y1 * f, b.x2 * f, b.y2 * f, b.conf, b.label)


class VideoProcessor:
    def __init__(self, pipeline: HelmetViolationPipeline, settings: Settings):
        self.p, self.s = pipeline, settings

    def _save_snapshot(self, frame: np.ndarray, m: Motorcycle, st: TrackState, out_dir: Path,
                       full: np.ndarray, scale: float) -> None:
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = m.box.expanded(0.6, W, H).as_int()
        name = f"violation_track{st.track_id}.jpg"
        save_image(out_dir / name, frame[max(0, y1 - int(0.6 * m.box.h)):y2, x1:x2])
        st.snapshot = name
        if m.plate and m.plate.box:
            # plate crop from the full-resolution frame (sharper than the scaled one)
            FH, FW = full.shape[:2]
            a1, b1, a2, b2 = _scaled(m.plate.box, 1 / scale).expanded(0.08, FW, FH).as_int()
            if a2 > a1 and b2 > b1:
                pname = f"plate_track{st.track_id}.jpg"
                save_image(out_dir / pname, full[b1:b2, a1:a2])
                st.plate_snapshot = pname

    def _read_plate(self, full: np.ndarray, det_img: np.ndarray, work: np.ndarray, m: Motorcycle,
                    scale: float) -> PlateRead:
        """Find + read the plate. With a scaled-down frame the search and OCR run on the full-resolution
        frame (plates are small; scaling would cost characters); the box is returned in scaled coordinates."""
        if scale == 1.0:
            return self.p.read_plate(det_img, m, work)
        plate = self.p.read_plate(full, Motorcycle(box=_scaled(m.box, 1 / scale)))
        if plate.box:
            plate.box = _scaled(plate.box, scale)
        return plate

    def process(self, path: Path, out_dir: Path, progress: Callable[[float], None] | None = None,
                force: frozenset[str] = frozenset()) -> dict:
        cap, fps, n_frames, w, h = open_video(path, self.s.video_max_seconds)
        stride = max(1, round(fps / self.s.video_target_fps))
        scale = min(1.0, self.s.video_max_side / max(w, h))
        ow, oh = (w, h) if scale == 1.0 else (int(round(w * scale)) // 2 * 2, int(round(h * scale)) // 2 * 2)
        raw_out = out_dir / "annotated.mp4"
        writer = make_writer(raw_out, fps, ow, oh)
        self.p.vehicles.reset_tracker()
        tracks: dict[int, TrackState] = {}
        last: FrameAnalysis | None = None
        timeline: list[dict] = []
        t_start = time.perf_counter()
        idx = analysed = 0
        try:
            while True:
                ok, full = cap.read()
                if not ok:
                    break
                frame = full if scale == 1.0 else cv2.resize(full, (ow, oh), interpolation=cv2.INTER_AREA)
                if idx % stride == 0:
                    analysis, work = self.p.analyse(frame, tracked=True, fast_enhance=True, read_plates=False,
                                                    force=force)
                    analysed += 1
                    for m in analysis.motorcycles:
                        if m.track_id is None:
                            continue
                        st = tracks.setdefault(m.track_id, TrackState(m.track_id, idx, idx))
                        st.last_frame = idx
                        st.votes[m.status] += 1
                        st.max_riders = max(st.max_riders, m.rider_count)
                        if m.status == HelmetStatus.VIOLATION:
                            if len(st.plates) < MAX_OCR_PER_TRACK:
                                det_img = work if analysis.enhancement.get("used_for_detection") else frame
                                plate = self._read_plate(full, det_img, work, m, scale)
                                st.plates.append(plate)
                                m.plate = plate
                            else:
                                m.plate = st.best_plate()
                            score = max(hd.conf for hd in m.heads if hd.label == "no_helmet")
                            if score > st.best_score:
                                st.best_score = score
                                self._save_snapshot(work, m, st, out_dir, full, scale)
                        # display the track's aggregated decision, not the single-frame one (no flicker)
                        agg = st.status(self.s.video_min_votes)
                        if agg != HelmetStatus.UNKNOWN:
                            m.status = agg
                        if agg == HelmetStatus.VIOLATION and m.plate is None:
                            m.plate = st.best_plate()
                    s = analysis.summary()
                    timeline.append({"frame": idx, "time_s": round(idx / fps, 2), "motorcycles": s["motorcycles"],
                                     "violations": s["violations"]})
                    last = analysis
                writer.write(draw(frame, last) if last is not None else frame)
                idx += 1
                if progress and n_frames and idx % 10 == 0:
                    progress(min(0.99, idx / n_frames))
        finally:
            cap.release()
            writer.release()
        video_file = to_browser_mp4(raw_out).name

        events = []
        for st in sorted(tracks.values(), key=lambda t: t.first_frame):
            status = st.status(self.s.video_min_votes)
            if status == HelmetStatus.UNKNOWN:
                continue
            plate = st.best_plate() if status == HelmetStatus.VIOLATION else None
            events.append({
                "track_id": st.track_id, "status": status.value,
                "start_s": round(st.first_frame / fps, 2), "end_s": round(st.last_frame / fps, 2),
                "votes": {k.value: v for k, v in st.votes.items()}, "rider_count": st.max_riders,
                "triple_riding": st.max_riders >= 3, "plate": plate.to_dict() if plate else None,
                "snapshot": st.snapshot if status == HelmetStatus.VIOLATION else None,
                "plate_snapshot": st.plate_snapshot if status == HelmetStatus.VIOLATION else None})
        viol = [e for e in events if e["status"] == "VIOLATION"]
        elapsed = time.perf_counter() - t_start
        if progress:
            progress(1.0)
        return {
            "video": {"fps": round(fps, 2), "frames": idx, "width": w, "height": h,
                      "processed_width": ow, "processed_height": oh,
                      "duration_s": round(idx / fps, 2), "analysed_frames": analysed, "stride": stride},
            "summary": {"motorcycles": len(events), "violations": len(viol),
                        "safe": sum(e["status"] == "SAFE" for e in events),
                        "plates_read": sum(1 for e in viol if e["plate"] and e["plate"]["text"]),
                        "processing_s": round(elapsed, 1),
                        "speed_fps": round(idx / max(elapsed, 1e-6), 1)},
            "events": events, "timeline": timeline, "annotated_video": video_file,
        }
