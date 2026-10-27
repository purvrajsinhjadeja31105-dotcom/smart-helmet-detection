"""Central configuration.

Every tunable value lives here so behaviour is reproducible and can be changed without touching code.
Any field can be overridden with an environment variable named ``HELMET_<FIELD_NAME>``
(e.g. ``HELMET_DEVICE=cpu`` or ``HELMET_HELMET_CONF=0.45``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    # ---- model files (Phase 1 outputs) -------------------------------------------------------
    helmet_model: Path = PROJECT_ROOT / "models" / "helmet" / "helmet_best.pt"
    plate_model: Path = PROJECT_ROOT / "models" / "plate" / "plate_crop_best.pt"
    # COCO-pretrained detector for motorcycles and persons (downloaded automatically on first use)
    vehicle_model: Path = PROJECT_ROOT / "models" / "coco" / "yolo26m.pt"

    # ---- inference sizes: must match training ---------------------------------------------------
    helmet_imgsz: int = 960
    plate_imgsz: int = 640
    vehicle_imgsz: int = 960
    # COCO was trained at 640: upscaling a small image (<= this side) to 960 made the detector miss
    # bikes and split one bike into several weak boxes (manual demo test, 19 Oct 2026), so small
    # images run at 640.
    vehicle_small_side: int = 960
    vehicle_small_imgsz: int = 640

    # ---- thresholds (deployment thresholds = max-F1 points from Phase 1 evaluation) -------------
    helmet_conf: float = 0.40
    plate_conf: float = 0.37
    vehicle_conf: float = 0.30
    iou: float = 0.70

    # ---- geometry used by the pipeline ---------------------------------------------------------
    motorcycle_crop_pad: float = 0.15    # same padding as the plate model's training crops
    plate_crop_pad: float = 0.08         # small margin around the plate before OCR
    head_search_up: float = 1.4          # heads may sit up to 1.4x bike-height above the bike top
    head_search_side: float = 0.25       # ...and 25% of bike-width beyond its left/right edges
    bike_dup_iou: float = 0.55           # two motorcycle boxes overlapping more than this = same bike
    bike_dup_inside: float = 0.85        # ...or a box lying this much inside a stronger one
    # no motorcycle found at all but heads are (close-ups, cropped web photos): judge the riders
    # from their heads instead of discarding them
    head_only_fallback: bool = True

    # ---- OCR ----------------------------------------------------------------------------------
    ocr_min_height: int = 96             # plate crops are upscaled to at least this height
    ocr_lang: str = "en"

    # ---- enhancement (applied only when the quality check says the image needs it) -------------
    enhance_mode: str = "auto"           # "auto" | "always" | "off"
    # Run the detectors on the enhanced image? Measured on 107 unseen images with synthetic
    # fog / low light / blur / noise: enhancement did not raise helmet mAP50 (fog 0.852 -> 0.849,
    # low light 0.830 -> 0.805), because the models were trained with weather augmentation.
    # So detection uses the original; the enhanced image is shown to the user and used as a
    # second OCR attempt for unreadable plates.
    enhance_for_detection: bool = False
    # Exception - heavy fog: a demo with 70-85% synthetic fog lost 1 of 5 motorcycles, and dehazing
    # recovered it (contrast 21 -> restored). When one of these steps was applied, detection runs on
    # the enhanced image even if enhance_for_detection is False.
    detect_on_enhanced_if: tuple = ("fog",)

    # ---- video ----------------------------------------------------------------------------------
    video_target_fps: float = 5.0        # frames analysed per second of video (others re-use results)
    video_max_seconds: float = 600.0     # refuse longer uploads (protects the server)
    video_min_votes: int = 2             # frames a motorcycle must be seen in before it is reported
    # 4K video took ~8 min for 15 s on CPU: frames are scaled to this long side for detection, drawing
    # and the output video (the models run at 960 anyway); plates are still read from the full frame
    video_max_side: int = 1920

    # ---- runtime --------------------------------------------------------------------------------
    device: str = "auto"                 # "auto" (GPU if available, else CPU) | "cpu" | "0"
    max_upload_mb: int = 500

    # ---- storage --------------------------------------------------------------------------------
    upload_dir: Path = PROJECT_ROOT / "uploads"
    output_dir: Path = PROJECT_ROOT / "outputs"
    report_dir: Path = PROJECT_ROOT / "reports"
    log_dir: Path = PROJECT_ROOT / "logs"

    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings, letting ``HELMET_*`` environment variables override the defaults."""
        s = cls()
        for f in fields(cls):
            raw = os.environ.get(f"HELMET_{f.name.upper()}")
            if raw is None or f.name == "extra":
                continue
            current = getattr(s, f.name)
            if isinstance(current, Path):
                setattr(s, f.name, Path(raw))
            elif isinstance(current, bool):
                setattr(s, f.name, raw.lower() in {"1", "true", "yes"})
            elif isinstance(current, int):
                setattr(s, f.name, int(raw))
            elif isinstance(current, float):
                setattr(s, f.name, float(raw))
            else:
                setattr(s, f.name, raw)
        for d in (s.upload_dir, s.output_dir, s.report_dir, s.log_dir):
            d.mkdir(parents=True, exist_ok=True)
        return s


_settings: Settings | None = None


def get_settings() -> Settings:
    """Process-wide singleton (FastAPI dependency and pipeline both use it)."""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings
