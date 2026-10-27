"""Plain data structures passed between pipeline stages (no logic, easy to serialise to JSON)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class HelmetStatus(str, Enum):
    SAFE = "SAFE"                # every detected head on the motorcycle wears a helmet
    VIOLATION = "VIOLATION"      # at least one rider without a helmet
    UNKNOWN = "UNKNOWN"          # motorcycle found but no head visible (e.g. seen from far, occluded)


@dataclass
class Box:
    """Axis-aligned box in pixel coordinates of the full image."""
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 1.0
    label: str = ""

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def expanded(self, pad: float, img_w: int, img_h: int) -> "Box":
        """Box grown by ``pad`` x width/height on every side, clipped to the image."""
        dx, dy = self.w * pad, self.h * pad
        return Box(max(0.0, self.x1 - dx), max(0.0, self.y1 - dy), min(float(img_w), self.x2 + dx),
                   min(float(img_h), self.y2 + dy), self.conf, self.label)

    def as_int(self) -> tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)

    def to_dict(self) -> dict:
        return {"x1": round(self.x1, 1), "y1": round(self.y1, 1), "x2": round(self.x2, 1),
                "y2": round(self.y2, 1), "conf": round(self.conf, 4), "label": self.label}


@dataclass
class PlateRead:
    text: str = ""               # normalised plate number, e.g. "MH12AB1234" ("" if unreadable)
    raw_text: str = ""           # exactly what OCR returned
    confidence: float = 0.0      # mean OCR confidence of the used text lines
    valid_format: bool = False   # matches the Indian registration-number format
    box: Box | None = None       # plate location in the full image

    def to_dict(self) -> dict:
        d = asdict(self)
        d["box"] = self.box.to_dict() if self.box else None
        d["confidence"] = round(self.confidence, 4)
        return d


@dataclass
class Motorcycle:
    box: Box
    heads: list[Box] = field(default_factory=list)
    riders: list[Box] = field(default_factory=list)
    status: HelmetStatus = HelmetStatus.UNKNOWN
    plate: PlateRead | None = None
    track_id: int | None = None  # set in video mode
    bike_detected: bool = True   # False = rider judged from heads only (vehicle detector found no bike)

    @property
    def rider_count(self) -> int:
        return max(len(self.heads), len(self.riders))

    def to_dict(self) -> dict:
        return {"track_id": self.track_id, "status": self.status.value, "bike_detected": self.bike_detected,
                "box": self.box.to_dict(),
                "rider_count": self.rider_count, "triple_riding": self.rider_count >= 3,
                "heads": [h.to_dict() for h in self.heads], "riders": [r.to_dict() for r in self.riders],
                "plate": self.plate.to_dict() if self.plate else None}


@dataclass
class FrameAnalysis:
    """Everything found in one image / video frame."""
    width: int
    height: int
    motorcycles: list[Motorcycle] = field(default_factory=list)
    unassigned_heads: list[Box] = field(default_factory=list)   # heads not on any motorcycle
    enhancement: dict = field(default_factory=dict)
    timings_ms: dict = field(default_factory=dict)

    @property
    def violations(self) -> list[Motorcycle]:
        return [m for m in self.motorcycles if m.status == HelmetStatus.VIOLATION]

    def summary(self) -> dict:
        count = lambda s: sum(m.status == s for m in self.motorcycles)
        # every rider judged is on a motorcycle, even if the vehicle detector missed the bike itself
        return {"motorcycles": len(self.motorcycles),
                "riders_without_bike_box": sum(not m.bike_detected for m in self.motorcycles),
                "violations": count(HelmetStatus.VIOLATION),
                "safe": count(HelmetStatus.SAFE), "unknown": count(HelmetStatus.UNKNOWN),
                "plates_read": sum(1 for m in self.violations if m.plate and m.plate.text)}

    def to_dict(self) -> dict:
        return {"width": self.width, "height": self.height, "summary": self.summary(),
                "motorcycles": [m.to_dict() for m in self.motorcycles],
                "unassigned_heads": [h.to_dict() for h in self.unassigned_heads],
                "enhancement": self.enhancement,
                "timings_ms": {k: round(v, 1) for k, v in self.timings_ms.items()}}
