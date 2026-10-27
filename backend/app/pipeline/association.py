"""Link detected heads and persons to motorcycles, then decide SAFE / VIOLATION / UNKNOWN.

Algorithm (per image):
1. Riders: a COCO *person* is a rider of the motorcycle whose box it overlaps most (at least 30% of
   the person box must overlap the motorcycle box expanded upwards to cover the rider's body).
2. Heads: a head is assigned through its rider (head centre inside the rider's box) when possible;
   otherwise geometrically - the head centre must lie above the motorcycle's bottom edge, at most
   ``head_search_up`` bike-heights above its top, and horizontally within the bike (+ margin).
   Among candidates the closest motorcycle (normalised horizontal distance) wins.
3. Decision per motorcycle: any ``no_helmet`` head -> VIOLATION; otherwise any ``helmet`` -> SAFE;
   no head at all -> UNKNOWN (never guessed - a wrong VIOLATION is worse than no decision).
Heads not assigned to any motorcycle (pedestrians, people standing nearby) never cause violations.
"""
from __future__ import annotations

from app.pipeline.types import Box, HelmetStatus, Motorcycle


def _inter(a: Box, b: Box) -> float:
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    return ix * iy


def _rider_zone(bike: Box, up: float) -> Box:
    """Region where a rider's body can be: the bike box extended upwards."""
    return Box(bike.x1, bike.y1 - up * bike.h, bike.x2, bike.y2)


def dedupe_bikes(bikes: list[Box], iou: float = 0.55, inside: float = 0.85) -> list[Box]:
    """Drop motorcycle boxes that duplicate a stronger one (high IoU, or lying almost entirely inside it).
    YOLO26 is NMS-free, so on hard images one bike can come out as several overlapping weak boxes."""
    keep: list[Box] = []
    for b in sorted(bikes, key=lambda x: x.conf, reverse=True):
        dup = False
        for k in keep:
            i = _inter(b, k)
            if i / max(b.area + k.area - i, 1e-6) > iou or i / max(b.area, 1e-6) > inside:
                dup = True
                break
        if not dup:
            keep.append(b)
    return keep


def head_only_riders(persons: list[Box], heads: list[Box], img_w: int, img_h: int) -> list[Motorcycle]:
    """Fallback when the vehicle detector found no motorcycle at all but the helmet model found heads.

    The helmet model was trained on rider heads only, so each head (or the heads inside one person box)
    is treated as a rider. The 'motorcycle' box is the person's box, or an estimate below the head, so
    the plate model can still search it. Marked ``bike_detected=False`` in the report."""
    groups: dict[int, list[Box]] = {}
    for hi, h in enumerate(heads):
        owner = next((pi for pi, p in enumerate(persons)
                      if p.x1 <= h.cx <= p.x2 and p.y1 - 0.1 * p.h <= h.cy <= p.y1 + 0.5 * p.h), None)
        groups.setdefault(-1 - hi if owner is None else owner, []).append(h)
    out = []
    for key, hs in groups.items():
        if key >= 0:
            p = persons[key]
            box = Box(p.x1, p.y1, p.x2, min(float(img_h), p.y2 + 0.15 * p.h), p.conf, "motorcycle")
        else:
            h = hs[0]
            box = Box(max(0.0, h.x1 - h.w), h.y1, min(float(img_w), h.x2 + h.w),
                      min(float(img_h), h.y2 + 5 * h.h), h.conf, "motorcycle")
        m = Motorcycle(box=box, heads=hs, riders=[persons[key]] if key >= 0 else [], bike_detected=False)
        labels = {x.label for x in hs}
        m.status = HelmetStatus.VIOLATION if "no_helmet" in labels else HelmetStatus.SAFE
        out.append(m)
    return out


def assign(bikes: list[Box], persons: list[Box], heads: list[Box], head_search_up: float = 1.4,
           head_search_side: float = 0.25) -> tuple[list[Motorcycle], list[Box]]:
    """Return (motorcycles with riders/heads/status, heads that belong to no motorcycle)."""
    motos = [Motorcycle(box=b) for b in bikes]
    rider_of: dict[int, int] = {}                    # person index -> motorcycle index

    for pi, p in enumerate(persons):
        best, best_frac = None, 0.3
        for mi, m in enumerate(motos):
            frac = _inter(p, _rider_zone(m.box, head_search_up)) / max(p.area, 1e-6)
            # the rider's feet are on the bike: person bottom must be inside the bike's vertical span
            if frac > best_frac and m.box.y1 - 0.2 * m.box.h <= p.y2 <= m.box.y2 + 0.2 * m.box.h:
                best, best_frac = mi, frac
        if best is not None:
            motos[best].riders.append(p)
            rider_of[pi] = best

    unassigned: list[Box] = []
    for h in heads:
        target = None
        # (a) through a rider whose box contains the head centre
        for pi, mi in rider_of.items():
            p = persons[pi]
            if p.x1 <= h.cx <= p.x2 and p.y1 - 0.1 * p.h <= h.cy <= p.y1 + 0.5 * p.h:
                target = mi
                break
        # (b) geometric fallback
        if target is None:
            best_d = None
            for mi, m in enumerate(motos):
                b = m.box
                if not (b.x1 - head_search_side * b.w <= h.cx <= b.x2 + head_search_side * b.w):
                    continue
                if not (b.y1 - head_search_up * b.h <= h.cy <= b.y2):
                    continue
                d = abs(h.cx - b.cx) / max(b.w, 1e-6)
                if best_d is None or d < best_d:
                    target, best_d = mi, d
        if target is None:
            unassigned.append(h)
        else:
            motos[target].heads.append(h)

    for m in motos:
        labels = {h.label for h in m.heads}
        m.status = (HelmetStatus.VIOLATION if "no_helmet" in labels else
                    HelmetStatus.SAFE if "helmet" in labels else HelmetStatus.UNKNOWN)
    return motos, unassigned
