"""Fast unit tests (no models needed): plate text repair, head/rider association, enhancement."""
import cv2
import numpy as np
import pytest

from app.pipeline.association import assign, dedupe_bikes, head_only_riders
from app.pipeline.enhancement import ImageEnhancer, analyse
from app.pipeline.plate_text import clean, normalise
from app.pipeline.types import Box, HelmetStatus


# ---------------------------------------------------------------- plate text
@pytest.mark.parametrize("raw, expected, valid", [
    ("GJ03 BZ0425", "GJ03BZ0425", True),          # two OCR lines joined
    ('"GJ03BZ3955', "GJ03BZ3955", True),           # stray quote from OCR
    ("-GJ03AZ6463", "GJ03AZ6463", True),
    ("MH12AB1234", "MH12AB1234", True),
    ("MH I2 AB I234", "MH12AB1234", True),        # I read instead of 1 in digit positions
    ("KA05M12", "KA05M12", False),                # too short -> not a valid plate
    ("KA05MO123", "KA05M0123", True),             # 'O' in a digit position is repaired to '0'
    ("DL3CAB1234", "DL3CAB1234", True),           # 'DL 3C' district format
    ("IND TS07CC4170", "TS07CC4170", True),       # 'IND' mark on HSRP plates is removed
    ("22BH1234AA", "22BH1234AA", True),           # Bharat series
    ("5S12AB1234", "5S12AB1234", False),          # not a state code
])
def test_plate_normalise(raw, expected, valid):
    assert normalise(raw) == (expected, valid)


def test_clean_keeps_only_alnum():
    assert clean("tn-09 cr 7907.") == "TN09CR7907"


# ---------------------------------------------------------------- association
def test_head_on_bike_decides_status():
    bike = Box(100, 300, 200, 450, 0.9, "motorcycle")
    no_helmet = Box(130, 180, 170, 220, 0.8, "no_helmet")       # above the bike -> rider's head
    pedestrian = Box(600, 100, 640, 140, 0.9, "no_helmet")      # far away -> not a violation
    motos, unassigned = assign([bike], [], [no_helmet, pedestrian])
    assert motos[0].status == HelmetStatus.VIOLATION
    assert unassigned == [pedestrian]


def test_safe_and_unknown():
    b1, b2 = Box(0, 300, 100, 400, label="motorcycle"), Box(500, 300, 600, 400, label="motorcycle")
    motos, _ = assign([b1, b2], [], [Box(30, 200, 60, 240, 0.9, "helmet")])
    assert [m.status for m in motos] == [HelmetStatus.SAFE, HelmetStatus.UNKNOWN]


def test_pillion_without_helmet_is_violation():
    bike = Box(100, 300, 220, 450, label="motorcycle")
    heads = [Box(120, 190, 150, 225, 0.9, "helmet"), Box(160, 185, 190, 222, 0.7, "no_helmet")]
    motos, _ = assign([bike], [], heads)
    assert motos[0].status == HelmetStatus.VIOLATION and motos[0].rider_count == 2


def test_head_goes_to_nearest_bike():
    left, right = Box(0, 300, 100, 400, label="motorcycle"), Box(110, 300, 210, 400, label="motorcycle")
    motos, _ = assign([left, right], [], [Box(140, 200, 170, 240, 0.9, "helmet")])
    assert motos[1].status == HelmetStatus.SAFE and motos[0].status == HelmetStatus.UNKNOWN


# ---------------------------------------------------------------- enhancement
def _scene() -> np.ndarray:
    rng = np.random.default_rng(0)
    img = rng.integers(40, 220, (480, 640, 3), dtype=np.uint8)
    return cv2.GaussianBlur(img, (0, 0), 1.0)


def test_dark_image_is_brightened():
    dark = (_scene() * 0.2).astype(np.uint8)
    assert analyse(dark).is_dark
    out, rep = ImageEnhancer("auto")(dark)
    assert "low_light" in rep["applied"] and out.mean() > dark.mean() * 1.5


def test_foggy_image_is_dehazed():
    img = _scene()
    fog = (img * 0.4 + 200 * 0.6).astype(np.uint8)
    assert analyse(fog).is_hazy
    out, rep = ImageEnhancer("auto")(fog)
    assert "fog" in rep["applied"] and out.std() > fog.std()


def test_off_mode_returns_input_untouched():
    img = _scene()
    out, rep = ImageEnhancer("off")(img)
    assert out is img and rep["applied"] == []


def test_rain_runs_only_when_requested():
    img = _scene()
    _, auto = ImageEnhancer("auto")(img)
    _, forced = ImageEnhancer("auto")(img, force=frozenset({"rain"}))
    _, forced_off = ImageEnhancer("off")(img, force=frozenset({"rain"}))
    assert "rain" not in auto["applied"]
    assert "rain" in forced["applied"] and forced_off["applied"] == ["rain"]


def test_always_mode_keeps_shape_and_dtype():
    img = _scene()
    out, _ = ImageEnhancer("always")(img, fast=True)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_dedupe_bikes_merges_overlapping_boxes():
    strong = Box(100, 100, 300, 400, 0.9, "motorcycle")
    dup = Box(110, 120, 290, 380, 0.4, "motorcycle")      # inside the strong one
    other = Box(500, 100, 700, 400, 0.5, "motorcycle")    # a different bike
    kept = dedupe_bikes([dup, strong, other])
    assert kept == [strong, other]


def test_head_only_fallback_judges_riders_from_heads():
    person = Box(100, 50, 200, 400, 0.8, "person")
    heads = [Box(130, 55, 170, 100, 0.9, "no_helmet"), Box(400, 60, 440, 100, 0.9, "helmet")]
    motos = head_only_riders([person], heads, 800, 600)
    assert [m.status for m in motos] == [HelmetStatus.VIOLATION, HelmetStatus.SAFE]
    assert not any(m.bike_detected for m in motos)
