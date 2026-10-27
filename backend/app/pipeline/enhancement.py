"""Image quality analysis and enhancement (fog, rain, motion blur, noise, contrast, low light).

Why "analyse first, then enhance only what is needed":
the detectors were trained on raw images (plus synthetic weather), so enhancing a *good* image
changes its statistics and can lower detection accuracy. Each step therefore runs only when a
measured quality score crosses its threshold ("auto" mode). "always" applies every step,
"off" disables enhancement (useful for A/B accuracy checks).

All algorithms are classical OpenCV/NumPy methods: no extra models, they run on CPU in
milliseconds-to-~1 s, and they are explainable. Deep restoration networks (e.g. DeblurGAN, PReNet)
would restore better but need their own weights and GPU time (listed as a possible improvement).
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class QualityReport:
    brightness: float     # mean V channel 0-255
    contrast: float       # std-dev of grey levels
    haze: float           # mean dark-channel value 0-1 (fog/haze raises it)
    saturation: float     # mean S channel 0-255 (fog washes colours out)
    sharpness: float      # variance of the Laplacian (low = blurry)
    noise: float          # estimated Gaussian noise sigma (Immerkaer method)
    rain: float           # fraction of pixels on thin bright near-vertical streaks

    # Thresholds calibrated on 250 clean training images vs. 80 synthetic copies per weather type
    # (Phase 1 data, see backend/README.md): each flag fires on <=5-10% of clean images and on most
    # degraded ones.
    @property
    def is_dark(self) -> bool:           # clean: 5th pct 50, low-light copies: median 46
        return self.brightness < 60

    @property
    def is_low_contrast(self) -> bool:   # clean: 5th pct 38
        return self.contrast < 35

    @property
    def is_hazy(self) -> bool:           # fog copies: haze 0.47 / contrast 41 / saturation 19 (median)
        return self.haze > 0.40 and self.contrast < 50 and self.saturation < 40

    @property
    def is_blurry(self) -> bool:         # motion-blur copies: median 26, clean: median 582
        return self.sharpness < 40

    @property
    def is_noisy(self) -> bool:          # clean: 95th pct 4.1, noise copies: 5th pct 5.2
        return self.noise > 5.5

    @property
    def is_rainy(self) -> bool:
        # The streak measure does NOT separate rainy from clear images on our data (clean median 0.03,
        # rain copies 0.02), so automatic rain detection is disabled. Rain removal runs when the user
        # asks for it (API ``remove_rain=true`` / CLI ``--remove-rain``) or in "always" mode.
        # FUTURE: replace this single property with a trained rain/no-rain classifier - nothing else
        # in the pipeline has to change.
        return False

    def flags(self) -> dict:
        return {"fog": self.is_hazy, "rain": self.is_rainy, "motion_blur": self.is_blurry,
                "noise": self.is_noisy, "low_contrast": self.is_low_contrast, "low_light": self.is_dark}


def _small(img: np.ndarray, max_side: int = 640) -> np.ndarray:
    s = max_side / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def _dark_channel(img: np.ndarray, patch: int = 15) -> np.ndarray:
    return cv2.erode(img.min(axis=2), cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch)))


def _rain_mask(gray: np.ndarray) -> np.ndarray:
    """Thin bright streaks: white top-hat with a vertical line kernel, minus wide bright regions."""
    k = max(9, gray.shape[0] // 40) | 1
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
    wide = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    streak = cv2.subtract(tophat, cv2.dilate(wide, np.ones((1, 5), np.uint8)) // 2)
    return (streak > 25).astype(np.uint8)


def analyse(img: np.ndarray) -> QualityReport:
    """Measure image quality on a downscaled copy (fast, resolution-independent)."""
    small = _small(img)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    # Immerkaer (1996) fast noise variance estimation
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    h, w = gray.shape
    sigma = float(np.abs(cv2.filter2D(gray.astype(np.float32), -1, kernel)).sum()
                  * np.sqrt(0.5 * np.pi) / (6.0 * (w - 2) * (h - 2)))
    return QualityReport(
        brightness=float(hsv[..., 2].mean()), contrast=float(gray.std()),
        haze=float(_dark_channel(small).mean() / 255.0), saturation=float(hsv[..., 1].mean()),
        sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()), noise=sigma,
        rain=float(_rain_mask(gray).mean()))


# ----------------------------------------------------------------------------------------------
# individual enhancement operations
# ----------------------------------------------------------------------------------------------
def _guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """He et al. guided filter (edge-preserving smoothing), box-filter implementation."""
    box = lambda x: cv2.boxFilter(x, cv2.CV_32F, (r, r))
    mean_i, mean_p = box(guide), box(src)
    cov_ip = box(guide * src) - mean_i * mean_p
    var_i = box(guide * guide) - mean_i * mean_i
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return box(a) * guide + box(b)


def remove_fog(img: np.ndarray, omega: float = 0.9, t_min: float = 0.15) -> np.ndarray:
    """Dark Channel Prior dehazing (He, Sun, Tang - CVPR 2009).

    Haze-free outdoor images have, in most patches, at least one colour channel near zero (the
    "dark channel"). Fog adds airlight, lifting the dark channel. From it we estimate the
    transmission t(x), refine it with a guided filter, and invert the haze model
    I = J*t + A*(1-t)  ->  J = (I - A) / t + A.
    """
    f = img.astype(np.float32) / 255.0
    small = _small(f, 800)
    dark = _dark_channel(small)
    flat = dark.ravel()
    top = flat.argsort()[-max(1, flat.size // 1000):]                 # brightest 0.1% of dark channel
    airlight = small.reshape(-1, 3)[top].max(axis=0).clip(0.5, 1.0)  # atmospheric light A
    t = 1.0 - omega * _dark_channel(small / airlight)
    t = _guided_filter(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), t.astype(np.float32), 40, 1e-3)
    t = cv2.resize(t, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR).clip(t_min, 1.0)
    out = (f - airlight) / t[..., None] + airlight
    return (out.clip(0, 1) * 255).astype(np.uint8)


def remove_rain(img: np.ndarray) -> np.ndarray:
    """Rain-streak removal: detect thin bright vertical streaks and inpaint them (Telea method)."""
    small_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = _rain_mask(small_gray)
    mask = cv2.dilate(mask, np.ones((3, 1), np.uint8)) * 255
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)


def reduce_motion_blur(img: np.ndarray, amount: float = 0.8) -> np.ndarray:
    """Unsharp masking: restores edge contrast lost to mild blur.

    True deconvolution needs the (unknown) blur kernel; for detection, sharper edges are what
    matters, and unsharp masking cannot create ringing artefacts strong enough to fool the model.
    """
    blurred = cv2.GaussianBlur(img, (0, 0), 2.0)
    return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)


def remove_noise(img: np.ndarray, fast: bool = False) -> np.ndarray:
    """Non-local means (high quality, ~0.5-2 s on large images) or bilateral filter (fast, video)."""
    if fast:
        return cv2.bilateralFilter(img, 5, 40, 5)
    return cv2.fastNlMeansDenoisingColored(img, None, 6, 6, 7, 21)


def enhance_contrast(img: np.ndarray, clip: float = 2.0) -> np.ndarray:
    """CLAHE on the L channel of LAB: local contrast without shifting colours or blowing highlights."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def enhance_low_light(img: np.ndarray) -> np.ndarray:
    """Adaptive gamma: chooses gamma so that the mean brightness moves towards mid-grey."""
    v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2].mean() / 255.0
    gamma = float(np.clip(np.log(0.45) / np.log(max(v, 1e-3)), 0.4, 1.0))   # <1 brightens
    lut = ((np.arange(256) / 255.0) ** gamma * 255).astype(np.uint8)
    return cv2.LUT(img, lut)


class ImageEnhancer:
    """Runs the enhancement chain in the order required by the project specification:
    fog -> rain -> motion blur -> noise -> contrast -> low light."""

    def __init__(self, mode: str = "auto"):
        if mode not in {"auto", "always", "off"}:
            raise ValueError(f"enhance mode must be auto/always/off, got {mode!r}")
        self.mode = mode

    def __call__(self, img: np.ndarray, fast: bool = False,
                 force: frozenset[str] = frozenset()) -> tuple[np.ndarray, dict]:
        """Return (enhanced image, report).

        fast:  cheaper filters (used for video frames).
        force: steps the user explicitly requested (e.g. {"rain"}); they run even in "off" mode and
               even if the quality check did not flag them.
        """
        if self.mode == "off" and not force:
            return img, {"mode": "off", "applied": []}
        q = analyse(img)
        if self.mode == "auto":
            need = q.flags()
        else:
            need = {k: self.mode == "always" for k in q.flags()}
        for step in force:
            need[step] = True
        steps = [("fog", remove_fog), ("rain", remove_rain), ("motion_blur", reduce_motion_blur),
                 ("noise", lambda x: remove_noise(x, fast)), ("low_contrast", enhance_contrast),
                 ("low_light", enhance_low_light)]
        out, applied = img, []
        for name, fn in steps:
            if need[name]:
                try:
                    out = fn(out)
                    applied.append(name)
                except cv2.error as e:     # never fail a detection because an enhancement failed
                    log.warning("enhancement step %s failed: %s", name, e)
        return out, {"mode": self.mode, "applied": applied, "requested": sorted(force),
                     "quality": {k: round(v, 3) for k, v in asdict(q).items()}}


ENHANCEMENT_STEPS = ("fog", "rain", "motion_blur", "noise", "low_contrast", "low_light")
