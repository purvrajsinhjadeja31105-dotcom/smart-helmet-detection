"""Loading/validating uploads and writing videos."""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from app.core.exceptions import InvalidMediaError

log = logging.getLogger(__name__)

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def media_kind(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext in VIDEO_EXT:
        return "video"
    raise InvalidMediaError(f"unsupported file type {ext!r}; allowed: {sorted(IMAGE_EXT | VIDEO_EXT)}")


def load_image(path: Path) -> np.ndarray:
    """Read an image as BGR (handles non-ASCII Windows paths, which cv2.imread cannot)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise InvalidMediaError(f"cannot decode image {path.name}")
    if min(img.shape[:2]) < 64:
        raise InvalidMediaError("image is too small (minimum 64 px)")
    return img


def save_image(path: Path, img: np.ndarray, quality: int = 92) -> None:
    ok, buf = cv2.imencode(path.suffix or ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise IOError(f"cannot encode {path}")
    buf.tofile(str(path))


def open_video(path: Path, max_seconds: float) -> tuple[cv2.VideoCapture, float, int, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise InvalidMediaError(f"cannot open video {path.name}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or fps > 240:
        fps = 25.0
    if n and n / fps > max_seconds:
        cap.release()
        raise InvalidMediaError(f"video is {n / fps:.0f} s long; the limit is {max_seconds:.0f} s")
    return cap, fps, n, w, h


def make_writer(path: Path, fps: float, w: int, h: int) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise IOError(f"cannot create video writer for {path}")
    return writer


def to_browser_mp4(src: Path) -> Path:
    """Re-encode to H.264 so browsers can play it (OpenCV's 'mp4v' is not playable in Chrome/Firefox).

    Uses the ffmpeg binary bundled with ``imageio-ffmpeg`` (or ffmpeg on PATH). If neither is available
    the original file is kept - it still downloads and plays in VLC / Windows Media Player.
    """
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        log.warning("ffmpeg not found - video left as mp4v (install imageio-ffmpeg for browser playback)")
        return src
    dst = src.with_name(src.stem + "_h264.mp4")
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-preset", "veryfast", "-crf", "23", "-movflags", "+faststart", str(dst)]
    try:
        subprocess.run(cmd, check=True, timeout=1800)
    except Exception as e:
        log.warning("H.264 conversion failed (%s) - keeping mp4v file", e)
        return src
    src.unlink(missing_ok=True)
    final = src.with_name(src.stem + ".mp4")
    dst.replace(final)
    return final
