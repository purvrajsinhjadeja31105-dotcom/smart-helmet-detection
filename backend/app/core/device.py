"""GPU / CPU selection with automatic CPU fallback."""
from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    device: str      # value passed to Ultralytics: "0" (first GPU) or "cpu"
    half: bool       # FP16 inference - only on CUDA GPUs (about 2x faster, same accuracy)
    name: str


def select_device(preference: str = "auto") -> DeviceInfo:
    """Pick the inference device.

    preference: "auto" -> first CUDA GPU if available, otherwise CPU; "cpu" forces CPU;
    "0", "1", ... selects a specific GPU (falls back to CPU, with a warning, if it is unavailable).
    """
    try:
        import torch
        cuda = torch.cuda.is_available()
    except Exception:          # torch missing or broken CUDA install -> CPU still works
        cuda = False
    if preference == "cpu" or not cuda:
        if preference not in ("auto", "cpu"):
            log.warning("GPU %s requested but CUDA is not available - falling back to CPU", preference)
        return DeviceInfo("cpu", False, "CPU")
    index = "0" if preference == "auto" else preference
    import torch
    name = torch.cuda.get_device_name(int(index))
    return DeviceInfo(index, True, name)
