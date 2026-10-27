"""Logging: human-readable console output plus a rotating log file (logs/app.log)."""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


def setup_logging(log_dir: Path, level: int = logging.INFO) -> None:
    """Configure the root logger once. Safe to call repeatedly (e.g. from tests and the API)."""
    global _configured
    if _configured:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_FORMAT, "%H:%M:%S"))
    file = logging.handlers.RotatingFileHandler(log_dir / "app.log", maxBytes=5_000_000, backupCount=3,
                                                encoding="utf-8")
    file.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(console)
    root.addHandler(file)
    # third-party libraries are very chatty at INFO
    for noisy in ("ultralytics", "ppocr", "paddle", "paddlex", "PIL", "urllib3", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True
