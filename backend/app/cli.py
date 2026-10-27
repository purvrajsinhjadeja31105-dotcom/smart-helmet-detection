"""Command-line entry point (no web server needed).

    cd backend
    python -m app.cli path/to/image.jpg
    python -m app.cli path/to/video.mp4
    python -m app.cli folder/ --enhance off        # every image/video in a folder
    python -m app.cli photo.jpg --open             # demo: opens the annotated result on screen
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Helmet violation detection")
    ap.add_argument("inputs", nargs="+", type=Path, help="image/video files or folders")
    ap.add_argument("--enhance", choices=["auto", "always", "off"], help="override enhancement mode")
    ap.add_argument("--device", help='"auto", "cpu" or a GPU index')
    ap.add_argument("--remove-rain", action="store_true", help="apply rain-streak removal (manual option)")
    ap.add_argument("--open", action="store_true", help="open the annotated result when done (demo mode)")
    args = ap.parse_args()
    if args.enhance:
        os.environ["HELMET_ENHANCE_MODE"] = args.enhance
    if args.device:
        os.environ["HELMET_DEVICE"] = args.device

    from app.config import get_settings
    from app.core.logging_config import setup_logging
    from app.services.detection_service import DetectionService
    from app.services.media import IMAGE_EXT, VIDEO_EXT

    settings = get_settings()
    setup_logging(settings.log_dir)
    files = []
    for p in args.inputs:
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXT | VIDEO_EXT)
        else:
            files.append(p)
    service = DetectionService(settings)
    for f in files:
        force = frozenset({"rain"}) if args.remove_rain else frozenset()
        report = service.process(f, progress=lambda x: print(f"\r  {f.name}: {x:.0%}", end="", file=sys.stderr),
                                 force=force)
        print(json.dumps({"file": f.name, "run_id": report["run_id"], "summary": report["summary"],
                          "violations": [{k: v[k] for k in ("plate_number", "plate_valid_format", "ocr_confidence")}
                                         for v in report["violations"]],
                          "outputs": str(settings.output_dir / report["run_id"])}, indent=2))
        if args.open:
            result_file = settings.output_dir / report["run_id"] / report["files"]["annotated"]
            if hasattr(os, "startfile"):          # Windows: open with the default image/video viewer
                os.startfile(result_file)
            else:
                print(f"result: {result_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
