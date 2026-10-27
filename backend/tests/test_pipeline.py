"""End-to-end tests with the real models (skipped if the Phase 1 weights are not present).

Put a few test images in tests/samples/ (any .jpg/.png with motorcycles) to run them.
"""
from pathlib import Path

import pytest

from app.config import get_settings

SAMPLES = sorted(p for p in (Path(__file__).parent / "samples").glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
settings = get_settings()
needs_models = pytest.mark.skipif(not (settings.helmet_model.exists() and settings.plate_model.exists()),
                                  reason="Phase 1 model weights not found in models/")


@pytest.fixture(scope="module")
def service():
    from app.services.detection_service import DetectionService
    return DetectionService(settings)


@needs_models
@pytest.mark.skipif(not SAMPLES, reason="no sample images in tests/samples")
def test_image_report_structure(service):
    report = service.process_image(SAMPLES[0])
    assert report["type"] == "image" and report["files"]["annotated"] == "annotated.jpg"
    assert (settings.output_dir / report["run_id"] / "annotated.jpg").exists()
    assert (settings.report_dir / f"{report['run_id']}.json").exists()
    s = report["summary"]
    assert s["motorcycles"] == s["violations"] + s["safe"] + s["unknown"]
    for v in report["violations"]:
        assert (settings.output_dir / report["run_id"] / v["evidence_image"]).exists()


@needs_models
def test_api_health_and_upload():
    from fastapi.testclient import TestClient
    from app.api.main import app
    with TestClient(app) as client:
        h = client.get("/api/v1/health").json()
        assert h["status"] == "ok"
        assert client.post("/api/v1/detect/image", files={"file": ("x.txt", b"hello")}).status_code == 415
        if SAMPLES:
            with SAMPLES[0].open("rb") as f:
                r = client.post("/api/v1/detect/image", files={"file": (SAMPLES[0].name, f, "image/jpeg")})
            assert r.status_code == 200
            run_id = r.json()["run_id"]
            assert client.get(f"/api/v1/results/{run_id}").status_code == 200
            assert client.get(f"/api/v1/files/{run_id}/annotated.jpg").status_code == 200
            assert client.get(f"/api/v1/files/{run_id}/..%2Fsecret").status_code in (400, 404)
