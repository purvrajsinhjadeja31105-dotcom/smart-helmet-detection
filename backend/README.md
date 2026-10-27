# Phase 2: AI Pipeline (backend)

The pipeline takes an image or video and returns, for every motorcycle, **SAFE / VIOLATION / UNKNOWN**. For every violation it also returns the **number plate** read by OCR, plus evidence images and a JSON violation report.

```
upload ──► (video: sample 5 fps + track motorcycles with ByteTrack)
       ──► quality check ──► enhancement only where needed
              fog (Dark Channel Prior) → motion blur (unsharp mask) → noise (NL-means / bilateral)
              → contrast (CLAHE) → low light (adaptive gamma)
       ──► COCO YOLO26-m: motorcycles + persons
       ──► helmet model (Phase 1): helmet / no_helmet heads
       ──► association: head → rider → motorcycle (geometry)
       ──► status per motorcycle: any no_helmet → VIOLATION · only helmets → SAFE · no head → UNKNOWN
       ──► VIOLATION: crop motorcycle +15% → plate model (Phase 1) → plate crop → PaddleOCR
              → Indian plate format repair/validation (MH12AB1234, DL3CAB1234, 22BH1234AA)
       ──► annotated image / H.264 video + evidence crops + reports/<run_id>.json
```

## Folder structure

```
backend/
├── app/
│   ├── config.py               all settings (paths, thresholds, sizes); override with HELMET_* env vars
│   ├── cli.py                  python -m app.cli <files/folders>
│   ├── core/
│   │   ├── device.py           GPU if available, else CPU (automatic fallback)
│   │   ├── logging_config.py   console + rotating logs/app.log
│   │   └── exceptions.py       ModelLoadError, InvalidMediaError, ProcessingError
│   ├── pipeline/
│   │   ├── types.py            Box, Motorcycle, PlateRead, FrameAnalysis (JSON-serialisable)
│   │   ├── enhancement.py      quality analysis + 6 enhancement operations
│   │   ├── detectors.py        YOLO wrapper (predict / track)
│   │   ├── association.py      heads/riders → motorcycles, status decision
│   │   ├── ocr.py              PaddleOCR plate reader
│   │   ├── plate_text.py       Indian plate cleaning, character repair, validation
│   │   ├── annotate.py         drawing
│   │   ├── image_pipeline.py   HelmetViolationPipeline (one image / frame)
│   │   └── video_pipeline.py   VideoProcessor (sampling, tracking, voting, timeline)
│   ├── services/
│   │   ├── media.py            safe loading, size/length limits, video writer, H.264 conversion
│   │   └── detection_service.py  runs a job, writes outputs/ + reports/
│   └── api/
│       ├── main.py             FastAPI app and endpoints
│       └── jobs.py             background video jobs
├── tests/                      pytest: unit tests + end-to-end tests (tests/samples)
└── requirements.txt
```

Runtime folders are created at the project root: `uploads/`, `outputs/<run_id>/`, `reports/<run_id>.json` and `logs/`.

## Setup (Windows, Python 3.10–3.12)

```bash
cd "D:\Helmet Detection"
python -m venv .venv
.venv\Scripts\activate
pip install -r backend\requirements.txt
```

The models must be in `models/helmet/helmet_best.pt` and `models/plate/plate_crop_best.pt`. The COCO model downloads automatically on the first run, and so do the PaddleOCR models (about 20 MB).

## Run

```bash
cd backend
python -m app.cli ..\path\to\image.jpg            # or a video, or a folder
uvicorn app.api.main:app --port 8000              # API, then open http://localhost:8000/docs
pytest -q                                          # tests
```

## API

| Method | Path | Result |
|---|---|---|
| GET | `/api/v1/health` | status, device, model loading errors |
| POST | `/api/v1/detect/image` | the report (synchronous) |
| POST | `/api/v1/detect/video` | `{job_id}` (202); processed in the background |
| GET | `/api/v1/jobs/{job_id}` | `queued / running / done / failed`, progress 0–1, `run_id` |
| GET | `/api/v1/results/{run_id}` | the stored report |
| GET | `/api/v1/files/{run_id}/{file}` | annotated image or video, `violation_*.jpg`, `plate_*.jpg` |

## Test results (October 2026, CPU: AMD Ryzen 5 5500U, no GPU)

| Test | Result |
|---|---|
| Unit + end-to-end + API tests (`pytest`) | **22 passed** |
| Sample video (4.4 s, 2 riders without helmets, plate KL 09 AQ 3439) | 1 motorcycle, **VIOLATION** in 22 of 22 analysed frames, 2 riders, plate read **KL09AO3439** (9 of 10 characters; Q read as O). H.264 output video. |
| Plate stage on 40 unseen test images (88 motorcycles) | 63 plates found, 28 read, 13 valid Indian numbers. Clearly visible plates read correctly (e.g. TN68Y0303, AP25BH3661); most failures are plates too blurred or small to read even for a human. |
| Speed on CPU | about 2–6 s per image; video about 1.3 frames/s overall (5 analysed frames per video-second). A CUDA GPU is used automatically and is about 10× faster. |

### Does enhancement improve detection? (A/B test)
Setup: 107 unseen test/val images with synthetic degradations. Metric: helmet-model mAP50 without vs with automatic enhancement.

| Condition | No enhancement | Auto enhancement |
|---|---|---|
| clean (reference) | 0.845 | – |
| fog | 0.852 | 0.849 |
| low light | **0.830** | 0.805 |
| motion blur | 0.725 | 0.723 |
| noise | 0.832 | 0.834 |

**Conclusion:** enhancement does not improve detection, because the models were trained with weather augmentation in Phase 1, and it lowers accuracy on low-light images. So **detection runs on the original image** (`enhance_for_detection = False`). The enhanced image is shown to the user and used as a **second OCR attempt** for plates that could not be read.

**Exception: heavy fog.** With 70–85% synthetic fog, 1 of 5 motorcycles was lost, and fog removal (Dark Channel Prior) recovered all 5; see `docs/figures/fog_removal_demo.jpg`. So when the quality check detects fog, detection runs on the de-fogged image (`detect_on_enhanced_if = ("fog",)`). Fog removal cannot recover vehicles that are invisible in the original image (very dense fog). That is a physical limit of any camera-based system.

## Key design decisions

| Decision | Why |
|---|---|
| Enhance **only when the quality check flags a problem** | The detectors were trained on raw images plus synthetic weather. Enhancing a good image changes it and can lower accuracy. Thresholds were calibrated on 250 clean vs 80 degraded images per weather type. |
| **Rain removal is a manual option** | The rain-streak measure could not separate rainy from clear images on our data (median 0.02 vs 0.03), and applying it to clear images smeared details. So the user switches it on: API form field `remove_rain=true`, or CLI `--remove-rain`. **Future upgrade:** replace the single `QualityReport.is_rainy` property with a trained rain/no-rain classifier, and rain removal becomes automatic with no other code changes. |
| Heads must belong to a **motorcycle** | A pedestrian without a helmet is not a violation. Unassigned heads are listed in the report, but never counted as violations. |
| **UNKNOWN** instead of guessing | When no head is visible, a wrong VIOLATION is worse than no decision. |
| Plate detection on the **motorcycle crop** | Matches how the plate model was trained (Phase 1: 65% → 92% mAP50). |
| Plate **format repair** | OCR confuses O/0, I/1, S/5 and similar. Each plate position must be a letter or a digit, so wrong-class characters are repaired. A real state code is required. |
| Video: **tracking + voting** | Each rider is reported once. The majority over at least 2 analysed frames removes single-frame errors. |
| torch is imported **before** paddle; Paddle's oneDNN is off | Avoids two known Windows crashes (DLL conflict, and `ConvertPirAttribute2RuntimeAttribute`). |
