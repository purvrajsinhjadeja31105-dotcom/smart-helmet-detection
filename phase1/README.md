# Phase 1 — Custom YOLO training on Kaggle

Two detectors, trained with the same notebook:

| Model | Classes | Why separate |
|---|---|---|
| **A · helmet** | `helmet`, `no_helmet` (head boxes) | the core decision: SAFE vs violation |
| **B · plate** | `number_plate` | plates are tiny; a dedicated model at 960 px (later run on rider crops) is more accurate and gives tighter boxes for OCR |

Motorcycles and persons come from a **COCO-pretrained YOLO** in Phase 2 (no training). Our datasets
disagree on what a "rider" box is (per person vs per vehicle), and training on conflicting
definitions would lower accuracy.

```
phase1/
├── src/                     # single source of truth (percent-format .py, testable locally)
│   ├── data_prep.py         # Notebook 1: download → clean → dedupe → split → augment → EDA
│   └── train.py             # Notebook 2: train → evaluate → per-source test → visual test → export
├── notebooks/               # generated, upload these to Kaggle
│   ├── 01_data_prep.ipynb
│   └── 02_train.ipynb
└── tools/build_notebooks.py # regenerates notebooks/ from src/
```

---

## How to run on Kaggle

**Notebook 1: data prep (CPU, about 10–20 min)**
1. Kaggle → *Create → New Notebook → File → Import Notebook* → `notebooks/01_data_prep.ipynb`.
2. Settings: **Accelerator: None**, **Internet: On**.
3. *(Optional, recommended)* **Roboflow key:** sign up at roboflow.com → Settings → API Keys → copy the *Private API Key*.
   In the notebook: *Add-ons → Secrets → Add* `ROBOFLOW_API_KEY` and tick it for this notebook.
4. *(Optional, recommended)* **EdgeVision:** open https://data.mendeley.com/datasets/j82bnw7gsr/1 →
   *Download All* (zip) → Kaggle *Datasets → New Dataset*, upload the zip, name it **edgevision**
   (private) → in the notebook *Add Input* → your `edgevision` dataset.
5. *Save Version → Save & Run All (Commit)*. Check the log for `[skip]` lines and the cleaning report.

**Notebook 2: training (GPU, up to about 11 h per model)**
1. Import `notebooks/02_train.ipynb`. Settings: **Accelerator: GPU T4 ×2**, **Internet: On**.
2. *Add Input → Notebook Output →* your data-prep notebook.
3. Keep `TASK = "helmet"` → *Save Version → Save & Run All (Commit)*. You can close the browser.
4. When it finishes, open the version → *Output* → download `weights/` and `results/` as a backup.
5. For the plate model: set `TASK = "plate"` and commit again (a new version).

**Kaggle quota:** GPU is limited to about 30 h per week. One helmet run plus one plate run takes about 21 h. A commit
continues even when the browser is closed. Tell me when a session stops, and I'll give you a short re-setup.

---

## What the data audit found (verified on the downloaded files, 2026-10-12)

| Source | Images | Used for | Notes |
|---|---|---|---|
| aryanvaid13 Indian Helmet | 942 | helmet + plate | `names: ['0'..'4']`; checked visually: 0 plate, 1 no-helmet face, 2 good helmet, 3 bad helmet, 4 rider. 51 images have whole-body boxes labelled as faces and are dropped. |
| anushka Smart Helmet | 485 | helmet | Many images are bike catalogue photos with no rider, and it contains copies of another dataset (`BikesHelmets*`, `- Copy`). 67 duplicates removed. |
| aneesarom | 124 | helmet + plate | RGBA images, converted to RGB. One corrupt image. |
| pkdarabi HelmetViolations | 1,937 → **255 unique** | plate only | 3× Roboflow augmentation, two overlapping versions, grayscale. Its helmet classes box the **whole rider**, so they can't be used for heads. |
| deepakat002 plates | 160 frames / 3 videos | plate | Each video is kept in one split, so frames from one video never appear in both train and test. |
| EdgeVision (Mendeley) | 2,392 | helmet | Needs a manual download (Cloudflare blocks scripted downloads). |
| Roboflow bhumik / miniproject | 2,376 / 1,083 | helmet + plate | Need a `ROBOFLOW_API_KEY`. The notebook checks automatically whether the "helmet" boxes are heads or bodies. |
| HF "42k Indian helmet" | – | **not used** | Both Hugging Face repos contain only a README and data.yaml, with zero images. |

Local test result (without EdgeVision and Roboflow): **helmet 1,397 unique images**
(1,142 / 122 / 133) and **plate 1,375** (1,195 / 103 / 77). Adding EdgeVision and Roboflow should bring
the helmet set to about **6,000–7,000** unique images.

---

## Phase 1, Step 1: the theory in brief

**Object detection** = for every object in an image, predict *what* it is (class) and *where* it is
(bounding box) with a confidence score. Classification gives one label per image; detection gives many
boxes per image.

**YOLO ("You Only Look Once")** is a *one-stage* detector. It runs a single network pass over the image and
predicts all boxes at once:
- **Backbone** (CNN): extracts features (edges → parts → objects) at several scales.
- **Neck** (FPN/PAN): mixes fine, high-resolution features with coarse, semantic ones, so both small
  heads and large bikes are found.
- **Head**: at every location of 3 feature maps (stride 8/16/32), it predicts class scores and box
  distances. Modern YOLO is *anchor-free*, predicting box edges directly instead of adjusting preset anchor boxes.
- **Loss**: box loss (CIoU / DFL) plus classification loss (BCE). Older YOLOs then remove duplicate boxes with
  **NMS**. **YOLO26** is NMS-free (end-to-end), which is faster and simpler to deploy.

| Detector | Type | Speed | Accuracy on small objects | Why not / why |
|---|---|---|---|---|
| **YOLO (11/26)** | one-stage, anchor-free | real-time (about 5–15 ms on a GPU) | good, better at higher image size | **chosen**: best accuracy per millisecond, simple training/export, mature tooling |
| Faster R-CNN | two-stage (region proposals, then classifier) | slow (about 50–100 ms) | good | 5–10× slower, heavier deployment, no real accuracy gain on this task |
| SSD | one-stage, anchors | fast | **weak** on small objects | older; heads and plates are small |
| DETR / RT-DETR | transformer, set prediction, no NMS | DETR slow, RT-DETR fast | good on large, weaker on small | needs more data and longer training; with about 7k images YOLO converges better |

**Why YOLO for this project:** we need many small objects per image (heads, plates), video speed
(Phase 2 processes every frame), CPU fallback, and ONNX export. YOLO is strongest on exactly
this mix.
