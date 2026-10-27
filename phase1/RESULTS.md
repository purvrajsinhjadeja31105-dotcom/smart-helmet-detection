# Phase 1 Results: Custom YOLO Models

**Project:** AI-Based Smart Helmet Detection System Using Deep Learning
**Phase 1 goal:** train accurate detectors for helmet use and number plates on Indian traffic images.
**Completed:** October 2026, trained on Kaggle (2× NVIDIA T4)

---

## 1. Final models

| Model | Classes | Architecture | Input | Test mAP50 | Test mAP50-95 | Precision | Recall | F1 | Deploy conf. |
|---|---|---|---|---|---|---|---|---|---|
| **A: Helmet** | `helmet`, `no_helmet` (head boxes) | YOLO26-m (21.8 M params) | 960 px, full image | **0.891** | 0.531 | 0.838 | 0.840 | 0.839 | 0.40 |
| **B: Plate** | `number_plate` | YOLO26-m | 640 px, motorcycle crop | **0.923** | 0.482 | 0.910 | 0.890 | 0.900 | 0.37 |

Motorcycles and persons are **not** trained. A COCO-pretrained YOLO26-m detects them (COCO class 3 = motorcycle, 0 = person).

Weights (PyTorch `.pt` and ONNX) are stored in `models/helmet/` and `models/plate/`.

**Test sets:** helmet uses 334 Indian images; plate uses 222 motorcycle crops from 204 Indian images. Both are held out completely: they were never used for training or for choosing the best epoch.

---

## 2. Why the system is built this way

```
image ──► COCO YOLO ──► motorcycle boxes ──► crop (+15% padding) ──► Model B ──► plate box ──► OCR (Phase 2)
   │
   └────► Model A ──► helmet / no_helmet head boxes ──► linked to a motorcycle by position
```

1. **Helmet labels are head boxes.** Every usable source labels the head (helmet or no helmet). Sources that boxed the whole rider were excluded from Model A.
2. **No trained "rider" class.** Sources disagree on what a rider box is: one box per person in some, one per vehicle in others. Training on conflicting definitions lowers accuracy, so motorcycles come from the COCO model instead.
3. **Plates are detected on motorcycle crops, not full images.** In a full traffic image a plate is often only 20–40 px wide. Cropping the motorcycle first makes the plate about 5× larger and removes car plates from the problem.

---

## 3. Data

### Sources (11 public datasets, all checked visually before use)

| Source | Raw images | Used for | Main issue found and fixed |
|---|---|---|---|
| aryanvaid13 Indian Helmet (Kaggle) | 942 | helmet + plate | Class names were `0..4`, so the meaning was checked visually. 51 images had whole-body boxes labelled as faces and were removed. |
| anushkagovindkadam Smart Helmet (Kaggle) | 485 | helmet | Contained copies of another dataset; 67 duplicates removed. |
| aneesarom (Kaggle) | 124 | helmet + plate | RGBA images were converted to RGB. |
| EdgeVision (Mendeley, 2026) | 2,392 | helmet | VOC XML labels were converted to YOLO format. |
| bhumik Helmet&NumberPlate (Roboflow) | 6,015 | helmet + plate | Each image appeared 3× as augmented copies; about 1,224 unique remained. |
| miniproject HelmetDetection (Roboflow) | 1,081 | helmet + plate | Duplicates removed. |
| pkdarabi HelmetViolations (Kaggle) | 1,937 | plate | Helmet boxes cover the whole rider, so only plates were used. 3× augmented, about 250 unique. |
| deepakat002 plates (Kaggle) | 160 | plate | Frames from 3 videos, so each video is kept within a single split. |
| devgurucodes Traffic Violations (Kaggle) | 11,772 | plate | 6,621 images had no plate labels at all and were dropped; 3,556 duplicates removed. |
| majorprojectssuu (Kaggle) | 1,596 | plate | Darknet labels with no names file, so class ids were checked visually. |
| Gujarat plates (Kaggle) | 355 | plate | – |

One more source, the Hugging Face "42k Indian helmet" set, was rejected: its repositories contain no images.

### Final datasets (unique images after cleaning)

| Dataset | Train | Val | Test | Total |
|---|---|---|---|---|
| Helmet | 4,747 (+908 weather copies) | 512 | 334 | 5,593 |
| Plate (full images) | 4,193 | 311 | 204 | 4,708 |

### Cleaning pipeline
- **Corrupt or tiny images:** removed.
- **Boxes:** clipped to the image; boxes under 4 px removed; same-class double labels merged (IoU > 0.9).
- **Label errors:** an image was removed if a head box covered more than 40% of it, or if a head was labelled both helmet and no-helmet (IoU > 0.6).
- **Duplicates:** an image was treated as a duplicate when its perceptual hashes matched (pHash ≤ 6, dHash ≤ 10, mirrored images included) **and** at least 60% of its label boxes matched, or when it shared a Roboflow base name. The box check matters because fixed-camera frames have identical hashes but show different riders, and those are kept.
- **Leakage control:** similar scenes (pHash ≤ 12) and frames from the same video are always placed in the same split.
- **Split:** each group's split is decided by a hash of the group, so it is stable even if data is added later. Test images come only from **Indian** sources (10% test / 10% val / 80% train).
- **Background images:** capped at 10%.
- **Class balance:** oversampling of the minority class (`no_helmet`) was built in, but wasn't needed after all sources were merged (7,456 helmet vs 5,170 no-helmet boxes).

### Augmentation
- **Offline, train only:** 20% of training images received one weather effect: fog, rain, low light, motion blur or sensor noise.
- **Online (Ultralytics):** mosaic, scale 0.5–1.5×, translation, ±5° rotation, shear, perspective, HSV colour and brightness, mixup (helmet only), and horizontal flip (helmet only; mirrored plate text is unrealistic).

---

## 4. Training setup

| Parameter | Value | Reason |
|---|---|---|
| Pretrained weights | YOLO26-m (COCO) | Transfer learning; the "m" size is the best accuracy that fits a T4 |
| Image size | 960 (helmet) / 640 (plate crops) | Heads are small, so 960 helps; crops are already enlarged |
| Batch | 16 (8 per GPU, DDP on 2 GPUs) | Fits in 15 GB of GPU memory with mixed precision (AMP) |
| Optimizer / LR | `auto` → MuSGD, lr 0.01, cosine schedule, 3 warm-up epochs | Ultralytics defaults tuned for YOLO26 |
| Epochs | up to 200, capped at 10.5 h, early stopping after 40 epochs without improvement | Fits within Kaggle's 12 h session limit |
| Loss | box (L1 / IoU) + classification (BCE) | YOLO26 end-to-end (no NMS) |
| Evaluation | conf 0.001 and IoU 0.7 for mAP; deployment threshold = maximum F1 | Standard COCO-style evaluation |
| Seed | 42 | Reproducible results |

---

## 5. Experiments (plate model)

| Run | Change | Test mAP50 | Test recall |
|---|---|---|---|
| Plate v1 | Full images at 960 px | 0.653 | 0.726 |
| Plate v2 | **Motorcycle crops** at 640 px, plus context crops for plates the motorcycle detector missed (train only) | 0.768 | 0.735 |
| **Plate v3** | v2 **+ 3 more plate datasets** (data nearly doubled) | **0.923** | **0.890** |

The v3 test set is larger (222 crops instead of 93). On the source present in both test sets with identical images (aryan, CCTV), v3 scores **0.838**, still clearly above v2's overall 0.768.

### Accuracy per test source

| Model | Source | Test images | mAP50 |
|---|---|---|---|
| Plate v3 | trafficviol | 121 | 0.959 |
| Plate v3 | aryan (CCTV, hardest) | 74 | 0.838 |
| Plate v3 | gujarat | 13 | 0.995 |
| Plate v3 | anees | 8 | 0.995 |
| Plate v3 | mpsuu | 6 | 0.942 |

---

## 6. Observations and limitations

- **The helmet model overfits after about epoch 70:** validation classification loss starts rising while training loss keeps falling. More (and harder) data is the next lever, rather than longer training.
- **The hardest domain is distant CCTV views** (the aryan source) for both models, because heads and plates are only a few pixels there.
- **The plate model depends on the motorcycle detector.** If the COCO model misses a motorcycle, its plate is not searched. Phase 2 should measure this end to end.
- **Some test subsets are small** (6–13 images), so their per-source scores are indicative only.
- **mAP50-95 is moderate** (0.48–0.53). Boxes are found reliably but aren't always tight. For OCR, Phase 2 will pad the plate crop slightly.

---

## 7. Reproducing

1. `phase1/notebooks/01_data_prep.ipynb` on Kaggle (CPU, internet on; Roboflow API key in Kaggle Secrets; EdgeVision attached as an input).
2. `phase1/notebooks/02_train.ipynb` on Kaggle (GPU T4 ×2), with the output of notebook 1 attached as input. Set `TASK` to `"helmet"` or `"plate_crop"`.
3. Each run writes `results/<task>/metrics.json`, the training curves, PR/F1 curves, confusion matrices (at conf 0.001 and at the deployment threshold), visual test images, and `weights/<task>_best.pt/.onnx`.

The notebooks are generated from `phase1/src/*.py` with `python phase1/tools/build_notebooks.py`.
