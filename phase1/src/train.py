# %% [markdown]
# # Phase 1 · Notebook 2 — Train, Evaluate, Test and Export YOLO
#
# One notebook trains **either** model — set `TASK` in the config cell:
#
# | `TASK` | Detects | Data (from Notebook 1 output) |
# |---|---|---|
# | `"helmet"` | `helmet`, `no_helmet` | `datasets/helmet/data.yaml` |
# | `"plate"`  | `number_plate` | `datasets/plate/data.yaml` |
#
# **Kaggle settings:** Accelerator = **GPU T4 ×2**, Internet = **ON**,
# Input = *Add Input → Notebook Output →* your data-prep notebook.
# Run with **Save Version → Save & Run All (Commit)** so training continues when you close the
# browser; everything in `/kaggle/working` (weights, plots, metrics) becomes the version's output.
#
# ```
# config → locate data → pick model → (optional) tune → train (time-boxed) → evaluate on TEST
#   → per-source accuracy → best confidence threshold → visual check on unseen images → export .pt/.onnx
# ```

# %% [markdown]
# ## 1. Configuration — every training parameter explained
#
# | Parameter | Value | What it does / why this value |
# |---|---|---|
# | `MODEL_CANDIDATES` | YOLO26-m → YOLO11-m | COCO-pretrained weights (transfer learning). "m" (≈20M params) is the best accuracy/speed trade-off that fits a T4 at 960 px; the first one that loads is used. |
# | `IMGSZ` | 960 | Heads and plates are small (median plate ≈ 5% of image width). At 640 px a plate is ~30 px; at 960 px ~48 px — small-object recall rises clearly. Cost: ~2.2× compute. |
# | `EPOCHS` | 200 (upper bound) | One epoch = one pass over the training set. `TIME_HOURS` stops earlier. |
# | `TIME_HOURS` | 10.5 | Kaggle sessions end at 12 h. Ultralytics scales the LR schedule to this time budget so training always finishes properly (final LR decay, best.pt saved). |
# | `PATIENCE` | 40 | Early stopping: stop if validation fitness has not improved for 40 epochs (prevents over-fitting and wasted GPU hours). |
# | `BATCH` | 16 (8 per GPU) | Images per optimisation step. Larger = smoother gradients but more GPU memory; 8 per T4 at 960 px with the m model fits in 15 GB with AMP. |
# | `optimizer` | `auto` | Ultralytics picks the optimizer and initial LR for the model family (AdamW for short runs, SGD/MuSGD for long ones) — these defaults were tuned on COCO. |
# | `lr0` / `lrf` | auto / 0.01 | Initial learning rate; final LR = `lr0 × lrf`. |
# | `cos_lr` | True | Cosine scheduler: LR falls smoothly to the final value — usually +0.5–1 mAP over linear decay for fine-tuning. |
# | `warmup_epochs` | 3 | LR ramps up from ~0 for the first 3 epochs so the pretrained weights are not destroyed by large early updates. |
# | `close_mosaic` | 10 | Disable mosaic for the last 10 epochs so the model finishes on realistic, un-stitched images. |
# | Loss | built-in | Box regression: CIoU + Distribution Focal Loss (YOLO11) / ProgLoss (YOLO26); classification: BCE. Weights `box=7.5`, `cls=0.5`, `dfl=1.5` (defaults). |
# | `CONF` / `IOU` (eval) | 0.001 / 0.7 | For mAP, predictions are kept down to 0.001 confidence (the full PR curve is needed). The **deployment** confidence threshold is chosen afterwards from the F1 curve. IoU 0.7 is the NMS overlap threshold. |
#
# **Online augmentation** (applied randomly each time an image is loaded, train only):
# `mosaic` (4 images stitched → many scales/contexts), `scale=0.5` (zoom 0.5–1.5×), `translate=0.1`,
# `degrees=5` (small rotation; riders lean), `shear=2`, `perspective=0.0005` (camera angle),
# `fliplr=0.5` (mirror — **off for plates**, mirrored text is unrealistic), `hsv_h/s/v` (colour,
# saturation, **brightness**), `mixup=0.1` (blend two images, regularises). Ultralytics additionally
# applies light blur / CLAHE **contrast** / grayscale via Albumentations. Fog, rain, low-light, motion
# blur and noise were added offline in Notebook 1.

# %%
import os, sys, json, glob, shutil, random, subprocess
from pathlib import Path

IS_KAGGLE = Path("/kaggle/working").exists()
if IS_KAGGLE:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "ultralytics"], check=False)

import numpy as np
import pandas as pd
import cv2
import yaml
import torch
import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ultralytics
from ultralytics import YOLO

TASK = os.environ.get("HELMET_TASK", "helmet")           # "helmet", "plate" or "plate_crop"
SMOKE = os.environ.get("HELMET_SMOKE") == "1"            # tiny CPU run to test the code path

MODEL_CANDIDATES = ["yolo26m.pt", "yolo11m.pt"]
FINETUNE_FROM = None       # e.g. "/kaggle/input/<previous-train-notebook>/weights/helmet_best.pt"
IMGSZ = 640 if TASK == "plate_crop" else 960   # crops are small: 640 already enlarges the plate
EPOCHS = 200
TIME_HOURS = 10.5
PATIENCE = 40
BATCH = 16
SEED = 42
RUN_TUNE = False           # hyper-parameter search (section 3) - expensive, off by default
TUNED_HYP = None           # path to best_hyperparameters.yaml from a previous tuning run
EVAL_CONF, EVAL_IOU = 0.001, 0.7

AUG = dict(hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=5.0, translate=0.1, scale=0.5, shear=2.0,
           perspective=0.0005, flipud=0.0, fliplr=0.5 if TASK == "helmet" else 0.0,
           mosaic=1.0, mixup=0.1 if TASK == "helmet" else 0.0, close_mosaic=10)

WORK = Path("/kaggle/working") if IS_KAGGLE else Path(os.environ.get("HELMET_WORK", "_work")).resolve()
RUNS, WEIGHTS, RESULTS = WORK / "runs", WORK / "weights", WORK / "results" / TASK
for d in (RUNS, WEIGHTS, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

N_GPU = torch.cuda.device_count()
DEVICE = list(range(N_GPU)) if N_GPU > 1 else (0 if N_GPU == 1 else "cpu")
WORKERS = 4 if IS_KAGGLE else 0
if SMOKE:
    MODEL_CANDIDATES, IMGSZ, EPOCHS, TIME_HOURS, BATCH = ["yolo11n.pt"], 320, 1, None, 4
if N_GPU == 0 and not SMOKE:
    print("WARNING: no GPU found - enable 'GPU T4 x2' in the notebook settings.")
random.seed(SEED); np.random.seed(SEED)
print(f"Ultralytics {ultralytics.__version__} | torch {torch.__version__} | GPUs: {N_GPU} -> device={DEVICE} | TASK={TASK}")


def show_image(path, title=None, size=(12, 8)):
    if not Path(path).exists():
        print("missing plot:", path); return
    fig = plt.figure(figsize=size); plt.imshow(cv2.imread(str(path))[..., ::-1]); plt.axis("off")
    if title: plt.title(title)
    plt.show() if "ipykernel" in sys.modules else plt.close(fig)

# %% [markdown]
# ## 2. Locate the prepared dataset and copy it to local disk
#
# `/kaggle/input` is read-only and network-backed. Copying to `/tmp` (local SSD, ~1–2 GB, a few
# seconds) makes data loading faster and lets Ultralytics write its label cache. `/tmp` is not saved
# to the output, so the version output stays small.

# %%
DATA_TASK = "plate" if TASK == "plate_crop" else TASK      # plate_crop is built from the plate dataset


def find_data_yaml():
    if os.environ.get("HELMET_DATA"):
        return Path(os.environ["HELMET_DATA"]) / DATA_TASK / "data.yaml"
    hits = glob.glob(f"/kaggle/input/**/datasets/{DATA_TASK}/data.yaml", recursive=True) or \
           glob.glob(str(WORK / f"datasets/{DATA_TASK}/data.yaml"))
    if not hits:
        raise FileNotFoundError(f"datasets/{DATA_TASK}/data.yaml not found - attach the data-prep notebook output")
    return Path(hits[0])


SRC_YAML = find_data_yaml()
if IS_KAGGLE:
    local = Path("/tmp/data") / DATA_TASK
    if not local.exists():
        shutil.copytree(SRC_YAML.parent, local)
    DATA_YAML = local / "data.yaml"
else:
    DATA_YAML = SRC_YAML
DATA_DIR = DATA_YAML.parent
CLASS_NAMES = yaml.safe_load(DATA_YAML.read_text())["names"]
MANIFEST = pd.read_csv(DATA_DIR / "manifest.csv")
print("data:", DATA_YAML, "| classes:", CLASS_NAMES)
print(MANIFEST.groupby(["split", "kind"]).size().unstack(fill_value=0).to_string())

# %% [markdown]
# ## 2b. Motorcycle crops for the plate model (`TASK = "plate_crop"` only)
#
# In a full traffic image a plate is often only 20–40 px wide, and some sources label car plates
# while others do not. The deployed pipeline will only ever look for a plate **on a motorcycle**, so
# the plate model is trained exactly that way:
#
# 1. a COCO-pretrained YOLO finds every `motorcycle` (COCO class 3) in each plate-dataset image;
# 2. the box is enlarged by 15% on each side (plates hang below/behind the bike) and cropped;
# 3. plate labels are moved into crop coordinates. A plate counts only if ≥ 70% of it lies inside the
#    crop; crops that cut a plate in half are skipped (that plate would be an unlabelled half-object);
# 4. crops without a plate are kept as negatives, capped at 10% of positives;
# 5. **train only:** plates not on a detected motorcycle (bike missed by the detector, close-up shots,
#    car plates) get a "context crop" 3–6× the plate size, so their examples are not wasted.
#    Val/test contain motorcycle crops only → their score is the real pipeline accuracy.
#
# The original train/val/test split is preserved, so no test image leaks into training. Phase 2 must
# crop the same way (COCO motorcycle box + 15% padding) before calling this model.

# %%
MOTO_CLASS, CROP_PAD, PLATE_IN_FRAC, MIN_CROP_PX, CROP_NEG_FRAC = 3, 0.15, 0.70, 48, 0.10


def build_plate_crops(src_dir, dst_dir):
    det = YOLO("yolo11n.pt" if SMOKE else MODEL_CANDIDATES[0])        # COCO-pretrained detector
    src_manifest = pd.read_csv(src_dir / "manifest.csv")
    rows, stats, rng = [], Counter(), random.Random(SEED)
    for sp in ("train", "val", "test"):
        (dst_dir / "images" / sp).mkdir(parents=True, exist_ok=True)
        (dst_dir / "labels" / sp).mkdir(parents=True, exist_ok=True)
    files = list(src_manifest.itertuples(index=False))
    for k in range(0, len(files), 32):
        chunk = files[k:k + 32]
        results = det.predict([str(src_dir / r.file) for r in chunk], classes=[MOTO_CLASS], conf=0.25,
                              imgsz=960, batch=8, device=0 if N_GPU else "cpu", verbose=False)
        for r, res in zip(chunk, results):
            im = res.orig_img; h, w = im.shape[:2]
            plates = []
            for line in (src_dir / r.file.replace("images/", "labels/", 1)).with_suffix(".txt").read_text().splitlines():
                _, cx, cy, bw, bh = map(float, line.split())
                plates.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
            def labels_in(X1, Y1, X2, Y2):
                """Plates inside a crop (crop coordinates), whether a plate is cut, indices inside."""
                inside, cut, idx = [], False, []
                for i, (a1, b1, a2, b2) in enumerate(plates):
                    ix = max(0, min(a2, X2) - max(a1, X1)); iy = max(0, min(b2, Y2) - max(b1, Y1))
                    frac = ix * iy / max((a2 - a1) * (b2 - b1), 1e-6)
                    if frac >= PLATE_IN_FRAC:
                        inside.append((max(a1, X1) - X1, max(b1, Y1) - Y1, min(a2, X2) - X1, min(b2, Y2) - Y1))
                        idx.append(i)
                    elif frac > 0.1:
                        cut = True
                return inside, cut, idx

            covered = set()
            for j, (x1, y1, x2, y2) in enumerate(res.boxes.xyxy.tolist()):
                pw, ph = (x2 - x1) * CROP_PAD, (y2 - y1) * CROP_PAD
                X1, Y1 = max(0, int(x1 - pw)), max(0, int(y1 - ph))
                X2, Y2 = min(w, int(x2 + pw)), min(h, int(y2 + ph))
                if min(X2 - X1, Y2 - Y1) < MIN_CROP_PX:
                    stats["crop_too_small"] += 1; continue
                inside, cut, idx = labels_in(X1, Y1, X2, Y2)
                if cut:
                    stats["crop_cuts_plate"] += 1; continue
                covered.update(idx)
                rows.append(dict(r=r, tag=f"m{j}", kind=r.kind, box=(X1, Y1, X2, Y2), plates=inside))
            stats["plates_on_motorcycles"] += len(covered)
            stats[f"plates_on_motorcycles_{r.split}"] += len(covered)
            # Train only: plates on bikes the detector missed (or car plates) still teach "what a plate
            # looks like". A context window 3-6x the plate size mimics the scale of a motorcycle crop.
            # Val/test keep motorcycle crops only, so their score measures the real pipeline.
            if r.split == "train":
                for i, (a1, b1, a2, b2) in enumerate(plates):
                    if i in covered:
                        continue
                    side = max(a2 - a1, b2 - b1) * rng.uniform(3, 6)
                    cx = (a1 + a2) / 2 + rng.uniform(-0.25, 0.25) * side
                    cy = (b1 + b2) / 2 + rng.uniform(-0.25, 0.25) * side
                    X1, Y1 = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
                    X2, Y2 = min(w, int(cx + side / 2)), min(h, int(cy + side / 2))
                    if min(X2 - X1, Y2 - Y1) < MIN_CROP_PX:
                        stats["context_too_small"] += 1; continue
                    inside, cut, idx = labels_in(X1, Y1, X2, Y2)
                    if cut or i not in idx:
                        stats["context_cuts_plate"] += 1; continue
                    covered.update(idx)
                    rows.append(dict(r=r, tag=f"c{i}", kind="plate_context", box=(X1, Y1, X2, Y2), plates=inside))
                    stats["plates_in_context_crops"] += len(idx)
            stats["plates_total"] += len(plates)
            stats[f"plates_total_{r.split}"] += len(plates)
        print(f"  crops: {k + len(chunk)}/{len(files)} images scanned", end="\r")
    pos = [x for x in rows if x["plates"]]; neg = [x for x in rows if not x["plates"]]
    rng.shuffle(neg); neg = neg[:int(CROP_NEG_FRAC * len(pos))]
    manifest = []
    for x in pos + neg:
        r = x["r"]; X1, Y1, X2, Y2 = x["box"]
        crop = cv2.imread(str(src_dir / r.file))[Y1:Y2, X1:X2]
        cw, ch = X2 - X1, Y2 - Y1
        stem = f"{Path(r.file).stem}_{x['tag']}"
        cv2.imwrite(str(dst_dir / "images" / r.split / f"{stem}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (dst_dir / "labels" / r.split / f"{stem}.txt").write_text("".join(
            f"0 {(a + c) / 2 / cw:.6f} {(b + d) / 2 / ch:.6f} {(c - a) / cw:.6f} {(d - b) / ch:.6f}\n" for a, b, c, d in x["plates"]))
        manifest.append(dict(file=f"images/{r.split}/{stem}.jpg", split=r.split, source=r.source, kind=x["kind"],
                             group=r.group, orig=r.file, n_boxes=len(x["plates"]), n_number_plate=len(x["plates"])))
    (dst_dir / "data.yaml").write_text(yaml.safe_dump({"train": "images/train", "val": "images/val",
                                                       "test": "images/test", "nc": 1, "names": ["number_plate"]}, sort_keys=False))
    pd.DataFrame(manifest).to_csv(dst_dir / "manifest.csv", index=False)
    print(f"\nplate crops: {len(pos)} with plates, {len(neg)} negatives | {dict(stats)}")
    for sp in ("train", "val", "test"):
        on, tot = stats[f"plates_on_motorcycles_{sp}"], stats[f"plates_total_{sp}"]
        print(f"{sp:<5}: plates on detected motorcycles {on}/{tot} ({100 * on / max(tot, 1):.0f}%)"
              + (f", +{stats['plates_in_context_crops']} in context crops" if sp == "train" else
                 " - the rest are car plates or bikes the detector missed"))


if TASK == "plate_crop":
    from collections import Counter
    CROP_DIR = (Path("/tmp/data") if IS_KAGGLE else WORK / "data_crops") / "plate_crop"
    if not (CROP_DIR / "data.yaml").exists():
        build_plate_crops(DATA_DIR, CROP_DIR)
    # The detector ran inside this notebook process on GPU 0. Training (DDP) starts new processes on
    # both GPUs, so the memory PyTorch cached here must be released first, or GPU 0 runs out of memory.
    import gc
    gc.collect()
    if N_GPU:
        torch.cuda.empty_cache()
        print(f"GPU 0 memory still held by this process: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    DATA_YAML, DATA_DIR = CROP_DIR / "data.yaml", CROP_DIR
    CLASS_NAMES = yaml.safe_load(DATA_YAML.read_text())["names"]
    MANIFEST = pd.read_csv(DATA_DIR / "manifest.csv")
    print(MANIFEST.groupby(["split", "kind"]).size().unstack(fill_value=0).to_string())

# %% [markdown]
# ## 3. (Optional) Hyper-parameter tuning
#
# `model.tune()` runs a **genetic search**: each iteration mutates the best hyper-parameters found so
# far (LR, momentum, augmentation strengths, loss weights…), trains a short run and keeps the
# mutation if validation fitness (0.1·mAP50 + 0.9·mAP50-95) improves.
# It is very expensive (iterations × epochs), so it runs on 640 px, 30% of the data and 10 epochs;
# results only *rank* settings, they do not replace the full training. Typical gain: +1–3 mAP.
# Output: `runs/<task>_tune/best_hyperparameters.yaml` → put its path in `TUNED_HYP` and re-run.

# %%
if RUN_TUNE:
    tuner = YOLO(MODEL_CANDIDATES[-1])
    tuner.tune(data=str(DATA_YAML), imgsz=640, epochs=10, iterations=25, fraction=0.3, batch=16,
               device=0 if N_GPU else "cpu", optimizer="AdamW", plots=False, save=False, val=True,
               project=str(RUNS), name=f"{TASK}_tune", workers=WORKERS, seed=SEED)
HYP = yaml.safe_load(Path(TUNED_HYP).read_text()) if TUNED_HYP else {}
print("tuned hyper-parameters:", HYP or "none (using defaults)")

# %% [markdown]
# ## 4. Train
#
# Transfer learning: start from COCO-pretrained weights (the network already knows edges, faces,
# vehicles), then fine-tune on our data. `best.pt` = epoch with the highest validation fitness;
# `last.pt` = final epoch (used to resume).

# %%
def load_model():
    for cand in ([FINETUNE_FROM] if FINETUNE_FROM else []) + MODEL_CANDIDATES:
        try:
            m = YOLO(cand)
            print("using model:", cand)
            return m, Path(cand).stem
        except Exception as e:
            print(f"could not load {cand}: {e}")
    raise RuntimeError("no model could be loaded")


model, MODEL_NAME = load_model()
RUN_NAME = f"{TASK}_{MODEL_NAME}_{IMGSZ}"
train_args = dict(data=str(DATA_YAML), imgsz=IMGSZ, epochs=EPOCHS, batch=BATCH, device=DEVICE,
                  workers=WORKERS, seed=SEED, deterministic=False, patience=PATIENCE, optimizer="auto",
                  cos_lr=True, lrf=0.01, warmup_epochs=3, amp=True, cache=False, plots=True,
                  project=str(RUNS), name=RUN_NAME, exist_ok=True, **AUG)
if TIME_HOURS:
    train_args["time"] = TIME_HOURS
if SMOKE:
    train_args.update(fraction=0.1, plots=False)
train_args.update(HYP)
model.train(**train_args)

SAVE_DIR = Path(model.trainer.save_dir)
BEST, LAST = SAVE_DIR / "weights" / "best.pt", SAVE_DIR / "weights" / "last.pt"
shutil.copy(BEST, WEIGHTS / f"{TASK}_best.pt")
if LAST.exists():
    shutil.copy(LAST, WEIGHTS / f"{TASK}_last.pt")
print("best weights:", BEST)

# %% [markdown]
# ## 5. Training curves
#
# `results.png` shows, per epoch: train/val **box loss** (localisation), **cls loss**
# (classification), **dfl loss** (box-edge distribution), and val **precision, recall, mAP50,
# mAP50-95**. Healthy run: both losses fall; val loss flattening while train loss keeps falling =
# over-fitting (early stopping / more augmentation / more data).

# %%
show_image(SAVE_DIR / "results.png", "Loss and metric curves", (16, 8))
show_image(SAVE_DIR / "labels.jpg", "Label distribution seen by the trainer", (10, 10))

# %% [markdown]
# ## 6. Evaluate on the held-out TEST set (never used for training or model selection)
#
# * **Precision** = correct detections / all detections (how many alarms are real).
# * **Recall** = found objects / all real objects (how many violators we catch).
# * **F1** = 2PR/(P+R) — balance of both.
# * **mAP50** = mean Average Precision (area under the PR curve) with a box counted correct at IoU ≥ 0.5.
# * **mAP50-95** = mAP averaged over IoU 0.50…0.95 — also rewards *tight* boxes (important for plate crops → OCR).
# * **Confusion matrix** — which class is mistaken for which (e.g. `no_helmet` predicted as `helmet`
#   = a missed violation), and the *background* row/column = false alarms / misses.

# %%
def evaluate(weights, data, split="test", name="test"):
    m = YOLO(str(weights))
    r = m.val(data=str(data), split=split, imgsz=IMGSZ, batch=8 if not SMOKE else 4, conf=EVAL_CONF, iou=EVAL_IOU,
              device=0 if N_GPU else "cpu", plots=True, project=str(RUNS), name=f"{RUN_NAME}_{name}",
              exist_ok=True, workers=WORKERS, verbose=False)
    b = r.box
    per_class = pd.DataFrame({
        "class": [CLASS_NAMES[i] for i in r.ap_class_index],
        "precision": b.p, "recall": b.r, "f1": b.f1, "mAP50": b.ap50, "mAP50-95": b.ap})
    p, rc = float(b.mp), float(b.mr)
    overall = dict(precision=p, recall=rc, f1=2 * p * rc / (p + rc + 1e-9), mAP50=float(b.map50),
                   **{"mAP50-95": float(b.map)}, speed_ms=r.speed)
    return r, overall, per_class


test_res, TEST_METRICS, TEST_PER_CLASS = evaluate(BEST, DATA_YAML, "test", "test")
print("TEST overall:", {k: round(v, 4) if isinstance(v, float) else v for k, v in TEST_METRICS.items()})
print(TEST_PER_CLASS.round(4).to_string(index=False))
EVAL_DIR = Path(test_res.save_dir)
for pattern, title in [("confusion_matrix_normalized.png", "Confusion matrix (normalised)"),
                       ("confusion_matrix.png", "Confusion matrix (counts)"),
                       ("*PR_curve.png", "Precision-Recall curve"), ("*P_curve.png", "Precision vs confidence"),
                       ("*R_curve.png", "Recall vs confidence"), ("*F1_curve.png", "F1 vs confidence")]:
    for f in sorted(EVAL_DIR.glob(pattern))[:1]:
        show_image(f, title)

# %% [markdown]
# ## 7. Accuracy per source and the deployment confidence threshold
#
# A single overall number can hide a weak domain. Evaluating each test source separately shows
# *where* the model fails (e.g. small CCTV heads vs close-up phone photos) → tells us which data to add.
#
# The **deployment confidence threshold** is the confidence at which mean F1 over classes is maximal
# (from the F1 curve). Phase 2 uses it; mAP itself does not depend on a threshold.

# %%
def per_source_eval():
    rows, test = [], MANIFEST[MANIFEST.split == "test"]
    tmp = Path("/tmp/per_source") if IS_KAGGLE else WORK / "per_source"
    tmp.mkdir(parents=True, exist_ok=True)
    for src, g in test.groupby("source"):
        lst = tmp / f"{TASK}_{src}.txt"
        lst.write_text("\n".join(str(DATA_DIR / f) for f in g.file) + "\n")
        y = tmp / f"{TASK}_{src}.yaml"
        y.write_text(yaml.safe_dump({"path": str(DATA_DIR), "train": str(lst), "val": str(lst), "test": str(lst),
                                     "names": CLASS_NAMES}))
        _, ov, _ = evaluate(BEST, y, "test", f"test_{src}")
        rows.append(dict(source=src, images=len(g), **{k: v for k, v in ov.items() if k != "speed_ms"}))
    return pd.DataFrame(rows)


PER_SOURCE = per_source_eval()
print(PER_SOURCE.round(4).to_string(index=False))

try:
    f1c, px = np.asarray(test_res.box.f1_curve), np.asarray(test_res.box.px)
    mean_f1 = f1c.mean(0)
    # a flat / near-zero F1 curve (badly trained model) gives a meaningless argmax -> default 0.25
    BEST_CONF = float(np.clip(px[int(mean_f1.argmax())], 0.05, 0.90)) if mean_f1.max() > 0.05 else 0.25
except Exception as e:
    print("F1 curve not available:", e); BEST_CONF = 0.25
print(f"deployment confidence threshold (max mean F1): {BEST_CONF:.3f}")

# The confusion matrix from section 6 is built at conf=0.001, where many low-confidence guesses
# inflate class swaps. The matrix that describes real behaviour is the one at the deployment threshold.
_ = YOLO(str(BEST)).val(data=str(DATA_YAML), split="test", imgsz=IMGSZ, batch=8 if not SMOKE else 4,
                        conf=BEST_CONF, iou=EVAL_IOU, device=0 if N_GPU else "cpu", plots=True,
                        project=str(RUNS), name=f"{RUN_NAME}_test_deploy_conf", exist_ok=True,
                        workers=WORKERS, verbose=False)
DEPLOY_DIR = Path(_.save_dir)
show_image(DEPLOY_DIR / "confusion_matrix_normalized.png", f"Confusion matrix at deployment conf {BEST_CONF:.2f}")
show_image(DEPLOY_DIR / "confusion_matrix.png", f"Confusion matrix (counts) at deployment conf {BEST_CONF:.2f}")

# %% [markdown]
# ## 8. Visual test on unseen images
#
# Left = ground truth (green), right = prediction at the deployment threshold. Look for: missed small
# heads, helmet/no-helmet swaps, caps/turbans/scarves predicted as helmets, plates missed at angles.

# %%
PALETTE = [(0, 200, 0), (0, 0, 255), (255, 128, 0)]


def draw(im, boxes, labels=None):
    out = im.copy(); h, w = out.shape[:2]
    for i, (c, x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), PALETTE[c % 3], max(2, w // 300))
        txt = labels[i] if labels else CLASS_NAMES[c]
        cv2.putText(out, txt, (int(x1), max(12, int(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, max(0.5, w / 1400), PALETTE[c % 3], 2)
    return out


def visual_test(n=8):
    best_model = YOLO(str(BEST))
    test = MANIFEST[(MANIFEST.split == "test") & (MANIFEST.n_boxes > 0)]
    for f in test.sample(min(n, len(test)), random_state=SEED).file:
        ip = DATA_DIR / f
        im = cv2.imread(str(ip)); h, w = im.shape[:2]
        gt = []
        for line in (DATA_DIR / f.replace("images/", "labels/", 1)).with_suffix(".txt").read_text().splitlines():
            c, cx, cy, bw, bh = line.split(); c = int(c); cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
            gt.append((c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
        r = best_model.predict(str(ip), imgsz=IMGSZ, conf=BEST_CONF, iou=EVAL_IOU, device=0 if N_GPU else "cpu", verbose=False)[0]
        pred = [(int(c), *xyxy) for c, xyxy in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist())]
        labels = [f"{CLASS_NAMES[c]} {s:.2f}" for c, s in zip(r.boxes.cls.int().tolist(), r.boxes.conf.tolist())]
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        ax[0].imshow(draw(im, [(0, *b[1:]) for b in gt], [CLASS_NAMES[b[0]] for b in gt])[..., ::-1]); ax[0].set_title(f"ground truth - {Path(f).name}")
        ax[1].imshow(draw(im, pred, labels)[..., ::-1]); ax[1].set_title(f"prediction (conf >= {BEST_CONF:.2f})")
        for a in ax: a.axis("off")
        fig.savefig(RESULTS / f"visual_{Path(f).stem}.jpg", dpi=80, bbox_inches="tight")
        plt.show() if "ipykernel" in sys.modules else plt.close(fig)


visual_test()

# %% [markdown]
# ## 9. Export and save
#
# * **PyTorch `.pt`** — used by the FastAPI backend (Phase 2) with Ultralytics; supports GPU and CPU.
# * **ONNX** — framework-independent graph; runs with ONNX Runtime on CPU without PyTorch, and is the
#   starting point for TensorRT/OpenVINO acceleration. Exported at the training image size.
#
# Everything is copied to `/kaggle/working/weights` and `/kaggle/working/results/<task>`. After the
# commit finishes, the next notebook (or Phase 2) attaches this notebook's output to use the weights.
# **Download `weights/` to your PC as a backup** after every successful run.

# %%
onnx_path = YOLO(str(BEST)).export(format="onnx", imgsz=IMGSZ, simplify=True, dynamic=False)
shutil.copy(onnx_path, WEIGHTS / f"{TASK}_best.onnx")

summary = dict(task=TASK, model=MODEL_NAME, imgsz=IMGSZ, ultralytics=ultralytics.__version__,
               train_args={k: v for k, v in train_args.items() if k not in ("project",)},
               test=TEST_METRICS, test_per_class=TEST_PER_CLASS.to_dict(orient="records"),
               test_per_source=PER_SOURCE.to_dict(orient="records"), deploy_conf=BEST_CONF,
               weights=sorted(p.name for p in WEIGHTS.glob(f"{TASK}_*")))
(RESULTS / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
for f in list(SAVE_DIR.glob("*.png")) + list(SAVE_DIR.glob("*.csv")) + list(EVAL_DIR.glob("*.png")):
    shutil.copy(f, RESULTS / f.name)
for f in DEPLOY_DIR.glob("confusion_matrix*.png"):
    shutil.copy(f, RESULTS / f"deploy_conf_{f.name}")
print(json.dumps({k: summary[k] for k in ("task", "model", "test", "deploy_conf", "weights")}, indent=2, default=str))
