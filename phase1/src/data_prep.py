# %% [markdown]
# # Phase 1 · Notebook 1 — Data Preparation (Helmet & Plate datasets)
#
# **Goal:** turn several public datasets (different class names, formats, duplicates, label errors)
# into two clean, leak-free YOLO datasets:
#
# | Dataset | Classes | Used by |
# |---|---|---|
# | `datasets/helmet` | `helmet`, `no_helmet` (head-level boxes) | Model A |
# | `datasets/plate`  | `number_plate` | Model B |
#
# Motorcycles and persons are **not** trained here: a COCO-pretrained YOLO already detects them very
# accurately, and our sources disagree on what a "rider" box covers (per person vs per vehicle).
# Training on conflicting definitions would lower accuracy, so the pipeline (Phase 2) links
# heads and plates to motorcycles geometrically instead.
#
# **Run settings on Kaggle:** Accelerator = *None* (CPU is enough), Internet = **ON**.
# Optional inputs: attach the **EdgeVision** dataset (uploaded by you) and add the Kaggle secret
# `ROBOFLOW_API_KEY` (Add-ons → Secrets) to include the two Roboflow datasets.
#
# **Output:** everything under `/kaggle/working/datasets`. Click *Save Version → Save & Run All*;
# the training notebook then attaches this notebook's output (Add Input → Notebook Output).
#
# **Pipeline in this notebook**
# ```
# download → read labels → map classes → clean labels → drop corrupt/unlabelled images
#   → hash images → remove duplicates → group similar scenes → split (train/val/test)
#   → cap backgrounds → write YOLO files → oversample minority class → weather augmentation (train)
#   → EDA plots + cleaning report
# ```

# %% [markdown]
# ## 1. Setup and configuration
#
# * Every tunable value lives in this one cell so experiments are reproducible (`SEED = 42`).
# * `SOURCES` describes each dataset: where it comes from, which **tasks** it is complete for
#   (a dataset is used for a task only if it labels *every* object of that task — otherwise its
#   unlabelled objects would teach the model that helmets/plates are background), how its class
#   names map to ours (`None` = drop), and whether it is Indian (only Indian images go to the test
#   set, because Indian roads are the deployment target).

# %%
import os, re, io, sys, json, glob, math, time, shutil, random, zipfile, hashlib, warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter, defaultdict
from urllib.request import urlopen, Request

import numpy as np
import pandas as pd
import cv2
import yaml
import matplotlib
if not hasattr(sys, "ps1") and "ipykernel" not in sys.modules:
    matplotlib.use("Agg")                      # headless when run as a plain script
import matplotlib.pyplot as plt

IS_KAGGLE = Path("/kaggle/working").exists()
WORK = Path("/kaggle/working") if IS_KAGGLE else Path(os.environ.get("HELMET_WORK", "_work")).resolve()
RAW = Path("/tmp/raw") if IS_KAGGLE else WORK / "raw"          # downloads (not saved to output)
OUT = WORK / "datasets"                                         # final datasets (saved to output)
EDA = WORK / "eda"
LOCAL_RAW = os.environ.get("HELMET_LOCAL_RAW")                  # local testing: pre-downloaded sources

SEED = 42
TASKS = {"helmet": ["helmet", "no_helmet"], "plate": ["number_plate"]}
HEAD_CLASSES = {"helmet", "no_helmet"}

MAX_SIDE = 1920            # longer images are downscaled (labels are normalised, so they stay valid)
MIN_SIDE = 128             # smaller images carry too little detail
MIN_BOX_PX = 4             # boxes thinner than this (in pixels) are annotation noise
MAX_HEAD_AREA = 0.40       # a "head" box covering >40% of the image is a labelling error
SOURCE_HEAD_MEDIAN_MAX = 0.10   # if a source's median head box is larger, it boxes bodies, not heads
CONFLICT_IOU = 0.60        # helmet & no_helmet on the same head  -> ambiguous image, dropped
DUPBOX_IOU = 0.90          # same-class boxes overlapping this much are one object labelled twice
DUP_PHASH, DUP_DHASH = 6, 10    # near-duplicate image thresholds (Hamming distance, 64-bit hashes)
DUP_BOX_MATCH = 0.6        # ...and >=60% of their boxes must coincide (else: same camera, other riders)
GROUP_PHASH = 12           # looser threshold: "same scene" images are kept but forced into one split
TEST_PCT, VAL_PCT = 10, 10 # test only from Indian sources; val from all sources
BACKGROUND_MAX_FRAC = 0.10 # images without any box: keep at most 10% (YOLO guideline 0-10%)
OVERSAMPLE_TARGET = 0.60   # minority class instances raised to >=60% of majority (train only)
OVERSAMPLE_MAX_COPIES = 2
WEATHER_FRAC = 0.20        # fraction of train images that get one extra weather-augmented copy
JPEG_QUALITY = 95

SOURCES = [
    dict(name="aryan", kind="kaggle", ref="aryanvaid13/indian-helmet-detection-dataset",
         tasks=["helmet", "plate"], indian=True,
         # data.yaml names are '0'..'4'; meaning verified visually (see README)
         class_map={"0": "number_plate", "1": "no_helmet", "2": "helmet", "3": "helmet", "4": None}),
    dict(name="anushka", kind="kaggle", ref="anushkagovindkadam/smart-helmet-detection-using-dl",
         tasks=["helmet"], indian=True,
         class_map={"driver_with_helmet": "helmet", "passenger_with_helemt": "helmet",
                    "driver_without_helmet": "no_helmet", "passenger_without_helemt": "no_helmet",
                    "bike": None, "driver": None, "passenger": None}),
    dict(name="anees", kind="kaggle", ref="aneesarom/rider-with-helmet-without-helmet-number-plate",
         tasks=["helmet", "plate"], indian=True,
         class_map={"with helmet": "helmet", "without helmet": "no_helmet",
                    "rider": None, "number plate": "number_plate"}),
    dict(name="edgevision", kind="input", pattern="*edgevision*", tasks=["helmet"], indian=True,
         class_map={"bikewithrider": None, "helmet": "helmet", "nohelmet": "no_helmet"}),
    dict(name="bhumik", kind="roboflow", workspace="bhumik", project="helmet-numberplate", version=2,
         tasks=["helmet", "plate"], indian=False,
         class_map={"helmet": "helmet", "no_helmet": "no_helmet", "rider": None,
                    "number_plate": "number_plate"}),
    dict(name="miniproject", kind="roboflow", workspace="miniproject-dfm83",
         project="helmetdetectiondataset", version=2, tasks=["helmet", "plate"], indian=False,
         class_map={"rider": None, "number plate": "number_plate", "with helmet": "helmet",
                    "without helmet": "no_helmet"}),
    dict(name="pkdarabi", kind="kaggle", ref="pkdarabi/helmet", tasks=["plate"], indian=False,
         # WithHelmet/WithoutHelmet box the whole rider, not the head -> plate task only
         class_map={"plate": "number_plate", "withhelmet": None, "withouthelmet": None}),
    dict(name="deepak", kind="kaggle", ref="deepakat002/indian-vehicle-number-plate-yolo-annotation",
         tasks=["plate"], indian=True, group_by_folder=True,      # frames of 3 videos
         class_map={"number_plate": "number_plate"}),
    # --- plate improvement round (added after the first plate model: 76.8% mAP50 on crops) ---
    dict(name="trafficviol", kind="kaggle", ref="devgurucodes/trafffic-violations-triple-riding-no-helmet-plate",
         tasks=["plate"], indian=True, require_boxes=True,
         # rider classes box whole riders (like pkdarabi); some sub-sources never label plates, so only
         # images with at least one plate box are used (others would teach "plate = background")
         class_map={"plate": "number_plate", "withhelmet": None, "withouthelmet": None, "tripleriding": None}),
    dict(name="mpsuu", kind="kaggle", ref="majorprojectssuu/motorcycle-rider-helmet-license-plate",
         tasks=["plate"], indian=True,
         # darknet labels without a names file; ids verified visually: 0 rider+bike, 1 plate, 2 helmet
         # (no "no helmet" class -> not usable for the helmet task)
         class_map={"#0": None, "#1": "number_plate", "#2": None}),
    dict(name="gujarat", kind="kaggle", ref="paneraghanshyam/gujarat-vehicle-number-plates-yolo-ready",
         tasks=["plate"], indian=True, class_map={"plate": "number_plate"}),
    # --- helmet improvement round: rear-view / open-face helmets (real-photo test found false
    # "no_helmet" on helmets seen from behind). These two sources label WHOLE riders as
    # WithHelmet / WithoutHelmet and show many riders from behind. Head boxes are produced by the
    # current helmet model and get the rider's (human) label - see section 4b. Machine-labelled, so
    # train_only: they never enter val/test, which keeps the test set identical for a fair comparison.
    dict(name="rear_tv", kind="reuse", reuse="trafficviol", tasks=["helmet"], indian=False,
         train_only=True, pseudo=True,
         class_map={"withhelmet": "rider_helmet", "withouthelmet": "rider_nohelmet", "plate": None,
                    "tripleriding": None}),
    dict(name="rear_pk", kind="reuse", reuse="pkdarabi", tasks=["helmet"], indian=False,
         train_only=True, pseudo=True,
         class_map={"withhelmet": "rider_helmet", "withouthelmet": "rider_nohelmet", "plate": None}),
]
SOURCE_RANK = {s["name"]: i for i, s in enumerate(SOURCES)}     # duplicate tie-break: earlier wins

random.seed(SEED); np.random.seed(SEED)
# flush every print immediately, so Kaggle's "Logs" tab shows live progress during a commit run
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:          # Jupyter's output stream has no reconfigure(); it flushes on its own
    pass
for d in (RAW, OUT, EDA):
    d.mkdir(parents=True, exist_ok=True)
print("Kaggle:", IS_KAGGLE, "| work dir:", WORK, "| OpenCV", cv2.__version__)


def show(fig, name):
    """Save a figure into the EDA folder and display it (inline in notebooks)."""
    fig.savefig(EDA / name, dpi=110, bbox_inches="tight")
    plt.show() if "ipykernel" in sys.modules else plt.close(fig)

# %% [markdown]
# ## 2. Download the datasets
#
# * **Kaggle** datasets: downloaded as zip from Kaggle's public download endpoint (works in
#   *Save & Run All* commits, where `kagglehub` cannot attach new datasets). If you attached a dataset
#   manually with *Add Input*, the mounted copy is used instead.
# * **Roboflow** datasets: the public REST export API with your key from Kaggle Secrets. We call the
#   API directly instead of installing the `roboflow` pip package, because that package pins old
#   versions of OpenCV/NumPy and can break the Kaggle environment.
# * **EdgeVision** (Mendeley) blocks scripted downloads, so you download it in a browser, upload it as
#   a private Kaggle dataset, and attach it. If it is not attached it is skipped with a warning.
#
# A source that fails to download is skipped — the notebook never stops because one source is missing.

# %%
def http_get(url, timeout=120):
    with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=timeout) as r:
        return r.read()


def get_secret(name):
    if os.environ.get(name):
        return os.environ[name]
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None


def find_input(pattern, max_depth=4):
    """Attached Kaggle inputs are mounted at varying depths, e.g. /kaggle/input/<slug> or
    /kaggle/input/datasets/<owner>/<slug>; search a few levels for a folder matching `pattern`."""
    for depth in range(max_depth):
        hits = sorted(p for p in glob.glob("/kaggle/input/" + "*/" * depth + pattern) if Path(p).is_dir())
        if hits:
            return Path(hits[0])
    return None


def download_file(url, dest, timeout=1800):
    with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=timeout) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)


def extract_all_zips(folder):
    """Extract every zip under `folder` (also zips inside zips) next to itself, once."""
    for _ in range(3):
        zips = [z for z in folder.rglob("*.zip") if not z.with_suffix("").exists()]
        if not zips:
            return
        for z in zips:
            zipfile.ZipFile(z).extractall(z.with_suffix(""))


def fetch_kaggle(src):
    slug = src["ref"].split("/")[1]
    if LOCAL_RAW and (Path(LOCAL_RAW) / slug).exists():
        return Path(LOCAL_RAW) / slug
    mounted = find_input(slug)                    # attached manually via "Add Input"
    if mounted:
        return mounted
    # Public download endpoint: works in commit (non-interactive) runs, unlike kagglehub.
    dest = RAW / src["name"]
    if not (dest.exists() and any(dest.rglob("*.txt"))):
        dest.mkdir(parents=True, exist_ok=True)
        zpath = RAW / f"{src['name']}.zip"
        download_file(f"https://www.kaggle.com/api/v1/datasets/download/{src['ref']}", zpath)
        zipfile.ZipFile(zpath).extractall(dest)
        zpath.unlink()
    return dest


def fetch_roboflow(src):
    key = get_secret("ROBOFLOW_API_KEY")
    if not key:
        raise RuntimeError("secret ROBOFLOW_API_KEY not set")
    dest = RAW / src["name"]
    if dest.exists() and any(dest.rglob("*.txt")):
        return dest
    api = f"https://api.roboflow.com/{src['workspace']}/{src['project']}/{src['version']}/yolov8?api_key={key}"
    link = None
    for _ in range(30):                       # Roboflow may need a minute to generate the export
        info = json.loads(http_get(api))
        link = info.get("export", {}).get("link")
        if link:
            break
        time.sleep(10)
    if not link:
        raise RuntimeError(f"Roboflow export not ready: {str(info)[:200]}")
    dest.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(http_get(link, timeout=1800))).extractall(dest)
    return dest


def fetch_input(src):
    root = find_input(src["pattern"])
    if root is None and LOCAL_RAW:
        hits = glob.glob(str(Path(LOCAL_RAW) / src["pattern"]))
        root = Path(hits[0]) if hits else None
    if root is None:
        raise RuntimeError(f"no attached input matching {src['pattern']} - check 'Add Input'")
    if not any(root.rglob("*.zip")):              # Kaggle already unpacked the upload
        return root
    dest = RAW / src["name"]                      # /kaggle/input is read-only -> unpack to /tmp
    if not dest.exists():
        shutil.copytree(root, dest, ignore=shutil.ignore_patterns("*.jpg", "*.jpeg", "*.png"))
        extract_all_zips(dest)
    return dest


print("attached inputs:", sorted(glob.glob("/kaggle/input/*")) + sorted(glob.glob("/kaggle/input/*/*"))[:20])
def fetch_reuse(src):
    """Second view of an already downloaded source (same files, different class mapping)."""
    if src["reuse"] not in ROOTS:
        raise RuntimeError(f"source {src['reuse']!r} was not downloaded")
    return ROOTS[src["reuse"]]


FETCHERS = {"kaggle": fetch_kaggle, "roboflow": fetch_roboflow, "input": fetch_input, "reuse": fetch_reuse}
ROOTS, SKIPPED = {}, {}
for src in SOURCES:
    try:
        ROOTS[src["name"]] = FETCHERS[src["kind"]](src)
        print(f"[ok]   {src['name']:<12} {ROOTS[src['name']]}")
    except Exception as e:
        SKIPPED[src["name"]] = f"download failed: {e}"
        print(f"[skip] {src['name']:<12} {e}")

# %% [markdown]
# ## 3. Read every annotation into one common structure
#
# **YOLO format** (what we convert everything to): one `.txt` per image, one line per object
# `class x_center y_center width height`, all values normalised to 0–1 by image width/height.
# **Pascal VOC** (EdgeVision option) stores absolute pixel corners `xmin ymin xmax ymax` in XML — we
# convert it. Roboflow sometimes exports **polygons** (`class x1 y1 x2 y2 …`); we convert those to
# their bounding box.
#
# Class names are read from each dataset's own `data.yaml` / `classes.txt` and normalised
# (lower-case, `_`/`-` → space) before mapping, so `driver_with_helmet` and `Driver With Helmet` match.
# A source with a class we do not know is **excluded** (an unknown class might be a helmet/plate we
# would otherwise silently lose).

# %%
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
norm = lambda s: re.sub(r"[\s_\-]+", " ", str(s).strip().lower())


def read_names(path):
    """Class names from a data.yaml (list or dict) or a one-name-per-line text file."""
    if path.suffix in {".yaml", ".yml"}:
        names = (yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore")) or {}).get("names")
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names, key=int)]
        return [str(n) for n in names] if names else None
    lines = [l.strip() for l in path.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
    return lines or None


_names_cache = {}
def names_for(label_path, root):
    """Walk up from the label folder to the dataset root and use the first names file found."""
    d = label_path.parent
    while True:
        if d not in _names_cache:
            found = None
            for fn in ("data.yaml", "data.yml", "classes.txt", "obj.names", "_darknet.labels"):
                if (d / fn).exists():
                    found = read_names(d / fn)
                    if found:
                        break
            _names_cache[d] = found
        if _names_cache[d] or d == root or d.parent == d:
            return _names_cache[d]
        d = d.parent


def label_path_for(img):
    parts = list(img.parts)
    if "images" in parts:
        i = len(parts) - 1 - parts[::-1].index("images")
        cand = Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")
        if cand.exists():
            return cand
    cand = img.with_suffix(".txt")
    return cand if cand.exists() else None


def parse_yolo(lp):
    """Returns (rows, malformed) where rows = [(class_id, cx, cy, w, h)]."""
    rows, bad = [], 0
    for line in lp.read_text(encoding="utf-8", errors="ignore").splitlines():
        t = line.split()
        if not t:
            continue
        try:
            c, v = int(float(t[0])), [float(x) for x in t[1:]]
        except ValueError:
            bad += 1; continue
        if len(v) == 4:
            rows.append((c, *v))
        elif len(v) >= 6 and len(v) % 2 == 0:           # polygon -> bounding box
            xs, ys = v[0::2], v[1::2]
            rows.append((c, (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)))
        else:
            bad += 1
    return rows, bad


def iter_yolo(root):
    for img in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT and "__MACOSX" not in p.parts):
        lp = label_path_for(img)
        if lp is None:
            yield img, None, None, 0
            continue
        rows, bad = parse_yolo(lp)
        names = names_for(lp, root)
        yield img, [(names[c] if names and 0 <= c < len(names) else f"#{c}", *r) for c, *r in rows], names, bad


def iter_voc(root):
    by_stem = {p.stem: p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT}
    for xp in sorted(root.rglob("*.xml")):
        try:
            tree = ET.parse(xp).getroot()
        except ET.ParseError:
            continue
        fn = Path(tree.findtext("filename") or xp.stem).stem
        img = by_stem.get(fn) or by_stem.get(xp.stem)
        if img is None:
            continue
        objs = []
        for o in tree.iter("object"):
            b = o.find("bndbox")
            if b is None:
                continue
            x1, y1, x2, y2 = (float(b.findtext(k) or 0) for k in ("xmin", "ymin", "xmax", "ymax"))
            objs.append((o.findtext("name") or "", x1, y1, x2, y2))     # absolute pixels, normalised later
        yield img, objs, "voc", 0


RAW_ITEMS, STATS = [], defaultdict(Counter)
for src in SOURCES:
    name = src["name"]
    if name not in ROOTS:
        continue
    root = Path(ROOTS[name])
    cmap = {norm(k): v for k, v in src["class_map"].items()}
    is_voc = any(root.rglob("*.xml")) and not any(root.rglob("labels/*.txt"))
    unknown, items = Counter(), []
    for img, objs, names, bad in (iter_voc(root) if is_voc else iter_yolo(root)):
        STATS[name]["found"] += 1
        if objs is None:
            STATS[name]["no_label_file"] += 1; continue
        if bad:
            STATS[name]["malformed_label"] += 1; continue   # a broken line may hide a lost box
        boxes = []
        for cname, *coords in objs:
            key = norm(cname)
            if key not in cmap:
                unknown[cname] += 1; continue
            if cmap[key]:
                boxes.append((cmap[key], *coords))
        items.append(dict(source=name, img=str(img), boxes=boxes, voc=is_voc,
                          folder=str(img.parent.relative_to(root))))
    if unknown:
        SKIPPED[name] = f"unmapped classes {dict(unknown)} - add them to class_map"
        print(f"[skip] {name}: {SKIPPED[name]}")
        continue
    RAW_ITEMS += items
    print(f"{name:<12} images={STATS[name]['found']:>5} usable={len(items):>5} format={'VOC' if is_voc else 'YOLO'}")

# %% [markdown]
# ## 4. Validate images and clean labels
#
# For every image: decode it with OpenCV (the same decoder Ultralytics uses, so EXIF rotation is
# handled identically) → drop corrupt/tiny images. For every box: clip to the image, drop boxes
# thinner than `MIN_BOX_PX`, merge same-class double labels (IoU > 0.9).
# Image-level rules for the helmet task (these images are dropped because we cannot know the truth):
# * a head box covering > 40% of the image (visual check found whole-image "helmet" boxes);
# * `helmet` and `no_helmet` drawn on the same head (IoU > 0.6).
#
# We also compute two **perceptual hashes** per image here (pHash, dHash, plus their mirrored
# versions) for duplicate detection in the next step, and a "black border" score that identifies
# Roboflow's rotated augmented copies.

# %%
def to_xyxy(b, w, h, voc):
    if voc:
        _, x1, y1, x2, y2 = b
        return x1 / w, y1 / h, x2 / w, y2 / h
    _, cx, cy, bw, bh = b
    return cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def phash_dhash(gray):
    """64-bit perceptual hash (DCT low frequencies) and difference hash (gradient signs)."""
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8].flatten()
    ph = dct[1:] > np.median(dct[1:])
    ph = np.concatenate([[False], ph])
    d = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    dh = (d[:, 1:] > d[:, :-1]).flatten()
    pack = lambda bits: int(np.packbits(bits.astype(np.uint8)).view(">u8")[0])
    return pack(ph), pack(dh)


def clean_item(it):
    im = cv2.imread(it["img"], cv2.IMREAD_COLOR)
    if im is None:
        return "corrupt"
    h, w = im.shape[:2]
    if min(h, w) < MIN_SIDE:
        return "too_small"
    boxes = []
    for b in it["boxes"]:
        x1, y1, x2, y2 = (min(max(v, 0.0), 1.0) for v in to_xyxy(b, w, h, it["voc"]))
        if (x2 - x1) * w < MIN_BOX_PX or (y2 - y1) * h < MIN_BOX_PX:
            STATS[it["source"]]["box_too_small"] += 1; continue
        cand = (b[0], x1, y1, x2, y2)
        if any(o[0] == cand[0] and iou(o[1:], cand[1:]) > DUPBOX_IOU for o in boxes):
            STATS[it["source"]]["box_duplicate"] += 1; continue
        boxes.append(cand)
    heads = [b for b in boxes if b[0] in HEAD_CLASSES]
    it["bad_head_area"] = any((b[3] - b[1]) * (b[4] - b[2]) > MAX_HEAD_AREA for b in heads)
    it["head_conflict"] = any(a[0] != b[0] and iou(a[1:], b[1:]) > CONFLICT_IOU
                              for i, a in enumerate(heads) for b in heads[i + 1:])
    it["boxes"], it["w"], it["h"] = boxes, w, h
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    it["ph"], it["dh"] = phash_dhash(gray)
    it["ph_f"], it["dh_f"] = phash_dhash(cv2.flip(gray, 1))
    m = max(2, int(0.03 * min(h, w)))
    border = np.concatenate([gray[:m].ravel(), gray[-m:].ravel(), gray[:, :m].ravel(), gray[:, -m:].ravel()])
    it["black_border"] = float((border < 8).mean())
    with open(it["img"], "rb") as f:
        it["sha1"] = hashlib.sha1(f.read()).hexdigest()
    return "ok"


ITEMS = []
t0 = time.time()
for i, it in enumerate(RAW_ITEMS):
    status = clean_item(it)
    if status != "ok":
        STATS[it["source"]][status] += 1
    else:
        ITEMS.append(it)
    if (i + 1) % 1000 == 0:
        print(f"  {i + 1}/{len(RAW_ITEMS)} images checked ({time.time() - t0:.0f}s)")
print(f"valid images: {len(ITEMS)} / {len(RAW_ITEMS)}")

# Source-level guard: a source whose "head" boxes are body-sized must not train the helmet model.
for src in SOURCES:
    areas = [(b[3] - b[1]) * (b[4] - b[2]) for it in ITEMS if it["source"] == src["name"]
             for b in it["boxes"] if b[0] in HEAD_CLASSES]
    if "helmet" in src["tasks"] and areas and np.median(areas) > SOURCE_HEAD_MEDIAN_MAX:
        src["tasks"] = [t for t in src["tasks"] if t != "helmet"]
        print(f"[guard] {src['name']}: median head box area {np.median(areas):.3f} - boxes are bodies, "
              f"source removed from the helmet task")

# %% [markdown]
# ## 4b. Head labels for rear-view riders (weak supervision)
#
# **Problem found in real-photo tests:** helmets seen *from behind* and open-face helmets were
# predicted as `no_helmet` - such views are rare in our head-labelled data.
# **Data that has them:** `trafficviol` and `pkdarabi` show many riders from behind, but label the
# whole rider (`WithHelmet` / `WithoutHelmet`), not the head.
# **Method:** the current helmet model (Phase 1, 89.1% mAP50) proposes head boxes (low threshold,
# both classes). Each head inside the upper half of a labelled rider box gets the **human** label of
# that rider. Where the model was wrong (e.g. said `no_helmet` on a helmet from behind) this is
# exactly the correction we want the new model to learn.
#
# Safety rules (an image is dropped when any rule fails - a missing or wrong head label is worse
# than a missing image):
# * every rider box must contain at least one proposed head;
# * a `WithoutHelmet` box with several heads keeps the model's per-head classes and must contain at
#   least one `no_helmet` (the dataset does not say which person is bare-headed);
# * a head outside all rider boxes is kept only if the model is confident (>= 0.5); an uncertain
#   unexplained head (0.25-0.5) drops the image;
# * rider boxes must be taller than wide (a wide box = a rotated augmentation copy, where "upper half
#   of the rider" is not where the head is); of each group of augmented copies the un-rotated one is used.
# These images are `train_only`: val/test stay 100% human-labelled and identical to Phase 1.
#
# Requires the trained helmet model: attach the **helmet-train** notebook output as an input (it
# contains `weights/helmet_best.pt`). Turn on a GPU for this notebook: labelling ~5k images takes
# a few minutes on a T4 (about 1-2 hours on CPU).

# %%
# head proposals below 0.25 were sometimes not heads at all (a wheel rim was boxed at ~0.2)
PSEUDO_CONF, PSEUDO_KEEP_OUTSIDE, PSEUDO_UNCERTAIN = 0.25, 0.50, 0.25
# at most this many rear-view images are labelled (a random, seeded sample) - bounds the run time;
# 2,000 rear-view images are plenty to teach the missing viewpoint
PSEUDO_LIMIT = int(os.environ.get("HELMET_PSEUDO_LIMIT", "2000"))
_RF = re.compile(r"^(.+?)_(jpg|jpeg|png)\.rf\.[0-9a-f]{20,}", re.I)


def find_helmet_model():
    if os.environ.get("HELMET_PSEUDO_MODEL"):
        return Path(os.environ["HELMET_PSEUDO_MODEL"])
    hits = sorted(glob.glob("/kaggle/input/**/helmet_best.pt", recursive=True))
    return Path(hits[0]) if hits else None


def pseudo_label(items, model_path):
    """Replace rider boxes by labelled head boxes; return the items that pass the safety rules."""
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    # one image per Roboflow base name (augmented copies only repeat the same labels); prefer the copy
    # with the least black border = the un-rotated original
    groups = {}
    for it in items:
        stem = Path(it["img"]).stem
        m = _RF.match(stem)
        key = (it["source"], (m.group(1) if m else stem).lower())
        riders = [b for b in it["boxes"] if b[0].startswith("rider_")]
        # real riders are taller than wide; a wide rider box means a rotated augmentation copy
        if riders and all((b[4] - b[2]) * it["h"] >= 0.9 * (b[3] - b[1]) * it["w"] for b in riders):
            groups.setdefault(key, []).append(it)
        else:
            STATS[it["source"]]["pseudo_rejected_rotated_or_no_rider"] += 1
    unique = [min(g, key=lambda x: x["black_border"]) for g in groups.values()]
    if PSEUDO_LIMIT and len(unique) > PSEUDO_LIMIT:
        unique = random.Random(SEED).sample(unique, PSEUDO_LIMIT)
    print(f"  labelling {len(unique)} rear-view images (device: GPU if available)")
    t_start = time.time()
    kept, examples = [], []
    for k in range(0, len(unique), 16):
        chunk = unique[k:k + 16]
        results = model.predict([it["img"] for it in chunk], imgsz=960, conf=PSEUDO_CONF, iou=0.7,
                                batch=8, verbose=False)
        for it, res in zip(chunk, results):
            h, w = res.orig_shape
            heads = [(model.names[int(c)], float(s), x1 / w, y1 / h, x2 / w, y2 / h)
                     for (x1, y1, x2, y2), s, c in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(),
                                                       res.boxes.cls.tolist())]
            riders = [b for b in it["boxes"] if b[0].startswith("rider_")]
            boxes, used, ok, corrected = [], set(), True, 0
            for lab, rx1, ry1, rx2, ry2 in riders:
                rh = ry2 - ry1
                inside = [i for i, (_, _, x1, y1, x2, y2) in enumerate(heads) if i not in used
                          and rx1 <= (x1 + x2) / 2 <= rx2 and ry1 - 0.05 * rh <= (y1 + y2) / 2 <= ry1 + 0.5 * rh]
                if not inside:
                    ok = False; break
                if lab == "rider_helmet":
                    labels = ["helmet"] * len(inside)
                elif len(inside) == 1:
                    labels = ["no_helmet"]
                else:
                    labels = [heads[i][0] for i in inside]
                    if "no_helmet" not in labels:
                        ok = False; break
                for i, l in zip(inside, labels):
                    used.add(i)
                    corrected += l != heads[i][0]
                    boxes.append((l, *heads[i][2:]))
            if ok:
                for i, (cls, s, *xyxy) in enumerate(heads):
                    if i in used:
                        continue
                    if s >= PSEUDO_KEEP_OUTSIDE:
                        boxes.append((cls, *xyxy))
                    elif s >= PSEUDO_UNCERTAIN:
                        ok = False; break
            if not ok or not boxes:
                STATS[it["source"]]["pseudo_rejected"] += 1
                continue
            it["boxes"], it["bad_head_area"], it["head_conflict"] = boxes, False, False
            it["pseudo_corrected"] = corrected
            STATS[it["source"]]["pseudo_kept"] += 1
            STATS[it["source"]]["pseudo_heads_corrected"] += corrected
            kept.append(it)
            if corrected and len(examples) < 12:
                examples.append(it)
        if (k // 16) % 10 == 0:
            print(f"  pseudo-labelling: {min(k + 16, len(unique))}/{len(unique)} images "
                  f"({time.time() - t_start:.0f}s)")
    return kept, examples


PSEUDO_EXAMPLES = []
pseudo_sources = {s["name"] for s in SOURCES if s.get("pseudo")}
if pseudo_sources:
    helmet_model = find_helmet_model()
    pseudo_items = [it for it in ITEMS if it["source"] in pseudo_sources]
    ITEMS = [it for it in ITEMS if it["source"] not in pseudo_sources]
    if helmet_model is None or not helmet_model.exists():
        for name in pseudo_sources:
            SKIPPED[name] = "helmet_best.pt not found - attach the helmet-train notebook output"
        print("[skip] rear-view sources: no helmet model found (attach the helmet-train output)")
    elif pseudo_items:
        print(f"pseudo-labelling {len(pseudo_items)} images with {helmet_model}")
        kept, PSEUDO_EXAMPLES = pseudo_label(pseudo_items, helmet_model)
        ITEMS += kept
        print(f"rear-view images kept: {len(kept)} | head labels corrected vs. the current model: "
              f"{sum(it['pseudo_corrected'] for it in kept)}")

# %% [markdown]
# ## 5. Duplicates, scene groups and the train / val / test split
#
# **Why this matters most for honest accuracy:** if a near-copy of a test image is in training, the
# test score measures memorisation, not generalisation. Our sources copy each other (e.g. `anushka`
# contains `BikesHelmets*` images from another dataset, `- Copy` files) and Roboflow exports contain
# 3 augmented copies per image (`name_jpg.rf.<hash>.jpg`).
#
# 1. **Exact / near duplicates** — same file bytes, same Roboflow base name, or pHash ≤ 6 **and**
#    dHash ≤ 10 (also vs. mirrored hashes, to catch flipped copies) **and** ≥ 60% of the label boxes
#    coincide. The box check matters: frames from a fixed traffic camera have almost identical hashes
#    but show *different riders* — those are valuable, not duplicates. One image per cluster is kept: the
#    earliest source in `SOURCES`, then the unrotated one (least black border), then most boxes.
# 2. **Scene groups** — images that are merely similar (pHash ≤ 12, or frames of the same video) are
#    all kept but always placed in the **same** split.
# 3. **Split** — each group gets a bucket 0–99 from the MD5 of its key, so an image always lands in
#    the same split even if datasets are added later (stable, frozen test set).
#    Indian sources: 10% test / 10% val / 80% train. Other sources: 10% val / 90% train.
#
# *Why 80/10/10 rather than 70/20/10?* With ~7k images, 10% (~700) is already a statistically stable
# validation set; every extra image in training improves accuracy more than a larger val set would.

# %%
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_rows(a, B):
    """Hamming distance between each hash in a (k,) and all hashes in B (n,) -> (k, n)."""
    x = (a[:, None] ^ B[None, :]).view(np.uint8).reshape(len(a), len(B), 8)
    return POP[x].sum(-1, dtype=np.uint16)


class DSU:
    def __init__(self, n): self.p = list(range(n))
    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]; i = self.p[i]
        return i
    def union(self, a, b): self.p[self.find(a)] = self.find(b)


RF_BASE = re.compile(r"^(.+?)_(jpg|jpeg|png|bmp|webp)\.rf\.[0-9a-f]{20,}", re.I)


def base_name(path):
    """Shared name of Roboflow augmented copies / '- Copy' files; None for ordinary names
    (ordinary names like 'frame-170' repeat across folders without being duplicates)."""
    stem = Path(path).stem
    plain = re.sub(r"\s*-\s*copy(\s*\(\d+\))?$", "", stem, flags=re.I)
    m = RF_BASE.match(plain)
    if m:
        return m.group(1).lower()
    return plain.lower() if plain != stem else None


def boxes_match(A, B, flipped=False, thr=0.5):
    """Share of boxes that coincide (IoU > thr) between two images. Images from a fixed camera
    have near-identical hashes but different riders -> low match -> NOT duplicates."""
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    if flipped:
        B = [(c, 1 - x2, y1, 1 - x1, y2) for c, x1, y1, x2, y2 in B]
    used, hit = set(), 0
    for a in A:
        best = max(((iou(a[1:], b[1:]), j) for j, b in enumerate(B) if j not in used), default=(0, -1))
        if best[0] > thr:
            used.add(best[1]); hit += 1
    return hit / max(len(A), len(B))


def build_task(task):
    items = [it for it in ITEMS if task in next(s["tasks"] for s in SOURCES if s["name"] == it["source"])]
    items = [dict(it, boxes=[b for b in it["boxes"] if b[0] in TASKS[task]]) for it in items]
    need = {s["name"] for s in SOURCES if s.get("require_boxes")}
    for it in items:
        if it["source"] in need and not it["boxes"]:
            STATS[it["source"]][f"{task}_dropped_unlabelled"] += 1
    items = [it for it in items if it["boxes"] or it["source"] not in need]
    if task == "helmet":
        for it in items:
            if it["bad_head_area"] or it["head_conflict"]:
                reason = "bad_head_area" if it["bad_head_area"] else "head_conflict"
                STATS[it["source"]][f"helmet_dropped_{reason}"] += 1
                AUDIT.append(dict(task=task, reason=reason, image=it["img"], kept=""))
        items = [it for it in items if not (it["bad_head_area"] or it["head_conflict"])]
    n = len(items)
    PH = np.array([it["ph"] for it in items], np.uint64); DH = np.array([it["dh"] for it in items], np.uint64)
    PHF = np.array([it["ph_f"] for it in items], np.uint64); DHF = np.array([it["dh_f"] for it in items], np.uint64)

    dup, grp = DSU(n), DSU(n)
    by_key = defaultdict(list)
    for i, it in enumerate(items):
        by_key[("sha", it["sha1"])].append(i)
        if base_name(it["img"]):
            by_key[("rf", it["source"], base_name(it["img"]))].append(i)
    for idx in by_key.values():
        for j in idx[1:]:
            dup.union(idx[0], j)
    for s in range(0, n, 512):
        e = min(n, s + 512)
        dp, dd = hamming_rows(PH[s:e], PH), hamming_rows(DH[s:e], DH)
        dpf, ddf = hamming_rows(PH[s:e], PHF), hamming_rows(DH[s:e], DHF)
        near = (dp <= DUP_PHASH) & (dd <= DUP_DHASH)
        near_f = (dpf <= DUP_PHASH) & (ddf <= DUP_DHASH)
        similar = (dp <= GROUP_PHASH) | (dpf <= GROUP_PHASH)
        for mask, flipped in ((near, False), (near_f, True)):
            for a, b in zip(*np.nonzero(mask)):
                i = s + a
                if i >= b:
                    continue
                A, B = items[i]["boxes"], items[b]["boxes"]
                if not A and not B:          # nothing to compare -> require near-identical pixels
                    ok = (dpf if flipped else dp)[a, b] <= 2
                else:
                    ok = boxes_match(A, B, flipped) >= DUP_BOX_MATCH
                if ok:
                    dup.union(i, b)
        for a, b in zip(*np.nonzero(similar)):
            if s + a < b: grp.union(s + a, b)

    clusters = defaultdict(list)
    for i in range(n):
        clusters[dup.find(i)].append(i)
    keep = []
    for members in clusters.values():
        best = min(members, key=lambda i: (SOURCE_RANK[items[i]["source"]], round(items[i]["black_border"], 2),
                                           -len(items[i]["boxes"]), -items[i]["w"] * items[i]["h"]))
        keep.append(best)
        for i in members:
            if i != best:
                STATS[items[i]["source"]][f"{task}_duplicate_removed"] += 1
                AUDIT.append(dict(task=task, reason="duplicate", image=items[i]["img"], kept=items[best]["img"]))
    keep.sort()

    src_cfg = {s["name"]: s for s in SOURCES}
    for i in keep:                                   # video frames: whole video in one split
        if src_cfg[items[i]["source"]].get("group_by_folder"):
            key = ("folder", items[i]["source"], items[i]["folder"])
            by_key[key].append(i)
    for k, idx in by_key.items():
        if k[0] == "folder":
            for j in idx[1:]:
                grp.union(idx[0], j)
    group_members = defaultdict(list)
    for i in keep:
        group_members[grp.find(i)].append(i)

    rows = []
    train_only = lambda i: src_cfg[items[i]["source"]].get("train_only", False)
    for members in group_members.values():
        # "train_only" sources (machine-labelled) never influence a group's key or eligibility, so the
        # val/test split of all other images is exactly the same as without them.
        regular = [i for i in members if not train_only(i)]
        if not regular:
            split, gkey = "train", min(items[i]["sha1"] for i in members)
        else:
            gkey = min(items[i]["sha1"] for i in regular)
            bucket = int(hashlib.md5(gkey.encode()).hexdigest()[:8], 16) % 100
            # a group's test eligibility requires every member to come from an Indian source
            indian = all(src_cfg[items[i]["source"]]["indian"] for i in regular)
            split = "test" if (indian and bucket < TEST_PCT) else "val" if TEST_PCT <= bucket < TEST_PCT + VAL_PCT else "train"
        for i in members:
            if train_only(i) and split != "train":       # similar to a val/test image -> would leak
                STATS[items[i]["source"]][f"{task}_dropped_leak_guard"] += 1
                continue
            rows.append(dict(items[i], split=split, group=gkey[:12]))

    # cap background images (no boxes) per split
    out = []
    for split in ("train", "val", "test"):
        part = [r for r in rows if r["split"] == split]
        bg = [r for r in part if not r["boxes"]]
        fg = [r for r in part if r["boxes"]]
        max_bg = int(BACKGROUND_MAX_FRAC * len(fg) / (1 - BACKGROUND_MAX_FRAC))
        random.Random(SEED).shuffle(bg)
        for r in bg[max_bg:]:
            STATS[r["source"]][f"{task}_background_capped"] += 1
        out += fg + bg[:max_bg]
    return out


SPLITS, AUDIT = {}, []
print(time.strftime("%H:%M:%S"), "- duplicates + split ...")
for task in TASKS:
    SPLITS[task] = build_task(task)
    c = Counter(r["split"] for r in SPLITS[task])
    print(f"{task:<7} kept={len(SPLITS[task]):>5}  train={c['train']}  val={c['val']}  test={c['test']}")

# %% [markdown]
# ## 6. Write the YOLO datasets, oversample the minority class, weather augmentation
#
# * Images are re-encoded as RGB JPEG (removes alpha channels / odd palettes such as the RGBA
#   images in `anees`) and downscaled to ≤ 1920 px on the long side. File names keep the source
#   prefix so we can later measure accuracy **per source**.
# * **Class imbalance:** in training only, images that contain the minority class are copied
#   (up to 2×) until it reaches ≥ 60% of the majority class count. Online augmentation (mosaic,
#   scale, colour) makes each copy look different during training. Val/test are never touched.
# * **Weather augmentation (offline, train only):** 20% of training images get one extra copy with a
#   single random effect: fog, rain, low-light, motion blur or sensor noise. These conditions are
#   exactly what the deployed system faces and what Ultralytics' online augmentation does not
#   simulate. Geometric/colour augmentations (flip, scale, rotation, perspective, HSV brightness,
#   mosaic) are applied online by Ultralytics during training — see the training notebook.

# %%
def wx_fog(im, rng):
    h, w = im.shape[:2]
    strength = rng.uniform(0.35, 0.65)
    yy = np.linspace(1.0, 0.4, h, dtype=np.float32)[:, None, None]        # denser haze near horizon
    haze = np.full_like(im, int(rng.uniform(200, 240)), dtype=np.uint8).astype(np.float32)
    a = strength * yy
    return np.clip(im.astype(np.float32) * (1 - a) + haze * a, 0, 255).astype(np.uint8)


def wx_rain(im, rng):
    h, w = im.shape[:2]
    layer = np.zeros((h, w), np.uint8)
    n = int(rng.uniform(0.0015, 0.004) * h * w / 10)
    length = max(8, int(h * rng.uniform(0.02, 0.05)))
    angle = rng.uniform(-15, 15)
    dx = int(length * math.sin(math.radians(angle))); dy = int(length * math.cos(math.radians(angle)))
    xs = rng.integers(0, w, n); ys = rng.integers(0, h, n)
    for x, y in zip(xs, ys):
        cv2.line(layer, (int(x), int(y)), (int(x + dx), int(y + dy)), 200, 1)
    layer = cv2.blur(layer, (3, 3))
    out = (im.astype(np.float32) * 0.85)                                 # rainy scenes are darker
    out += layer[..., None].astype(np.float32) * 0.6
    return cv2.GaussianBlur(np.clip(out, 0, 255).astype(np.uint8), (3, 3), 0)


def wx_lowlight(im, rng):
    gamma = rng.uniform(1.8, 3.0)
    out = 255.0 * (im.astype(np.float32) / 255.0) ** gamma * rng.uniform(0.6, 0.9)
    out += rng.normal(0, rng.uniform(4, 10), im.shape)                    # dark images are noisy
    return np.clip(out, 0, 255).astype(np.uint8)


def wx_motion_blur(im, rng):
    k = int(rng.integers(7, 17)) | 1
    kernel = np.zeros((k, k), np.float32); kernel[k // 2, :] = 1.0 / k
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k)); kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(im, -1, kernel)


def wx_noise(im, rng):
    return np.clip(im.astype(np.float32) + rng.normal(0, rng.uniform(8, 20), im.shape), 0, 255).astype(np.uint8)


WEATHER = {"fog": wx_fog, "rain": wx_rain, "lowlight": wx_lowlight, "motionblur": wx_motion_blur, "noise": wx_noise}


def yolo_lines(boxes, names):
    out = []
    for c, x1, y1, x2, y2 in boxes:
        out.append(f"{names.index(c)} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
    return "\n".join(out) + ("\n" if out else "")


def write_task(task, rows):
    names = TASKS[task]
    root = OUT / task
    if root.exists():
        shutil.rmtree(root)
    for sp in ("train", "val", "test"):
        (root / "images" / sp).mkdir(parents=True); (root / "labels" / sp).mkdir(parents=True)
    rng = np.random.default_rng(SEED)
    counters, manifest = Counter(), []

    def save(im, boxes, split, stem, row, kind):
        cv2.imwrite(str(root / "images" / split / f"{stem}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        (root / "labels" / split / f"{stem}.txt").write_text(yolo_lines(boxes, names))
        manifest.append(dict(file=f"images/{split}/{stem}.jpg", split=split, source=row["source"], kind=kind,
                             group=row["group"], orig=row["img"], n_boxes=len(boxes),
                             **{f"n_{c}": sum(b[0] == c for b in boxes) for c in names}))

    train_rows = [r for r in rows if r["split"] == "train"]
    inst = Counter(b[0] for r in train_rows for b in r["boxes"])
    copies = {}
    if len(names) > 1 and inst:
        major = max(inst.values())
        for c in names:
            if 0 < inst[c] < OVERSAMPLE_TARGET * major:
                copies[c] = min(OVERSAMPLE_MAX_COPIES, math.ceil(OVERSAMPLE_TARGET * major / inst[c]) - 1)
    print(f"{task}: train instances {dict(inst)} -> oversample copies per minority image {copies}")

    for row in rows:
        im = cv2.imread(row["img"], cv2.IMREAD_COLOR)
        s = MAX_SIDE / max(im.shape[:2])
        if s < 1:
            im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
        counters[row["source"]] += 1
        stem = f"{row['source']}_{counters[row['source']]:05d}"
        save(im, row["boxes"], row["split"], stem, row, "original")
        if row["split"] != "train":
            continue
        k = max([copies.get(b[0], 0) for b in row["boxes"]], default=0)
        for j in range(k):
            save(im, row["boxes"], "train", f"{stem}_os{j + 1}", row, "oversample")
        if rng.random() < WEATHER_FRAC:
            effect = list(WEATHER)[int(rng.integers(len(WEATHER)))]
            save(WEATHER[effect](im, rng), row["boxes"], "train", f"{stem}_wx_{effect}", row, f"weather_{effect}")

    (root / "data.yaml").write_text(yaml.safe_dump(
        {"train": "images/train", "val": "images/val", "test": "images/test",
         "nc": len(names), "names": names}, sort_keys=False))
    df = pd.DataFrame(manifest)
    df.to_csv(root / "manifest.csv", index=False)
    return df


print(time.strftime("%H:%M:%S"), "- writing datasets ...")
MANIFESTS = {task: write_task(task, SPLITS[task]) for task in TASKS}
print(time.strftime("%H:%M:%S"), "- datasets written")

# %% [markdown]
# ## 7. EDA — what the final data looks like
#
# Check these before training:
# * **Class distribution per split** — every split must contain every class.
# * **Box size** — head/plate boxes are small; this is why we train at 960 px instead of 640.
# * **Image resolution per source** — very small images (e.g. 300×168) limit what can be learned.
# * **Sample grid** — labels must sit exactly on heads/plates after all conversions.
# * **Cleaning report** — how many images each rule removed, per source.

# %%
for task, df in MANIFESTS.items():
    names = TASKS[task]
    if df.empty:
        print(f"\n=== {task} === no images - check the skipped sources above"); continue
    print(f"\n=== {task} ===")
    tab = df.groupby(["split", "kind"]).size().unstack(fill_value=0)
    print("images per split / kind:\n", tab.to_string())
    inst = df.groupby("split")[[f"n_{c}" for c in names]].sum()
    print("box instances per split:\n", inst.to_string())
    print("original images per source / split:\n",
          df[df.kind == "original"].groupby(["source", "split"]).size().unstack(fill_value=0).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    inst.T.plot.bar(ax=axes[0], title=f"{task}: instances per class & split", rot=0)
    df[df.kind == "original"].groupby("source").size().plot.bar(ax=axes[1], title=f"{task}: original images per source", rot=30)
    sizes = []
    for f in df[df.kind == "original"]["file"]:
        lp = (OUT / task / f.replace("images/", "labels/", 1)).with_suffix(".txt")
        for line in lp.read_text().splitlines():
            c, _, _, bw, bh = line.split()
            sizes.append((names[int(c)], math.sqrt(float(bw) * float(bh))))
    sz = pd.DataFrame(sizes, columns=["cls", "rel_size"])
    for c in names:
        axes[2].hist(sz[sz.cls == c].rel_size, bins=50, alpha=0.6, label=c)
    axes[2].set_title(f"{task}: box size (sqrt of area / image area)"); axes[2].legend()
    show(fig, f"{task}_distribution.png")

# image resolution per source (raw, before resizing)
res = pd.DataFrame([dict(source=it["source"], w=it["w"], h=it["h"]) for it in ITEMS])
fig, ax = plt.subplots(figsize=(8, 6))
for s, g in res.groupby("source"):
    ax.scatter(g.w, g.h, s=6, alpha=0.5, label=s)
ax.set_xlabel("width"); ax.set_ylabel("height"); ax.set_title("Image resolution per source"); ax.legend()
show(fig, "resolution.png")

# %%
COLORS = [(0, 200, 0), (0, 0, 255), (255, 128, 0)]


def sample_grid(task, n=12, split="train"):
    df = MANIFESTS[task]
    if df.empty:
        return
    pick = df[(df.split == split) & (df.kind == "original") & (df.n_boxes > 0)].sample(
        min(n, int(((df.split == split) & (df.kind == "original") & (df.n_boxes > 0)).sum())), random_state=SEED)
    tiles = []
    for f in pick.file:
        im = cv2.imread(str(OUT / task / f)); h, w = im.shape[:2]
        for line in (OUT / task / f.replace("images/", "labels/", 1)).with_suffix(".txt").read_text().splitlines():
            c, cx, cy, bw, bh = line.split(); c = int(c)
            cx, cy, bw, bh = (float(v) for v in (cx, cy, bw, bh))
            p1 = (int((cx - bw / 2) * w), int((cy - bh / 2) * h)); p2 = (int((cx + bw / 2) * w), int((cy + bh / 2) * h))
            cv2.rectangle(im, p1, p2, COLORS[c], max(2, w // 300))
            cv2.putText(im, TASKS[task][c], (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, max(0.5, w / 1400), COLORS[c], 2)
        s = 400 / max(h, w); im = cv2.resize(im, (int(w * s), int(h * s)))
        tile = np.zeros((400, 400, 3), np.uint8); tile[:im.shape[0], :im.shape[1]] = im; tiles.append(tile)
    while len(tiles) % 4:
        tiles.append(np.zeros((400, 400, 3), np.uint8))
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    fig = plt.figure(figsize=(16, 4 * len(tiles) // 4)); plt.imshow(grid[..., ::-1]); plt.axis("off")
    plt.title(f"{task} - {split} samples with final labels")
    show(fig, f"{task}_{split}_samples.png")


for task in TASKS:
    sample_grid(task)


def removed_grid(reason, per_source=4):
    """Show images a cleaning rule removed (with their original boxes) so the rule can be verified."""
    tiles = []
    for src in sorted({it["source"] for it in ITEMS}):
        flag = "bad_head_area" if reason == "bad_head_area" else "head_conflict"
        picks = [it for it in ITEMS if it["source"] == src and it.get(flag)][:per_source]
        for it in picks:
            im = cv2.imread(it["img"]); h, w = im.shape[:2]
            for c, x1, y1, x2, y2 in it["boxes"]:
                col = (0, 200, 0) if c == "helmet" else (0, 0, 255) if c == "no_helmet" else (255, 128, 0)
                cv2.rectangle(im, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), col, max(2, w // 200))
            cv2.putText(im, src, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, max(0.7, w / 800), (255, 255, 255), 2)
            s = 360 / max(h, w); im = cv2.resize(im, (int(w * s), int(h * s)))
            t = np.zeros((360, 360, 3), np.uint8); t[:im.shape[0], :im.shape[1]] = im; tiles.append(t)
    if not tiles:
        return
    while len(tiles) % 4:
        tiles.append(np.zeros((360, 360, 3), np.uint8))
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    fig = plt.figure(figsize=(16, 4 * len(tiles) // 4)); plt.imshow(grid[..., ::-1]); plt.axis("off")
    plt.title(f"Removed by rule '{reason}' (green=helmet, red=no_helmet, orange=plate)")
    show(fig, f"removed_{reason}.png")


removed_grid("bad_head_area")
removed_grid("head_conflict")


def pseudo_grid():
    """Rear-view images whose head label was corrected (model said one class, rider label says the other)."""
    if not PSEUDO_EXAMPLES:
        return
    tiles = []
    for it in PSEUDO_EXAMPLES:
        im = cv2.imread(it["img"]); h, w = im.shape[:2]
        for c, x1, y1, x2, y2 in it["boxes"]:
            col = (0, 200, 0) if c == "helmet" else (0, 0, 255)
            cv2.rectangle(im, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), col, max(2, w // 200))
            cv2.putText(im, c, (int(x1 * w), max(15, int(y1 * h) - 4)), cv2.FONT_HERSHEY_SIMPLEX, max(0.5, w / 900), col, 2)
        s = 360 / max(h, w); im = cv2.resize(im, (int(w * s), int(h * s)))
        t = np.zeros((360, 360, 3), np.uint8); t[:im.shape[0], :im.shape[1]] = im; tiles.append(t)
    while len(tiles) % 4:
        tiles.append(np.zeros((360, 360, 3), np.uint8))
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    fig = plt.figure(figsize=(16, 4 * len(tiles) // 4)); plt.imshow(grid[..., ::-1]); plt.axis("off")
    plt.title("Rear-view pseudo labels where the current model was corrected (green=helmet, red=no_helmet)")
    show(fig, "pseudo_label_corrections.png")


pseudo_grid()

# weather augmentation examples
ex = MANIFESTS["helmet"][MANIFESTS["helmet"].kind.str.startswith("weather")].groupby("kind").head(1)     if not MANIFESTS["helmet"].empty else []
if len(ex):
    fig, axes = plt.subplots(1, len(ex), figsize=(4 * len(ex), 4))
    for ax, (_, r) in zip(np.atleast_1d(axes), ex.iterrows()):
        ax.imshow(cv2.imread(str(OUT / "helmet" / r.file))[..., ::-1]); ax.set_title(r.kind); ax.axis("off")
    show(fig, "weather_examples.png")

# %%
report = pd.DataFrame(STATS).T.fillna(0).astype(int)
report.index.name = "source"
print("Cleaning report (counts per rule):\n", report.to_string())
print("\nSkipped sources:", SKIPPED or "none")
summary = dict(seed=SEED, skipped=SKIPPED, cleaning=report.to_dict(orient="index"),
               tasks={t: ({k: int(v) for k, v in m.groupby("split").size().items()} if not m.empty else {})
                      for t, m in MANIFESTS.items()},
               config=dict(MAX_SIDE=MAX_SIDE, DUP_PHASH=DUP_PHASH, DUP_DHASH=DUP_DHASH, GROUP_PHASH=GROUP_PHASH,
                           TEST_PCT=TEST_PCT, VAL_PCT=VAL_PCT, WEATHER_FRAC=WEATHER_FRAC,
                           OVERSAMPLE_TARGET=OVERSAMPLE_TARGET))
pd.DataFrame(AUDIT).to_csv(OUT / "audit_removed.csv", index=False)   # every removed image + reason
(OUT / "report.json").write_text(json.dumps(summary, indent=2, default=str))
print("\nDone. Datasets written to", OUT)
