"""Build a categorised demo test set from the held-out TEST split (images the models never trained on).

Usage:
    python phase1/tools/build_demo_dataset.py <datasets_dir> <out_dir> [--per 20] [--extra-real <folder>]

<datasets_dir> is the data-prep output folder holding helmet/ and plate/ (manifest.csv + images/labels).
Writes <out_dir>/<category>/*.jpg, answer_key.csv (ground truth per image) and README.txt.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

SEED = 42


def labels_of(root: Path, rel: str) -> list[tuple[int, float, float, float, float]]:
    lp = (root / rel.replace("images/", "labels/", 1)).with_suffix(".txt")
    out = []
    for line in lp.read_text().splitlines():
        c, x, y, w, h = line.split()
        out.append((int(c), float(x), float(y), float(w), float(h)))
    return out


def brightness(img: np.ndarray) -> float:
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2].mean())


# synthetic conditions (same family as the training augmentation, stronger, clearly labelled as synthetic)
def fog(im, rng):
    a = rng.uniform(0.55, 0.7) * np.linspace(1.0, 0.6, im.shape[0], dtype=np.float32)[:, None, None]
    return np.clip(im * (1 - a) + 225 * a, 0, 255).astype(np.uint8)


def blur(im, rng):
    k = int(rng.integers(11, 19)) | 1
    ker = np.zeros((k, k), np.float32); ker[k // 2, :] = 1.0 / k
    M = cv2.getRotationMatrix2D((k / 2 - .5, k / 2 - .5), rng.uniform(0, 180), 1.0)
    ker = cv2.warpAffine(ker, M, (k, k)); ker /= max(ker.sum(), 1e-6)
    return cv2.filter2D(im, -1, ker)


def rain(im, rng):
    h, w = im.shape[:2]
    layer = np.zeros((h, w), np.uint8)
    length = max(10, int(h * 0.04)); ang = math.radians(rng.uniform(-12, 12))
    for _ in range(int(0.004 * h * w / 10)):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.line(layer, (x, y), (int(x + length * math.sin(ang)), int(y + length * math.cos(ang))), 210, 1)
    out = im.astype(np.float32) * 0.8 + cv2.blur(layer, (3, 3))[..., None].astype(np.float32) * 0.7
    return cv2.GaussianBlur(np.clip(out, 0, 255).astype(np.uint8), (3, 3), 0)


def low_light(im, rng):
    out = 255.0 * (im / 255.0) ** rng.uniform(2.2, 2.8) * 0.7 + rng.normal(0, 6, im.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--per", type=int, default=20)
    ap.add_argument("--extra-real", type=Path, help="folder of new real-world photos to include as a category")
    ap.add_argument("--video", type=Path, help="a demo video to include")
    ap.add_argument("--kaggle-manifests", type=Path,
                    help="folder with manifest_helmet.txt / manifest_plate.txt from the Kaggle run (test-split check)")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    rnd = random.Random(SEED)

    h_root, p_root = a.datasets / "helmet", a.datasets / "plate"
    hm = [r for r in csv.DictReader(open(h_root / "manifest.csv", encoding="utf-8"))
          if r["split"] == "test" and r["kind"] == "original"]
    pm = [r for r in csv.DictReader(open(p_root / "manifest.csv", encoding="utf-8"))
          if r["split"] == "test" and r["kind"] == "original"]
    if a.kaggle_manifests:
        # keep only images that the Kaggle training run also put in its test split
        def kaggle_test(task):
            keep = set()
            for r in csv.DictReader(open(a.kaggle_manifests / f"manifest_{task}.txt", encoding="utf-8")):
                if r["split"] == "test" and r["kind"] == "original":
                    keep.add((r["source"], Path(r["orig"].replace("\\", "/")).name))
            return keep
        kh, kp = kaggle_test("helmet"), kaggle_test("plate")
        orig_key = lambda r: (r["source"], Path(r["orig"].replace("\\", "/")).name)
        hm = [r for r in hm if orig_key(r) in kh]
        pm = [r for r in pm if orig_key(r) in kp]
    rnd.shuffle(hm); rnd.shuffle(pm)

    if a.out.exists():
        shutil.rmtree(a.out)
    key, used = [], set()
    cats = {
        "01_no_helmet": "Riders without helmets (expected: VIOLATION)",
        "02_helmet_safe": "All riders wear helmets (expected: SAFE)",
        "03_mixed": "Some riders with and some without helmets in one image",
        "04_crowded_traffic": "Dense traffic with 6 or more riders' heads",
        "05_night_dark": "Real dark / night images (low brightness)",
        "06_number_plates": "Clearly visible number plates (plate-model test set)",
        "07_fog_synthetic": "Test images with strong synthetic fog added",
        "08_motion_blur_synthetic": "Test images with synthetic motion blur",
        "09_rain_synthetic": "Test images with synthetic rain streaks",
        "10_low_light_synthetic": "Test images darkened synthetically",
    }

    def put(cat, src_root, row, img=None, note=""):
        d = a.out / cat
        d.mkdir(parents=True, exist_ok=True)
        name = Path(row["file"]).name
        if img is None:
            shutil.copy(src_root / row["file"], d / name)
        else:
            cv2.imwrite(str(d / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        labs = labels_of(src_root, row["file"])
        nh = sum(1 for l in labs if l[0] == 0) if src_root == h_root else ""
        nn = sum(1 for l in labs if l[0] == 1) if src_root == h_root else ""
        npl = len(labs) if src_root == p_root else ""
        key.append({"category": cat, "file": f"{cat}/{name}", "source_dataset": row["source"],
                    "true_helmet_heads": nh, "true_no_helmet_heads": nn, "true_plates": npl, "note": note})

    # real categories, highest-priority first, each image used once
    stats = []
    for r in hm:
        im = cv2.imread(str(h_root / r["file"]))
        stats.append((r, int(r["n_helmet"]), int(r["n_no_helmet"]), brightness(im)))
    rules = [
        ("05_night_dark", lambda h, n, b: b < 75 and h + n > 0),
        ("04_crowded_traffic", lambda h, n, b: h + n >= 6),
        ("03_mixed", lambda h, n, b: h > 0 and n > 0),
        ("01_no_helmet", lambda h, n, b: n > 0 and h == 0),
        ("02_helmet_safe", lambda h, n, b: h > 0 and n == 0),
    ]
    for cat, rule in rules:
        count = 0
        for r, h, n, b in stats:
            if count >= a.per or r["file"] in used:
                continue
            if rule(h, n, b):
                put(cat, h_root, r, note=f"brightness {b:.0f}" if cat == "05_night_dark" else "")
                used.add(r["file"]); count += 1

    # clear plates: plate box at least 6% of the image width
    count = 0
    for r in pm:
        if count >= a.per:
            break
        labs = labels_of(p_root, r["file"])
        if labs and max(l[3] for l in labs) >= 0.06:
            put("06_number_plates", p_root, r); count += 1

    # synthetic conditions on unused test images that contain heads
    pool = [r for r, h, n, b in stats if r["file"] not in used and h + n > 0 and b >= 75]
    per_syn = max(5, a.per // 2)
    for cat, fn in (("07_fog_synthetic", fog), ("08_motion_blur_synthetic", blur),
                    ("09_rain_synthetic", rain), ("10_low_light_synthetic", low_light)):
        for r in pool[:per_syn]:
            im = cv2.imread(str(h_root / r["file"]))
            put(cat, h_root, r, fn(im.astype(np.float32) if fn in (fog, low_light) else im, rng), note="synthetic")
        pool = pool[per_syn:]

    if a.extra_real and a.extra_real.is_dir():
        d = a.out / "11_real_world_new"
        d.mkdir(parents=True, exist_ok=True)
        cats["11_real_world_new"] = "New real street photos (Pexels), not from any dataset; no ground truth"
        for f in sorted(a.extra_real.glob("*")):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                shutil.copy(f, d / f.name)
                key.append({"category": "11_real_world_new", "file": f"11_real_world_new/{f.name}",
                            "source_dataset": "pexels", "true_helmet_heads": "", "true_no_helmet_heads": "",
                            "true_plates": "", "note": "no ground truth"})
    if a.video and a.video.is_file():
        d = a.out / "12_video"
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(a.video, d / a.video.name)
        cats["12_video"] = "Demo video: 2 riders without helmets, plate KL 09 AQ 3439"

    with open(a.out / "answer_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0].keys()))
        w.writeheader(); w.writerows(key)
    counts = {c: len(list((a.out / c).glob("*"))) if (a.out / c).is_dir() else 0 for c in cats}
    lines = ["DEMO TEST DATASET - Smart Helmet Detection", "",
             "Every image in categories 01-10 comes from the held-out TEST split: the models never saw",
             "these images (or near-copies of them) during training, so results here match the reported",
             "test accuracy. Synthetic categories (07-10) are test images with fog / blur / rain / darkness",
             "added by a program. answer_key.csv lists the true number of helmet and no-helmet heads",
             "(and plates) per image, so any result can be checked against the ground truth.", "",
             "How to test: drag any image onto demo.bat, or open start_app.bat and upload it.", ""]
    lines += [f"{c:<28} {counts.get(c, 0):>3} files  {d}" for c, d in cats.items()]
    (a.out / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
