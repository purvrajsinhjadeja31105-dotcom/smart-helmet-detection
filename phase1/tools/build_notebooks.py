"""Convert the percent-format sources in phase1/src (# %% cells) into Kaggle-ready .ipynb notebooks.

The .py files are the single source of truth: they are easy to diff, test locally
(`python src/data_prep.py`) and review; the notebooks are generated from them.

Usage:  python tools/build_notebooks.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOBS = {"src/data_prep.py": "notebooks/01_data_prep.ipynb", "src/train.py": "notebooks/02_train.ipynb"}
CELL = re.compile(r"^# %%( \[markdown\])?\s*$")


def split_cells(text: str):
    """Yield (cell_type, source) pairs from a percent-format script."""
    kind, buf = None, []
    for line in text.splitlines():
        m = CELL.match(line)
        if m:
            if kind is not None:
                yield kind, buf
            kind, buf = ("markdown" if m.group(1) else "code"), []
            continue
        if kind is None:
            continue
        buf.append(re.sub(r"^# ?", "", line) if kind == "markdown" else line)
    if kind is not None:
        yield kind, buf


def to_notebook(py_path: Path) -> dict:
    cells = []
    for kind, lines in split_cells(py_path.read_text(encoding="utf-8")):
        while lines and not lines[-1].strip():
            lines.pop()
        while lines and not lines[0].strip():
            lines.pop(0)
        if not lines:
            continue
        src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        cell = {"cell_type": kind, "id": f"cell-{len(cells):03d}", "metadata": {}, "source": src}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "language_info": {"name": "python"}}}


if __name__ == "__main__":
    for src, dst in JOBS.items():
        nb = to_notebook(ROOT / src)
        out = ROOT / dst
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{src} -> {dst} ({len(nb['cells'])} cells)")
