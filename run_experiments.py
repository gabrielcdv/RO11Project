"""Builds the 3 usable_data splits with app.py, then runs both training notebooks on each.

For each experiment:
  1. write data/split_config.yaml, load app.py headless (streamlit AppTest) and click
     "Run train/test split" -> usable_data/{train,test}
  2. execute transfer_learning.ipynb and cnn_from_scratch.ipynb on it
  3. keep the executed notebooks, the split labels, the confusion matrices and the
     checkpoints in results/<name>/

Usage: .venv/bin/python run_experiments.py [name ...]
"""
import base64
import os
import shutil
import time
import sys
from pathlib import Path

import nbformat
import yaml
from nbclient import NotebookClient
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "data/split_config.yaml"
USABLE = ROOT / "usable_data"
RESULTS = ROOT / "results"
ALL_DATASETS = ["CREMA-D", "EmoDB", "RAVDESS", "emoUERJ"]
NOTEBOOKS = ["transfer_learning", "cnn_from_scratch"]
# kernels started headless default to Agg: plt.show() would then produce no image output
os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"

# 20 % of the speakers (with all their recordings) go to test, the others to train
SPEAKER_STAGE = [{"column": "speaker", "pct": 80, "mode": "group"}]
SEED = 42

EXPERIMENTS = {
    "emodb": ["EmoDB"],
    "cremad": ["CREMA-D"],
    "all": ALL_DATASETS,
}


def make_split(keep):
    excluded = [d for d in ALL_DATASETS if d not in keep]
    config = {
        "exclusions": [{"column": "dataset", "values": excluded}] if excluded else [],
        "force_test": [],
        "stages": SPEAKER_STAGE,
        "seed": SEED,
    }
    CONFIG.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    shutil.rmtree(USABLE, ignore_errors=True)  # app.py never empties it
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=1800)
    at.run()
    [button] = [b for b in at.button if b.label == "Run train/test split"]
    button.click().run()
    if at.exception:
        raise RuntimeError(at.exception)
    for msg in at.success:
        print("  app.py:", msg.value)
    for msg in at.warning:
        print("  app.py WARNING:", msg.value)


def run_notebook(nb_name, out_dir):
    nb = nbformat.read(ROOT / f"{nb_name}.ipynb", as_version=4)
    NotebookClient(nb, timeout=None, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.write(nb, out_dir / f"{nb_name}.ipynb")

    for cell in nb.cells:
        if cell.cell_type != "code" or "ConfusionMatrixDisplay(" not in cell.source:
            continue
        text = "".join(o.get("text", "") for o in cell.outputs if o.output_type == "stream")
        (out_dir / f"{nb_name}_report.txt").write_text(text)
        for o in cell.outputs:
            if "image/png" in o.get("data", {}):
                (out_dir / f"{nb_name}_confusion.png").write_bytes(
                    base64.b64decode(o["data"]["image/png"]))


def main(names):
    RESULTS.mkdir(exist_ok=True)
    # the notebooks write their checkpoint at the repo root: protect the existing ones
    backup = RESULTS / "_backup"
    backup.mkdir(exist_ok=True)
    originals = [p for p in [CONFIG, *ROOT.glob("*.pt")] if p.exists()]
    for p in originals:
        shutil.copy2(p, backup / p.name)
    before = set(ROOT.glob("*.pt"))

    try:
        for name in names:
            print(f"=== {name}: {EXPERIMENTS[name]}")
            out_dir = RESULTS / name
            out_dir.mkdir(parents=True, exist_ok=True)
            make_split(EXPERIMENTS[name])
            for split in ["train", "test"]:
                shutil.copy2(USABLE / split / "labels.csv", out_dir / f"{split}_labels.csv")
            for nb_name in NOTEBOOKS:
                print(f"  running {nb_name}.ipynb")
                start = time.time()
                run_notebook(nb_name, out_dir)
                for ckpt in ROOT.glob("*.pt"):
                    if ckpt.stat().st_mtime >= start:
                        shutil.copy2(ckpt, out_dir / ckpt.name)
    finally:
        for p in set(ROOT.glob("*.pt")) - before:
            p.unlink()
        for p in originals:
            shutil.copy2(backup / p.name, p)
        shutil.rmtree(backup)


if __name__ == "__main__":
    main(sys.argv[1:] or list(EXPERIMENTS))
