"""Converts the trained checkpoints to ONNX so the web app can run them in the browser.

Run from the repo root, after run_experiments.py has filled results/emodb and results/all:

    .venv/bin/python -m pip install onnx onnxruntime onnxscript
    .venv/bin/python live_app/tools/export_models.py

Writes to live_app/models/:
  - one .onnx per model (input "input" (B, C, H, W) float32, output "logits" (B, n_classes))
  - models.json: the model list (classes, input size, normalisation, validation accuracy)
    and the spectrogram parameters
  - dsp.json: the mel filterbank and the viridis colormap, so the browser builds exactly
    the same images as mel_to_rgb in prepare_*.ipynb
"""
import json
import re
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import cm
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "live_app" / "models"

# (id, label, checkpoint, notebook the accuracy report comes from)
MODELS = [
    ("cnn_emodb", "CNN from scratch · EmoDB",
     "results/emodb/cnn_scratch_emodb.pt", "results/emodb/cnn_from_scratch_report.txt"),
    ("resnet_emodb", "ResNet-18 transfer · EmoDB",
     "results/emodb/resnet18_emodb.pt", "results/emodb/transfer_learning_report.txt"),
    ("cnn_all", "CNN from scratch · all datasets",
     "results/all/cnn_scratch_crema-d_emodb_ravdess_emouerj.pt",
     "results/all/cnn_from_scratch_report.txt"),
    ("resnet_all", "ResNet-18 transfer · all datasets",
     "results/all/resnet18_crema-d_emodb_ravdess_emouerj.pt",
     "results/all/transfer_learning_report.txt"),
]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# same architecture as cnn_from_scratch.ipynb
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.pool(x)


class EmotionCNN(nn.Module):
    def __init__(self, n_classes, in_ch=1):
        super().__init__()
        self.block1 = ConvBlock(in_ch, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.block4 = ConvBlock(128, 256)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.block4(self.block3(self.block2(self.block1(x))))
        return self.fc(self.dropout(torch.flatten(self.gap(x), 1)))


def load(path):
    ckpt = torch.load(ROOT / path, map_location="cpu", weights_only=False)
    n = len(ckpt["classes"])
    if "in_channels" in ckpt:  # only the scratch CNN checkpoint has this field
        model = EmotionCNN(n, ckpt["in_channels"])
        channels, norm = ckpt["in_channels"], ckpt["norm"]
    else:
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, n)
        channels, norm = 3, {"mean": IMAGENET_MEAN, "std": IMAGENET_STD}
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt, channels, norm


def accuracy(report):
    text = (ROOT / report).read_text()
    return float(re.search(r"^\s*accuracy\s+([\d.]+)", text, re.M).group(1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    entries, spec = [], None
    for model_id, label, ckpt_path, report in MODELS:
        model, ckpt, channels, norm = load(ckpt_path)
        spec = spec or ckpt["spec"]
        assert ckpt["spec"] == spec, "all models must use the same spectrogram parameters"

        size = ckpt["img_size"]
        dummy = torch.randn(2, channels, size, size)
        onnx_path = OUT / f"{model_id}.onnx"
        torch.onnx.export(model, dummy, onnx_path, input_names=["input"], output_names=["logits"],
                          dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                          opset_version=17, dynamo=False)

        # the ONNX graph must give the same logits as torch
        with torch.no_grad():
            expected = model(dummy).numpy()
        got = ort.InferenceSession(onnx_path).run(None, {"input": dummy.numpy()})[0]
        err = np.abs(got - expected).max()
        assert err < 1e-3, f"{model_id}: ONNX differs from torch ({err})"

        entries.append({
            "id": model_id, "label": label, "file": onnx_path.name,
            "arch": "resnet" if channels == 3 else "cnn",
            "datasets": ckpt["datasets"], "classes": ckpt["classes"],
            "img_size": size, "channels": channels, "mean": norm["mean"], "std": norm["std"],
            "val_accuracy": accuracy(report),
        })
        print(f"{onnx_path.name}: {onnx_path.stat().st_size / 1e6:.1f} MB, max |torch - onnx| = {err:.1e}")

    (OUT / "models.json").write_text(json.dumps({"spec": spec, "models": entries}, indent=2))

    # mel filterbank (n_mels x (n_fft/2 + 1)), stored sparsely: first non-zero bin + weights
    mel = librosa.filters.mel(sr=spec["sr"], n_fft=spec["n_fft"], n_mels=spec["n_mels"])
    filters = []
    for row in mel:
        nz = np.nonzero(row)[0]
        filters.append({"start": int(nz[0]), "weights": [float(w) for w in row[nz[0]:nz[-1] + 1]]})
    viridis = (cm.viridis(np.arange(256))[:, :3] * 255).astype(np.uint8).tolist()
    (OUT / "dsp.json").write_text(json.dumps({"mel_filters": filters, "viridis": viridis}))
    print("wrote models.json and dsp.json to", OUT)


if __name__ == "__main__":
    main()
