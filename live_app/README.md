# Live app: emotion from your voice

A static web page: record a voice sample (or upload an audio file) and see the emotion
predicted by the models trained in `cnn_from_scratch.ipynb` and `transfer_learning.ipynb`.
You can pick either model, trained on either **EmoDB only** or **all datasets**.

Everything runs in the browser. There is no server, and the audio never leaves the device:

1. the audio is decoded, resampled to 16 kHz mono and trimmed of silence at the start and end
   (keeping a 0.2 s margin),
2. cut into 3 s segments, each turned into a mel-spectrogram image (`dsp.js`). The output
   is pixel-identical to `mel_to_rgb` in `prepare_*.ipynb`, and the grayscale conversion
   and resize match PIL/torchvision,
3. the segments go through the model, exported to ONNX and run with
   [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/),
4. the per-segment probabilities are averaged.

```
index.html, style.css, app.js   page and UI
dsp.js                          audio -> spectrogram -> tensor (same maths as the notebooks)
models/                         4 ONNX models + models.json (classes, sizes, accuracy) + dsp.json
tools/export_models.py          .pt checkpoints -> models/
```

## Run locally

The page uses ES modules, so it needs a web server (opening the file directly won't work):

```bash
cd live_app
python3 -m http.server 8000
# open http://localhost:8000
```

Browsers only allow microphone access on `https://` or `localhost`.

## Deploy

It's a plain static site with no build step:

- **Vercel**: `cd live_app && npx vercel` (or import the repo and set *Root Directory* to
  `live_app`, framework preset *Other*),
- **Netlify / GitHub Pages / Cloudflare Pages**: publish the `live_app` folder.

## Updating the models

The models come from the checkpoints archived by `run_experiments.py` in `results/emodb/`
and `results/all/`. After retraining, re-export them from the repo root:

```bash
.venv/bin/python -m pip install onnx onnxruntime onnxscript
.venv/bin/python live_app/tools/export_models.py
```

The script checks that each ONNX model gives the same outputs as the PyTorch one.
