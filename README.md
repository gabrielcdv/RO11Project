# CNN_RO11 Speech emotion recognition

This project classifies the emotion in a short speech recording, using mel-spectrograms
as input to convolutional networks. It's built from five public datasets
(CREMA-D, EmoDB, RAVDESS, emoUERJ, MESD) and compares two models on classification.

The project was built in stages, each one building on the output of the previous one.

## 1. Getting and preparing the data : `prepare_*.ipynb`

Each dataset has its own quirks (file naming, sample rate, metadata format), so each one
gets its own notebook: `prepare_cremad.ipynb`, `prepare_emoDB.ipynb`,
`prepare_ravdess_audio.ipynb`, `prepare_ravdess_video.ipynb`, `prepare_emouerj.ipynb`,
`prepare_mesd.ipynb`. Despite the different sources, they all converge on the same
pipeline:

1. download/locate the raw corpus,
2. resample the audio to 16 kHz,
3. cut every recording into 3-second chunks (achunk under 1 s is dropped),
4. turn each chunk into a mel-spectrogram, saved as a PNG.

Every chunk produced by every notebook is registered as one row of
`data/spectrograms/labels.csv`, with the emotion label, the speaker, the source
dataset, and whatever metadata that corpus provides (language, gender, transcript...).


## 2. Building custom training sets: `app.py`

With five datasets and nine possible emotions all mixed into one CSV, we needed a way to
pick and choose which slice of the data to actually train on — a subset of dataset,
a specific train/test split strategy, excluding some values without hand-editing
code every time. `app.py` is a Streamlit app (built with AI assistance) that does
exactly that:

- it loads `data/spectrograms/labels.csv` and lets you explore the dataset visually
  (counts by emotion, by corpus, by speaker, etc.);
- it lets you define exclusion rules (e.g. "drop everything except EmoDB") and a
  train/test split (row-wise or, more usefully, grouped by speaker so that a given
  voice never appears in both train and test);
- clicking "Run train/test split" writes `usable_data/train/` and `usable_data/test/`
  (images + `labels.csv`), which is what the training notebooks actually read.

Run with:

```bash
streamlit run app.py
```

## 3. Training two models : `transfer_learning.ipynb` and `cnn_from_scratch.ipynb`

Both notebooks read whatever split is currently sitting in `usable_data/`, so they work
on any combination of dataset the app produced, and both save a checkpoint (`.pt`) with
everything needed to reuse the model later in the live app.

- **`transfer_learning.ipynb`**: starts from a ResNet-18 pretrained on ImageNet, first
  trains only a new classification head on top of the frozen backbone, then unfreezes
  `layer4` for fine-tuning at a lower learning rate. RGB input, 224×224.
- **`cnn_from_scratch.ipynb`**: a small CNN with no pretrained weights — 4 blocks of
  two 3×3 convolutions + BatchNorm + ReLU + max-pool (1→32→64→128→256 channels), global
  average pooling, dropout, one linear layer (~1.2 M parameters). Grayscale input,
  128×128.

We ran both notebooks by hand on a few different dataset configurations (single
dataset, then everything combined) to check they behaved sensibly before automating
the whole comparison.

## 4. Automating the comparison : `run_experiments.py`

Once we were happy with both notebooks individually, we handed the whole pipeline to
an AI to script end-to-end: for each dataset configuration, `run_experiments.py`
drives `app.py` with no interface to (re)build the split, then
executes both notebooks on it, and archives the executed notebooks, the split's label
files, the confusion matrices and the checkpoints into `results/<config>/`.

```bash
.venv/bin/python run_experiments.py            # all configs
.venv/bin/python run_experiments.py emodb      # a single one
```

## 5. Results

All outputs are written
up in [RESULTS.md](RESULTS.md).

## 6. Live app

As requested in class, we made a live app (with AI) that is using our trained models on a voice sample that one can upload or record directly within the app

Link : [https://ro11-murex.vercel.app/](https://ro11-murex.vercel.app/) (inference can take more than 10s)
