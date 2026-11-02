# PISA-FFT

Research code for image-based permeability prediction in porous media.

This repository contains a modular prototype combining three-channel microstructure images, patch-level morphological descriptors, gated visual–physical feature fusion, and Fourier-domain feature processing.

## Repository contents

```text
PISA-FFT/
├── models/fft_gated.py        # Patch embeddings, gated morphology fusion, spectral encoder
├── data/patch_loader.py       # Image/CSV loading and patch-level descriptors
├── trainers/runner.py         # Training, checkpoint selection, and test metrics
├── scripts/run_patch_pred.py  # Dataset paths and runnable experiment settings
├── config/                    # Reference configurations (not loaded automatically)
├── start_training.py          # Training entry point
└── requirements.txt           # Python dependencies
```

## Setup

Use a Python environment with a compatible PyTorch installation. CUDA is recommended for training.

```bash
python -m pip install -r requirements.txt
```

The dataset is not included in this repository. The loader expects a directory containing PNG images and a CSV file whose first column contains the image identifier (without `.png`) and whose second column contains the permeability label.

## Run the included prototype

1. Set the absolute `image_dir` and `csv_file` paths in `scripts/run_patch_pred.py`.
2. Review the model and training settings in that script.
3. Run:

```bash
python start_training.py
```

Training logs, the best validation checkpoint, and test metrics are written to `training_output/`.

## Implemented workflow

- Three-channel image loading and normalization.
- Patch-level estimates of pore fraction and largest connected-component ratio.
- Learnable gated fusion of visual patch embeddings and morphological features.
- Multi-head Fourier feature processing and regression from a [CLS] token.
- A 60/20/20 random split using a configurable seed.
- Training with AdamW, validation-based learning-rate scheduling, checkpointing, and early stopping.
- Evaluation after inverting the `ln(1+k)` target transformation, including R², RMSE, and median absolute relative error (MedARE).

## Configuration and reproducibility note

The parameters used by the current entry point are defined in `scripts/run_patch_pred.py` and `models/fft_gated.py`; files under `config/` are examples and are not loaded automatically. The included script currently uses 224×224 input images and 56×56 patches. The spectral module in this prototype transforms the per-head feature dimension.

**This repository is a modular research prototype, not a verified one-to-one implementation of every configuration or supplementary experiment in the latest manuscript.** Reproduction of reported manuscript results requires matching the released code, model architecture, preprocessing, data partition, and evaluation protocol to the specific experiment.

## Data

The DeePore dataset and external sandstone benchmarks should be obtained from their original sources under their respective usage terms.
