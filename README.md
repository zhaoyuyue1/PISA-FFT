
# FFT Patch Predictor

This repository provides a modular implementation of a hybrid physics-informed vision model for predicting permeability from porous media microstructure images.

The pipeline utilizes gated fusion between image-derived patch embeddings and handcrafted statistical features, processed via a custom spectral transformer backbone. The architecture supports flexible attention modeling and optimized forward inference for patch-level regression.

---

##  Core Features

- Multi-headed **Spectral Attention** mechanism over frequency-domain features
- Integration of **physics-based patch statistics** with learnable embeddings
- Modular transformer block structure and gated fusion mechanism
- Full **training / validation / test** pipeline with logging and checkpointing
- Compatible with **CUDA acceleration** and AMP mixed-precision training
- Evaluation metrics include R², RMSE, MSE, and Median Absolute Relative Error (MedARE)

---

##  Structure Overview

```
fft_patch_predictor/
├── models/           # Core model architecture
├── data/             # Custom dataset loader and feature extractor
├── trainers/         # Training, evaluation, and logging utilities
├── scripts/          # Execution pipelines
├── config/           # Optional configuration templates
```

---

##  Dependencies

This codebase was tested with Python 3.10 and PyTorch 2.x.  
Minimal requirements include:

```bash
pip install torch torchvision opencv-python pandas scikit-learn
```

Some modules may require additional libraries (e.g. `albumentations`, `scikit-image`) depending on specific configurations.

---

##  Quick Start

Replace the image and CSV paths inside:
```python
scripts/run_patch_pred.py
```

Then run:
```bash
python start_training.py
```

All logs and results will be saved under `training_output/`.

---

##  Notes

- For full flexibility, adapt the model's transformer depth and embedding dimensions in `models/fft_gated.py`.
- Configuration files under `config/` are included for documentation purposes and do not affect training.
- Please ensure image sizes are divisible by the patch size.

---

##  License

This work is released under the MIT License. See `LICENSE` for details.
