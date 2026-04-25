# GN2 Jet Flavour Tagger

[![CI](https://github.com/lucanasini/CMEPDA_project_transformer_jet_tagging/actions/workflows/tests.yml/badge.svg)](https://github.com/lucanasini/CMEPDA_project_transformer_jet_tagging/actions/workflows/ci.yml)
[![Docs](https://github.com/lucanasini/CMEPDA_project_transformer_jet_tagging/actions/workflows/docs.yml/badge.svg)](https://lucanasini.github.io/CMEPDA_project_transformer_jet_tagging/)
![pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch)
![model](https://img.shields.io/badge/model-transformer-orange)
![cuda](https://img.shields.io/badge/CUDA-11.8-green)
![cuda](https://img.shields.io/badge/CUDA-12.1-green)
![license](https://img.shields.io/github/license/USER/REPO)
![CI](https://img.shields.io/github/actions/workflow/status/USER/REPO/workflow.yml)
![coverage](https://img.shields.io/codecov/c/github/USER/REPO)
![PyPI](https://img.shields.io/pypi/v/packagename)
![Python](https://img.shields.io/pypi/pyversions/packagename)

Implementation of the GN2 tagger described in:
> *"Transforming jet flavour tagging at ATLAS"*, Nature Communications (2026) 17:541

Simplified version with only the **jet flavor classification head**
(without the auxiliary objectives of track origin and vertex grouping).

---

## Project Structure

```
CMEPDA_project_transformer_jet_tagging/
├── main.py                         ← entry point
├── configs/
│   └── config.json                 ← all hyperparams and settings
├── dataset/
│   └── mc-flavtag-ttbar-*.h5       ← put here the .h5 file
├── src/trasformer_jet_tagging/
│   ├── __init__.py                 ← package init
│   ├── _version.py                 ← package version
│   ├── constants.py                ← feature names, class mapping
│   ├── dataset.py                  ← dataset and data loading
│   ├── model.py                    ← GN2 architecture (transformer) and D_b, D_c discriminant
│   ├── train.py                    ← GN2 loss, learning rate scheduler and training loop
│   ├── evaluate.py                 ← valutazione, plot
│   ├── plotting.py                 ← plotting functions (variables distributions and correlations, learning curves)
│   └── utils.py                    ← utility functions
├── tests/                          ← unit tests
│   ├── conftest.py                 ← package init
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_train.py
│   └── test_utils.py
└── outputs/
    ├── checkpoints/
    │   ├── runs/
    │   │   └── events.out.tfevents.xxxx
    │   └── best_model.pt
    ├── plots/
    │   └── *
    └── preprocess/
        ├── runs/
        └── norm_stats.json
```

---

## Requirements

Python 3.13+ with:

```
torch>=2.10.0
numpy>=2.4.0
h5py>=3.16.0
scikit-learn>=1.8.0
tensorboard>=2.20.0
matplotlib>=3.10.8
mplhep>=1.1.2
```

---

## Installation

### 1. Make a virtual environment (reccomended)

```bash
python3 -m venv venv
source venv/bin/activate          # Linux/macOS
# or: venv\Scripts\activate       # Windows
```

### 2. Install dependencies

**With GPU (CUDA 11.8):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install numpy h5py scikit-learn matplotlib
```

**With GPU (CUDA 12.1):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy h5py scikit-learn matplotlib
```

**Only CPU (slow):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy h5py scikit-learn matplotlib
```

### 3. Put the HDF5 file in `data/` folder

```bash
cp /path/to/your/file.h5 data/file_name.h5
```

The expected structure of the HDF5 file is:
```
/jets          — variables jet-level (structured array)
/tracks        — variables track-level (structured array)
/eventwise     — variables events
/truth_hadrons — info truth hadron
```

Update the file name in
`configs/config.json`.

---

## Execution

### Full training (default config)

```bash
python main.py --config configs/config.json
```

---

## Outputs
The trained model and training logs will be saved in the directory specified in `config["output"]["preprocess_dir"]` (default: `outputs/checkpoints/`):

```
outputs/checkpoints/
├── runs/
│   ├── events.out.tfevents.xxxx
│   └── …
└── best_model.pt
```

---

## Configuration

All hyperparameters and settings are in `configs/config.json`. You can edit it directly or pass a different config file with `--config`.

---

## Features and Target

### Input features
- Jet features (2):
  - `jet_pt`: transverse momentum of the jet
  - `jet_eta`: pseudorapidity of the jet
- Track features (19):
  - `qOverP`: charge over momentum
  - `deta`: difference in pseudorapidity between the track and the jet
  - `dphi`: difference in azimuthal angle between the track and the jet
  - `d0`: transverse impact parameter
  - `z0SinTheta`: longitudinal impact parameter times sin(theta)
  - `qOverPUncertainty`: uncertainty on qOverP
  - `thetaUncertainty`: uncertainty on theta
  - `phiUncertainty`: uncertainty on phi
  - `lifetimeSignedD0Significance`: signed transverse impact parameter significance
  - `lifetimeSignedZ0SinThetaSignificance`: signed longitudinal impact parameter significance
  - `numberOfPixelHits`: number of pixel hits
  - `numberOfSCTHits`: number of SCT hits
  - `numberOfInnermostPixelLayerHits`: number of hits in the innermost pixel layer
  - `numberOfNextToInnermostPixelLayerHits`: number of hits in the next-to-innermost pixel layer
  - `numberOfInnermostPixelLayerSharedHits`: number of shared hits in the innermost pixel layer
  - `numberOfInnermostPixelLayerSplitHits`: number of split hits in the innermost pixel layer
  - `numberOfPixelSharedHits`: number of shared hits in the pixel detector
  - `numberOfPixelSplitHits`: number of split hits in the pixel detector
  - `numberOfSCTSharedHits`: number of shared hits in the SCT detector

### Target
- `HadronConeExclTruthLabelID`: PDG ID → class
  - 5 → 0 (b-jet)
  - 4 → 1 (c-jet)
  - 0 → 2 (light-jet)
  - 15 → 3 (τ-jet)

---

## Architecture

```
For each jet (B = batch size):

  [jet_pt, jet_eta]               (2 feature)
  [40 tracks × 19 feature]        + boolean mask
        │
        ▼
  Concanate jet features to each tracks (B, 40, 21)
        │
        ▼
  Track Initialiser MLP: 21 → 256 → 256
        │
        ▼
  Transformer Encoder × 4 layer
  (8 heads, embed=256, ffn=512, preLayerNorm)
        │
        ▼
  Projection: 256 → 128
        │
        ▼
  Attention Pooling → (B, 128)   [jet representation]
        │
        ▼
  Classification Head: 128 → 128 → 64 → 32 → 4
        │
        ▼
  Softmax → [pb, pc, pu, pτ]
        │
        ▼
  D_b = log[ pb / (0.2 pc + 0.05 pτ + 0.75 pu) ]
  D_c = log[ pb / (0.3 pb + 0.01 pτ + 0.69 pu) ]
```