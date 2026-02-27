# Fashion-MNIST on a Large Canvas + CAM / Grad-CAM Localization

This repository contains a PyTorch experiment that turns **Fashion-MNIST classification** into a **weak localization** task by pasting the original 28×28 image onto a **128×128 blank canvas** at a random location.  
A lightweight CNN is trained to classify the item, then **Class Activation Maps (CAM)** and **Grad-CAM** are computed to highlight the predicted region and estimate a bounding box. The predicted box is compared to the ground-truth paste location using **IoU**.

---

## What this project does

### 1) Dataset: CanvasFashionMNIST
- Loads Fashion-MNIST images (1×28×28).
- Creates a new sample by:
  - initializing a **1×128×128** zero canvas
  - randomly choosing `(x0, y0)`
  - pasting the 28×28 image into the canvas at that location
  - returning:
    - `canvas` (normalized)
    - class label `y`
    - ground-truth bounding box `bbox = [x0, y0, x1, y1]`

### 2) Model: CNN_CAM
A CNN that outputs:
- `logits` for 10 Fashion-MNIST classes
- `feature map (fmap)` from the last convolution block  
It uses **Global Average Pooling (GAP)** before the final linear classifier so CAM can be computed from classifier weights.

### 3) Training & Evaluation
- Train/Validation split (default: 90% / 10%)
- Metrics:
  - Train/Val **loss** and **accuracy**
  - Test **ROC-AUC (One-vs-Rest)** for multi-class probabilities
  - Micro-average **ROC curve**
- Plots saved:
  - `loss_curve.png`
  - `acc_curve.png`
  - `roc_micro.png`

### 4) Localization with CAM
- Builds CAM heatmaps using the classifier weights of the predicted class.
- Upsamples CAM to **128×128**
- Thresholds CAM to derive a predicted bounding box
- Computes **IoU** against the true paste bounding box
- Saves overlay images:
  - `cam_overlay_0.png`, `cam_overlay_1.png`, ...

### 5) Localization with Grad-CAM (optional)
- Uses gradients of the predicted class score w.r.t. the last conv features
- Produces Grad-CAM overlays:
  - `gradcam_overlay_0.png`, ...

### 6) Metrics Summary File
Writes a plain text summary:
- device name
- epochs
- final accuracies
- test ROC-AUC
- CAM threshold
- mean IoU

Saved as: `out/metrics_summary.txt`

---

## Outputs

All outputs are saved under:
```text
./out/
```

## Generated files:
- loss_curve.png
- acc_curve.png
- roc_micro.png
- cam_overlay_*.png
- gradcam_overlay_*.png (if Grad-CAM section is executed)
- metrics_summary.txt

## Requirements
Core dependencies:
- Python 3.8+
- torch, torchvision
- numpy, matplotlib
- scikit-learn
Optional / notebook-related:
- IPython (only needed for display(Markdown(...)))

## Installation
### 1) Create environment (recommended)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```
### 2) Install dependencies
```bash
pip install torch torchvision numpy matplotlib scikit-learn ipython
```
If you want GPU support for PyTorch, install the correct CUDA build from the official PyTorch website.

## How to run

This script is written like a notebook-style file (it uses IPython.display).
You can run it in:

### Option A) Jupyter / Colab (recommended)
- Put the code in a notebook cell and run top-to-bottom.

### Option B) As a Python script
1. Save as main.py
2. (Optional) remove the last lines that use IPython.display if you don't want notebook output.
3. Run:
```bash
python main.py
```
Fashion-MNIST will be downloaded automatically into:
```bash
./data/
```

## Key hyperparameters (edit in code)
```bash
SEED = 42

BATCH_TRAIN = 64
BATCH_EVAL  = 256
EPOCHS = 25
LR = 1e-3

CANVAS = 128   # canvas size
PASTE  = 28    # pasted image size (Fashion-MNIST size)

N_VIS = 8      # number of visualization samples
CAM_THR = 0.4  # CAM threshold for bbox extraction

OUT_DIR = "./out"
```
## How CAM bounding boxes are computed
1. Compute CAM for predicted class:
- cam = ReLU( Σ_c (w_c * feature_map_c) )
2. Normalize CAM to [0, 1]
3. Threshold to a binary mask:
- mask = cam >= CAM_THR
4. Extract predicted bbox from mask min/max coordinates
5. Compute IoU with ground-truth paste bbox

## Notes & Tips
- IoU values here are computed only for the first N_VIS test samples used for visualization.
- CAM quality depends on:
  - training convergence
  - architecture capacity
  - paste randomness (task difficulty)
  - CAM threshold CAM_THR

- If CAM produces no pixels above threshold, bbox becomes None and IoU is 0.0.

## Repository structure (suggested)
```bash
.
├─ main.py
├─ data/               # auto-downloaded Fashion-MNIST (ignored in git)
├─ out/                # generated plots and overlays (ignored in git)
└─ README.md
```
Recommended .gitignore entries:
```bash
data/
out/
.venv/
__pycache__/
*.pt
*.pth
```
## LICENSE
```bash
MIT © 2026 Mohammad Azimi
```
## Author
- **Mohammad Azimi**

## Acknowledgments
- Fashion-MNIST dataset (Zalando Research)
- CAM / Grad-CAM methods widely used for CNN interpretability
