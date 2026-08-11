# MASTER BLUEPRINT: Multi-Scale Deformable Attention and Hierarchical Feature Fusion for Centralized Diabetic Retinopathy Grading
**Target Venue:** International Conference on Computer and Information Technology (ICCIT)
**Execution Mode:** Centralized Paradigm (Single-Node Training)
**Document Purpose:** This is the unified, consolidated blueprint merging the original masterplan and the LLM code-generation prompt into a single reference for both the scientific paper and the full production codebase.
**Role for LLM (e.g., Claude):** Act as an expert AI Researcher and Senior PyTorch Developer. Read this entire document end-to-end before generating any code. Every tensor dimension, file, and instruction below is binding — do not skip or summarize any part of the codebase section.

---

## TABLE OF CONTENTS
1. Scientific Methodology & Experiment Setup
2. Architectural Specification & Tensor Mathematics (The Novelty)
3. Training, Evaluation, and Ablation Protocols
4. Exact Directory Structure & File-by-File Implementation Instructions
5. Requirements & Environment
6. LLM Execution Instructions

---

## PART 1: SCIENTIFIC METHODOLOGY & EXPERIMENT SETUP

### 1.1 Dataset Distribution & Validation Strategy

* **Primary Dataset (Training, Validation, Internal Test): APTOS 2019 Blindness Detection (Kaggle)**
  * **Volume:** 3,662 images.
  * **Classes (0–4):** 0 (No DR), 1 (Mild), 2 (Moderate), 3 (Severe), 4 (Proliferative DR).
  * **On-disk location:** `datasets/APTOS_19/`
    * Image folders: `train_images/`, `val_images/`, `test_images/`
    * Label files: `train_1.csv`, `valid.csv`, `test.csv` — each with exactly two columns, `id_code, diagnosis`.
  * **Splits are pre-defined on disk — do NOT re-split at runtime.** The 70/15/15 Train/Val/Internal-Test partition already exists as three separate CSV files, each pointing at its own image folder. `test.csv` **does** carry ground-truth `diagnosis` labels (0–4), so the internal test split is fully labeled and can be scored directly — there is no need to call `sklearn.model_selection.train_test_split` at load time to manufacture these splits. `datamodule.py`'s job is simply to **read** `train_1.csv` / `valid.csv` / `test.csv` and instantiate one `DRDataset` per split, each pointed at its matching image folder (`train_images/`, `val_images/`, `test_images/` respectively).
  * Class weights (see Part 1.3) must still be computed **at runtime, from `train_1.csv`'s realized label distribution only** (via `sklearn.utils.class_weight.compute_class_weight('balanced', classes, y_train)`) — this is a class-imbalance computation over the training split's actual labels, not a splitting operation, and remains required.

* **External Validation Dataset (Generalizability / Domain-Shift Test): Messidor-2**
  * **Volume:** ~1,748 images.
  * **On-disk location:** `datasets/Messidor_2/`
    * Image folder: `images/` (filenames like `20051020_43808_0100_PP.png`)
    * Label file: `messidor_data.csv` — columns `id_code, diagnosis, adjudicated_dme, adjudicated_gradable`.
  * **Usage:** The model must **never** train on this dataset. It is used exclusively post-training for a separate confusion matrix and full metric suite, to demonstrate robustness against cross-clinic domain shift (different camera hardware, illumination, population).
  * **Gradability filter:** Before running external validation, filter out any row where `adjudicated_gradable == 0` — ungradable images should not be scored against the model as valid grading targets. `adjudicated_dme` is not used for the DR-grading task in this project and can simply be carried along/ignored when building the external `DRDataset`, unless a future extension adds DME staging as a task.

### 1.2 Automated Data Preprocessing (No Manual Annotation)
All preprocessing must be strictly automated using `cv2` and `numpy`, executed inside the PyTorch `Dataset.__getitem__` method (no offline preprocessing scripts, no manual cropping):

1. **Retinal Boundary Masking (Fundus Cropping):**
   * Read the image with `cv2` and convert to grayscale.
   * Apply `cv2.threshold` to binarize and isolate the circular fundus region from the black peripheral background.
   * Use `cv2.boundingRect` on the largest contour to obtain `(x, y, w, h)`.
   * Crop: `img = img[y:y+h, x:x+w]`. This removes non-informative black padding surrounding the retinal image.
2. **Spatial Standardization:**
   * Resize the cropped region to exactly **384 × 384 pixels** using bi-cubic interpolation. This resolution (vs. the more common 224×224) better preserves microaneurysms and other small lesion structures critical for early-stage DR grading.
3. **Luminosity Standardization (Ben Graham's Method):**
   * `img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), 10), -4, 128)`
   * This neutralizes illumination and color-balance differences introduced by different fundus cameras across clinics, a major source of domain shift in retinal imaging datasets.
4. **Color Space & Tensor Conversion:**
   * Convert the enhanced image back to RGB, apply the Albumentations transform pipeline (see Section 4, `transforms.py`), and return the final tensor along with its integer label.

### 1.3 Handling Heavy Class Imbalance
APTOS-2019 is heavily skewed toward class 0 (No DR). The training pipeline must implement a three-tier mitigation strategy:
1. **Dynamic Class Weights:** Compute inverse class frequencies from the training split only — i.e. from `train_1.csv`'s realized `diagnosis` labels (via `sklearn.utils.class_weight.compute_class_weight('balanced', classes, y_train)`) — and pass the resulting weight tensor into the loss function.
2. **Focal Loss:** Combine Focal Loss (`γ = 2.0`) with weighted Cross-Entropy to force the model to focus gradient signal on hard, minority-class examples (Severe / Proliferative DR) rather than being dominated by the abundant No-DR class.
3. **Heavy Augmentation (Albumentations), training set only:** `RandomRotate90`, `HorizontalFlip`, `VerticalFlip`, `ColorJitter`, `GridDistortion`, `ShiftScaleRotate`. Validation/Test/External sets receive **no** augmentation beyond resize and normalization, to keep evaluation faithful to real deployment conditions.

---

## PART 2: ARCHITECTURAL SPECIFICATION & TENSOR MATHEMATICS (THE NOVELTY)

The core idea is to modify a hierarchical vision transformer backbone so that it explicitly preserves and re-injects fine-grained lesion features (microaneurysms, small hemorrhages) that are normally diluted by the time the network reaches its deepest, most semantic layers.

### 2.1 Core Backbone
* **Model:** `timm.create_model('swinv2_base_window12to16_192to256', pretrained=True, features_only=True)`
* **Input Tensor:** `(Batch, 3, 384, 384)`
* **⚠️ Critical Implementation Note — Window/Position-Embedding Resizing:** The backbone is pretrained at `192×192`/`256×256`, but this pipeline feeds it `384×384` tensors. Swin Transformers use windowed attention with relative positional encodings tied to a fixed window size, so simply resizing the input (unlike a standard CNN) can throw a shape-mismatch error unless the backbone is told to adapt. Ensure `timm.create_model(...)` includes `dynamic_img_size=True` (or explicitly passes `img_size=384`) so Swin-V2 automatically interpolates its relative positional encodings and window sizes to accommodate the larger input tensor. This must be set at model construction time in `dr_model.py`.
* **Backbone Outputs — 4 Hierarchical Stages:**

| Stage | Output Shape `(B, C, H, W)` | Semantic Role |
|---|---|---|
| Stage 1 | `(B, 128, 96, 96)` | Low-level features (edges, vessel boundaries) |
| Stage 2 | `(B, 256, 48, 48)` | Fine-grained detail (microaneurysms, small exudates) |
| Stage 3 | `(B, 512, 24, 24)` | Mid-level semantic features |
| Stage 4 | `(B, 1024, 12, 12)` | High-level deep semantics (Proliferative DR signs: neovascularization, large hemorrhages) |

### 2.2 Novelty Module 1 — MSDA (Multi-Scale Deformable Attention)
* **Location:** Applied on the deep semantic outputs — **Stage 3** `(B, 512, 24, 24)` and **Stage 4** `(B, 1024, 12, 12)`.
* **Mechanism:**
  * Implemented via `torchvision.ops.DeformConv2d`.
  * Instead of sampling a fixed square convolutional grid, MSDA learns per-location 2D spatial **offsets**, allowing the sampling grid to bend and follow the irregular, non-rectangular morphology of retinal hemorrhages and exudates rather than being constrained to axis-aligned receptive fields.
  * A small offset-prediction convolution generates the offset field, which is then consumed by `DeformConv2d` to produce the deformable-attended feature map at the same spatial resolution as its input stage.
* **⚠️ Critical Implementation Note — Deformable Conv v2 Modulation Mask:** `torchvision.ops.DeformConv2d` expects both spatial offsets **and** a modulation mask (the Deformable Conv v2 standard) — predicting offsets alone will crash at runtime. The offset-prediction convolution inside `MSDABlock` must output `3 * kernel_size**2 * out_channels` channels total, packing both the 2D spatial offsets `(x, y)` and the 1D modulation mask together. Use `torch.split` to separate this combined output into `offsets` and `mask` (applying `torch.sigmoid` to the mask portion) before passing both, along with the input feature map, into `torchvision.ops.DeformConv2d`.

### 2.3 Novelty Module 2 — HFF (Hierarchical Feature Fusion)
* **Goal:** Bridge shallow, high-resolution Stage 2 features `(B, 256, 48, 48)` directly into deep Stage 4 features `(B, 1024, 12, 12)`, so fine lesion detail is not lost by the final classification stage.
* **Mechanism (exact tensor math):**
  1. Pass Stage 2 through a **Projection Block**: `nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=4, stride=4)`.
  2. This stride-4 convolution downsamples spatially from `48×48` to exactly `12×12` while simultaneously projecting channels from `256` to `1024` — matching Stage 4 exactly.
  3. Fuse via element-wise addition: `fused_features = stage4_features + projected_stage2`, implemented as `torch.add()`.
  * This is a residual-style shortcut that lets fine-grained edge/microaneurysm information skip directly into the deepest representation used for classification.

### 2.4 Classification Head
* **Input:** Fused features `(B, 1024, 12, 12)`.
* Apply `nn.AdaptiveAvgPool2d((1, 1))` → `(B, 1024, 1, 1)`.
* Flatten → `(B, 1024)`.
* `nn.Dropout(p=0.5)` (regularization against the relatively small APTOS training set).
* `nn.Linear(1024, 5)` → 5-class logits (DR grades 0–4).

---

## PART 3: TRAINING, EVALUATION, AND ABLATION PROTOCOLS

### 3.1 Training Engine
* **Hardware Target:** 1× NVIDIA GPU (e.g., A100/V100/H100-class).
* **Optimizer:** `AdamW` (`lr=1e-4`, `weight_decay=0.05`).
* **Scheduler:** `CosineAnnealingLR` with a 5-epoch linear warmup.
* **Epochs:** 50 max, with **Early Stopping** (patience = 7, monitored on validation loss and/or validation QWK).
* **Batch Size:** 32 (configurable).
* **Metrics Logged Every Epoch:** Loss, Accuracy, Macro F1-Score, Precision, Recall, and **Quadratic Weighted Kappa (QWK)** — QWK is the primary clinical-relevance metric for ordinal DR grading, since it penalizes distant misclassifications (e.g., predicting class 0 when the true label is class 4) far more heavily than adjacent-grade errors.

### 3.2 Explainable AI (XAI)
* **Library:** `pytorch-grad-cam`.
* **Method:** `EigenCAM` or `GradCAM`, specifically targeted at the final deformable-convolution layer or the final Swin-V2 stage (Vision-Transformer-compatible CAM configuration).
* **Action:** Generate attention heatmaps overlaid on the cropped, preprocessed fundus image (showing where the model attended — e.g., exudates, hemorrhage regions) and persist as `.png` files for qualitative analysis in the paper.

### 3.3 The Ablation Matrix (Command-Line Toggles)
The codebase must support four distinct experiment configurations via `argparse`, to isolate and quantify the contribution of each novelty module for the conference paper's ablation table:

| Experiment | Configuration |
|---|---|
| Exp 1 — Baseline | Swin-V2 backbone + GAP + Linear head only (no MSDA, no HFF) |
| Exp 2 — Baseline + MSDA | Adds deformable attention on Stage 3/4 |
| Exp 3 — Baseline + HFF | Adds Stage 2 → Stage 4 hierarchical fusion |
| Exp 4 — Proposed (SOTA) | Baseline + MSDA + HFF (full proposed architecture) |

These map to the `--ablation` CLI flag: `baseline`, `msda_only`, `hff_only`, `proposed`.

---

## PART 4: EXACT DIRECTORY STRUCTURE & FILE-BY-FILE IMPLEMENTATION INSTRUCTIONS

LLM (Claude/GPT) instructions: Generate the following structure strictly as `.py` files (no `.ipynb`). Ensure type hinting, modularity, and comprehensive logging throughout. The `datasets/` tree below already exists on disk exactly as shown — do not rename these folders/files or assume a different layout when writing `dataset.py` / `datamodule.py` / `config.yaml`.

```text
dr_classification/
│
├── configs/
│   └── config.yaml          # All hyperparameters, dataset paths, and ablation toggles
├── datasets/
│   ├── APTOS_19/
│   │   ├── train_images/    # training image files
│   │   ├── val_images/      # validation image files
│   │   ├── test_images/     # internal-test image files
│   │   ├── train_1.csv      # columns: id_code, diagnosis
│   │   ├── valid.csv        # columns: id_code, diagnosis
│   │   └── test.csv         # columns: id_code, diagnosis (labeled internal test set)
│   └── Messidor_2/
│       ├── images/          # external-validation image files
│       └── messidor_data.csv  # columns: id_code, diagnosis, adjudicated_dme, adjudicated_gradable
├── src/
│   ├── data/
│   │   ├── transforms.py    # Albumentations pipelines (train and val/test)
│   │   ├── dataset.py       # PyTorch Dataset overriding __getitem__ with cv2 automated preprocessing
│   │   └── datamodule.py    # Reads pre-split CSVs, builds dataloaders, computes class weights
│   ├── models/
│   │   ├── components.py    # MSDABlock (DeformConv2d) and HFFBlock (stride-4 projection) modules
│   │   └── dr_model.py      # ICCIT_DR_Net: glues timm Swin-V2 backbone, MSDA, HFF, and classifier head
│   ├── training/
│   │   ├── loss.py          # Combined Focal Loss + weighted CrossEntropyLoss
│   │   └── trainer.py       # Train/val loops, early stopping, checkpoint saving
│   ├── evaluation/
│   │   ├── metrics.py       # Acc, Macro F1, Precision, Recall, QWK
│   │   ├── xai.py           # EigenCAM/GradCAM heatmap generation
│   │   └── visualizer.py    # Training curves, confusion matrix, ROC curve plots
│   └── utils/
│       └── logger.py        # Console + file logging setup
│
├── main.py                  # CLI entry point (argparse: --mode, --ablation)
└── requirements.txt
```

### 📁 `configs/config.yaml`
* Hyperparameters: `img_size: 384`, `batch_size: 32`, `lr: 1e-4`, `weight_decay: 0.05`, `epochs: 50`.
* Dataset paths (must match the on-disk layout exactly):
  * `datasets/APTOS_19/train_images` + `datasets/APTOS_19/train_1.csv`
  * `datasets/APTOS_19/val_images` + `datasets/APTOS_19/valid.csv`
  * `datasets/APTOS_19/test_images` + `datasets/APTOS_19/test.csv`
  * `datasets/Messidor_2/images` + `datasets/Messidor_2/messidor_data.csv`
* Ablation flags: `use_msda: True`, `use_hff: True`.

### 📁 `src/data/`

**`transforms.py`**
* Implements Albumentations pipelines.
* **Train transform:** `Resize(384, 384)` → `RandomRotate90` → `HorizontalFlip` → `VerticalFlip` → `ColorJitter` → `GridDistortion` → `ShiftScaleRotate` → `Normalize(mean, std)` → `ToTensorV2`.
* **Val/Test/External transform:** `Resize(384, 384)` → `Normalize` → `ToTensorV2` (no augmentation).

**`dataset.py`**
* Class `DRDataset(Dataset)`: takes an image directory, a labels DataFrame (with `id_code`/`diagnosis` columns — the same schema for both APTOS-2019 and Messidor-2), and a transform pipeline. Image filename resolution should join the folder path with the `id_code` column (adding the appropriate extension, e.g. `.png`, if not already present in `id_code`).
* **`__getitem__` — Automated OpenCV Preprocessing Logic (must be implemented exactly):**
  1. Read image with `cv2`; convert to grayscale.
  2. Apply `cv2.threshold` to isolate the fundus contour.
  3. Use `cv2.boundingRect` to get `(x, y, w, h)` and crop: `img = img[y:y+h, x:x+w]` (removes black padding).
  4. Resize to `384×384`.
  5. Apply Ben Graham's enhancement: `img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), 10), -4, 128)`.
  6. Convert back to RGB, apply the Albumentations transform, and return `(tensor, label)`.

**`datamodule.py`**
* Reads the three **already-split** APTOS-2019 CSVs directly — `train_1.csv`, `valid.csv`, `test.csv` — each paired with its own image folder (`train_images/`, `val_images/`, `test_images/`). **No runtime stratified splitting via `train_test_split` is performed**; the 70/15/15 partition already exists on disk and must simply be loaded as-is.
* Computes class weights via `sklearn.utils.class_weight.compute_class_weight('balanced', classes=..., y=y_train)`, using `train_1.csv`'s realized `diagnosis` distribution, and returns the weight tensor for use in the loss function.
* Reads `datasets/Messidor_2/messidor_data.csv` for the external set, **filters to rows where `adjudicated_gradable == 1`** before building the external `DRDataset` (images are read from `datasets/Messidor_2/images/`).
* Provides factory functions to build PyTorch `DataLoader`s for train / val / internal-test / external (Messidor-2) sets — four loaders total.
* **⚠️ Critical Implementation Note — Device Placement of Class Weights:** `compute_class_weight` returns a NumPy array on CPU. A very common failure mode is leaving this weight tensor on CPU while model outputs and targets live on GPU, which crashes `CrossEntropyLoss` with a device-mismatch error. The array returned here must be explicitly converted to `torch.FloatTensor` and moved with `.to(device)` — this conversion should happen either at the end of `datamodule.py` (if `device` is known there) or immediately upon receipt inside `loss.py`/`trainer.py`, but it must happen before the weights are ever passed into `CrossEntropyLoss` or the combined Focal+CE loss.

### 📁 `src/models/`

**`components.py`**
* `MSDABlock(nn.Module)`: wraps `torchvision.ops.DeformConv2d`; internally predicts the 2D offset field **and** the DCNv2 modulation mask via a single small regular convolution (output channels = `3 * kernel_size**2 * out_channels`), splits the result with `torch.split` into `offsets` and `mask` (mask passed through `sigmoid`), then applies deformable convolution using both to the input feature map (Stage 3 or Stage 4). Omitting the modulation mask and passing only offsets will cause a runtime crash — both must be provided.
* `HFFBlock(nn.Module)`: contains the `nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=4, stride=4)` projection described in Part 2.3, followed by the residual `torch.add()` fusion step against Stage 4 features.

**`dr_model.py`**
* Class `ICCIT_DR_Net(nn.Module)`:
  * Initializes the backbone: `timm.create_model('swinv2_base_window12to16_192to256', pretrained=True, features_only=True, dynamic_img_size=True)`. The `dynamic_img_size=True` (or explicit `img_size=384`) argument is mandatory here — without it, feeding `384×384` tensors into a backbone pretrained at `192×192`/`256×256` will raise a positional-embedding/window-size shape mismatch at the first forward pass.
  * In `forward()`: extracts `stage1, stage2, stage3, stage4` from the backbone.
  * If `use_msda` is enabled (per ablation config): applies `MSDABlock` to `stage3` and `stage4`.
  * If `use_hff` is enabled: applies `HFFBlock` to fuse `stage2` into `stage4`.
  * Passes the resulting fused (or baseline) features through `AdaptiveAvgPool2d` → `Dropout` → `Linear` classification head.
  * Ablation toggles must be constructor arguments (`use_msda: bool`, `use_hff: bool`) so `main.py` can instantiate any of the 4 experiment configurations from Part 3.3 without code duplication.

### 📁 `src/training/`

**`loss.py`**
* Custom loss class combining `FocalLoss(gamma=2.0)` with `CrossEntropyLoss(weight=class_weights)`, as motivated in Part 1.3.
* **Mandatory device check:** before `class_weights` is passed into `CrossEntropyLoss`, confirm it is a `torch.FloatTensor` and explicitly call `.to(device)` on it (matching the device of model outputs/targets). Do this inside the loss class `__init__` (or immediately before instantiation in `trainer.py`) — never leave it as a raw CPU NumPy array from `sklearn`.

**`trainer.py`**
* Implements `train_epoch()` and `val_epoch()`.
* Manages `AdamW` optimizer and `CosineAnnealingLR` scheduler with 5-epoch linear warmup.
* Implements early stopping (patience = 7, monitored on validation loss/QWK).
* Logs metrics every epoch (via `utils/logger.py`) and saves the best checkpoint as `best_iccit_model.pth`.

### 📁 `src/evaluation/`

**`metrics.py`**
* Wraps `sklearn.metrics`: `accuracy_score`, `f1_score(average='macro')`, `precision_score`, `recall_score`, and `cohen_kappa_score(weights='quadratic')` (QWK).

**`visualizer.py`** — Matplotlib functions, each saving to disk:
1. `plot_training_curves(train_loss, val_loss, train_acc, val_acc)` → `training_curves.png`.
2. `plot_confusion_matrix(y_true, y_pred)` → `confusion_matrix.png`.
3. `plot_roc_curve(y_true, y_pred_probs)` → `roc_multiclass.png`.

**`xai.py`**
* Initializes `EigenCAM` or `GradCAM` targeting the final deformable-conv layer or final Swin-V2 stage.
* `generate_heatmap(image_tensor, model)`: overlays the attention heatmap on the original preprocessed fundus image and saves as `xai_heatmap.png`.
* **⚠️ Critical Implementation Note — Exact Target Layer Path for timm Swin-V2:** `pytorch-grad-cam` requires an exact module reference for the target layer. This is trivial for a standard ResNet (`model.layer4[-1]`) but timm's Swin-V2 uses a deeply nested, version-specific naming convention. When initializing `EigenCAM`/`GradCAM`, point the target layer specifically at the final Swin stage block — e.g., `model.backbone.layers[-1].blocks[-1].norm2` — or, when the proposed (MSDA+HFF) ablation is active, at the output of the final `MSDABlock` instead, since that is the last spatial feature map feeding the classifier. Because timm's exact attribute names can vary by version, the implementation must first print/inspect `model.named_modules()` (or equivalent) to verify the correct nested attribute path before hardcoding it, rather than assuming the path blindly.

### 📁 `src/utils/logger.py`
* Standard Python `logging` setup, writing to both console and a `training_logs.log` file.

### 📄 `main.py` — Entry Point
* Uses `argparse` with:
  * `--mode`: choices `[train, test, external_validation, xai]`.
  * `--ablation`: choices `[baseline, msda_only, hff_only, proposed]` — dynamically toggles `use_msda`/`use_hff` when constructing `ICCIT_DR_Net`.
* **Execution logic:**
  * `--mode train`: instantiate dataloaders (via `datamodule.py`, reading `train_1.csv`/`valid.csv`), model (`dr_model.py`, ablation-configured), loss (`loss.py`), and trainer (`trainer.py`); run the training loop.
  * `--mode test`: load `best_iccit_model.pth`, run inference on the APTOS internal test split (`test.csv` + `test_images/`, already labeled), report metrics via `metrics.py` and plots via `visualizer.py`.
  * `--mode external_validation`: load `best_iccit_model.pth`, load the Messidor-2 dataset (`messidor_data.csv` + `images/`, filtered to `adjudicated_gradable == 1`, no training), run inference, and report the full metric suite plus a dedicated confusion matrix to demonstrate cross-domain generalizability.
  * `--mode xai`: load the trained model and generate Grad-CAM/EigenCAM heatmaps for a sample of test images.

---

## PART 5: REQUIREMENTS & ENVIRONMENT

**`requirements.txt`:**
```
torch
torchvision
timm
albumentations
opencv-python
scikit-learn
matplotlib
pandas
pyyaml
grad-cam
```

---

## PART 6: LLM EXECUTION INSTRUCTIONS

**Final instruction for code generation:** Based on this complete blueprint, output the full Python code for every file listed in the directory structure (Part 4). Do not skip any file. Before finalizing, explicitly verify each of these four zero-tolerance runtime constraints:

1. **Swin-V2 resizing:** `timm.create_model(...)` includes `dynamic_img_size=True` (or `img_size=384`) so positional embeddings/windows interpolate correctly for the `384×384` input.
2. **DCNv2 modulation mask:** `MSDABlock`'s offset-prediction conv outputs `3 * kernel_size**2 * out_channels`, split via `torch.split` into offsets + sigmoid-activated mask, both passed into `DeformConv2d`.
3. **Class-weight device placement:** the `sklearn`-computed class-weight array is converted to `torch.FloatTensor` and moved with `.to(device)` before reaching `CrossEntropyLoss`.
4. **XAI target layer path:** the exact nested attribute path for the target Swin-V2 layer (or final `MSDABlock` output) is verified against `model.named_modules()` rather than assumed.

In addition, ensure that:
* The HFF dimension-matching math (`Conv2d(256, 1024, kernel_size=4, stride=4)` mapping `48×48 → 12×12`) is implemented exactly as specified.
* The MSDA offset-prediction and `DeformConv2d` usage is fully implemented, not stubbed.
* The automated OpenCV preprocessing logic inside `Dataset.__getitem__` (fundus cropping, resize, Ben Graham enhancement) is completely written out, not simplified or omitted.
* `datamodule.py` reads the pre-split CSVs (`train_1.csv`, `valid.csv`, `test.csv`, `messidor_data.csv`) exactly as they exist on disk — it must not attempt to re-split APTOS-2019 with `train_test_split`, since the splits are already provided.
* All four ablation configurations (baseline, msda_only, hff_only, proposed) are runnable end-to-end from `main.py` without modifying source files.
* No `.ipynb` notebooks are produced — only modular `.py` files as structured above.
* Token Limit Management: Because this is a comprehensive codebase, you may hit your maximum output token limit before finishing all the files. If you are about to hit your limit, finish the current .py file, then STOP and ask me if you should continue. Do NOT output truncated code or combine files to save space. Write the codebase in chunks if necessary (e.g., Outputting configs, data, and models first, waiting for my prompt, then outputting training, evaluation, and main.py).
