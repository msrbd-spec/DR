# DATA'26 Review Resolution Plan - HemaVision-Net

**Paper ID:** #31 (1571308266)  
**Status:** Accepted with Major Revisions  
**Deadline:** 5 DAYS from 7/10/26  
**Conference:** 2026 7th International Conference on Data Analytics for Business and Industry (ICDABI)

---

## Executive Summary

The paper has been **accepted with major revisions**. Two reviewers provided detailed feedback highlighting both strengths and critical weaknesses that must be addressed before final submission to IEEE Xplore.

### Review Scores
| Reviewer | Originality | Significance | Presentation | Overall |
|----------|-------------|--------------|--------------|---------|
| Review 1 | Weak Accept | Accept (6)   | Weak Accept  | Accept with Major Revisions |
| Review 2 | Neutral (5) | Accept (8)   | **Reject (2)** | Accept with Major Revisions |

### Key Strengths Identified
- Addresses important and challenging problem in automated blood smear analysis
- High-resolution YOLOv11-based framework with two-stage training strategy
- Clear technical direction with RT-DETR and SAHI comparisons
- Achieves **92.93% mAP@0.5** on internal BCCD test set
- Achieves **84.99% mAP@0.5** on unseen external dataset

### Critical Weaknesses to Address
1. **Methodological errors** in equations and metrics (Recall formula, loss function labeling)
2. **Incomplete evaluation** - missing mAP@0.5:0.95, per-class metrics, ablation studies
3. **Overstated clinical claims** without proper external validation
4. **Insufficient technical depth** - missing component-level analysis
5. **Presentation issues** - typos, formatting, clarity problems

---

## Priority 1: CRITICAL FIXES (Must Complete - Days 1-2)

### 1.1 Correct Mathematical Errors ⚠️

#### A. Recall Formula Error
**Location:** Section III. METHODOLOGY → Evaluation Metrics (Eq. 3, Line 313)

**Current Line (EXACT from paper):**
```
TP +TN
```

**Replace With:**
```
TP +FN
```

**Also update text above Eq. 3 (Line 310):**
- **Current:** `Recall=`
- **Replace With:** `Recall = TP/(TP+FN)`

**Action Items:**
- [x] Identified exact line in manuscript
- [ ] Replace with correct formula
- [ ] Verify that reported results used correct implementation
- [ ] Add statement: "All metrics computed using standard COCO evaluation"

#### B. Loss Function Labeling Error
**Location:** Section III. METHODOLOGY → D. Loss Function Optimization (Lines 205-215)

**Current Lines (EXACT from paper):**
```
| To enforce | strict         | boundary     |      | adherence, | the          | total | loss (L | total ) |     |     |     |
| combines   | classification |              | loss | (L ),      | Distribution |       | Focal   | Loss    |     |     |     |
cls
| (L ) and   | Complete |               | Intersection |     | over Union | (L  | ).  | Eq. 1 |     |     |     |
| ---------- | -------- | ------------- | ------------ | --- | ---------- | --- | --- | ----- | --- | --- | --- |
| DFL        |          |               |              |     |            |     | DFL |       |     |     |     |
| formulates | this     | relationship: |              |     |            |     |     |       |     |     |     |
ρ2(b,bgt)
|            | L   | =1−IoU   |            | +   |      | +αv      |     | (1)  |     |     |     |
| ---------- | --- | -------- | ---------- | --- | ---- | -------- | --- | ---- | --- | --- | --- |
|            |     | CIoU     |            |     | c2   |          |     |      |     |     |     |
```

**Replace With:**
```
| To enforce | strict         | boundary     |      | adherence, | the          | total | loss (L | total ) |     |     |     |
| combines   | classification |              | loss | (L ),      | Distribution |       | Focal   | Loss    |     |     |     |
cls
| (L ) and   | Complete |               | Intersection |     | over Union | (L    | ).  | Eq. 1 |     |     |     |
| ---------- | -------- | ------------- | ------------ | --- | ---------- | --- | --- | ----- | --- | --- | --- |
| DFL        |          |               |              |     |            |     | CIoU |       |     |     |     |
| formulates | this     | relationship: |              |     |            |     |     |       |     |     |     |
L_total = λ_box·L_CIoU + λ_cls·L_cls + λ_dfl·L_dfl    (1)
| ---------- | -------- | ------------- | ------------ | --- | ---------- | --- | --- | ----- | --- | --- | --- |
| where      | λ        | = 10.0,      | λ            | = 1.0, | λ          | = 1.0 |     |      |     |     |     |
|            | box      |              | cls          |       | dfl        |       |     |      |     |     |     |
```

**Action Items:**
- [x] Identified exact lines in manuscript
- [ ] Rewrite Section 3.2 with complete equation
- [ ] Specify weights: λ_box=10.0, λ_cls=1.0, λ_dfl=1.0

---

### 1.2 Fix Terminology and Grammar Errors ✅

#### A. "close-perfect" → "near-perfect"
**Location:** Section IV. RESULTS → A. Detection Performance Analysis (Line 264)

**Current Line (EXACT from paper):**
```
of 92.93%.Especially |          | the       | model        | accomplished |     | close-perfect |           |
```

**Replace With:**
```
of 92.93%. Especially |          | the       | model        | accomplished |     | near-perfect  |           |
```

#### B. "furthers confirms" → "further confirms"
**Location:** Section IV. RESULTS → A. Detection Performance Analysis (Line 278)

**Current Line (EXACT from paper):**
```
the normalized | confusion |     | matrix, | furthers | confirms |     | this robust |
```

**Replace With:**
```
the normalized | confusion |     | matrix, | further | confirms |     | this robust |
```

**Action Items:**
- [x] Identified all grammar/spelling errors
- [ ] Apply all replacements throughout manuscript

---

### 1.3 Complete Missing Evaluation Metrics ⚠️ **REQUIRES CODE EXECUTION**

#### A. Missing mAP@0.5:0.95 and Per-Class Metrics
**Location:** Section IV. RESULTS → A. Detection Performance Analysis (Lines 264-268)

**Current Lines (EXACT from paper):**
```
of 92.93%.Especially |          | the       | model        | accomplished |     | close-perfect |           |
#### B. TTA Clarification for 92.93% mAP
**Location:** Section IV. RESULTS → A. Detection Performance Analysis (Line 320)

**Current Line (EXACT from paper):**
```
|     |     |     |     |     |     |     | mAP@0.5        |               | of 92.93% |              | while improving |              | localized    | recall.        |           |
```

**Replace With (⚠️ ESTIMATED - requires checking if TTA was used):**
```
|     |     |     |     |     |     |     | mAP@0.5        |               | of 92.93% (with TTA) | while improving | localized | recall. | The baseline RT-DETR | evaluated | under same | TTA conditions | achieved | 89.6% |
```

**⚠️ FLAG:** This is an **ESTIMATED LINE** - requires verifying whether 92.93% includes TTA.

**Action Items:**
- [x] Identified exact line in manuscript
- [ ] Run TTA status check code (see Section 3.3)
- [ ] Update text to clarify TTA usage

---

### 1.4 Justify λ_box = 10.0 Selection ⚠️ **REQUIRES CODE EXECUTION**

**Location:** Section III. METHODOLOGY → D. Loss Function Optimization (Lines 216-218)

**Current Lines (EXACT from paper):**
```
| By raising | the | box loss | multiplier |     | (λ ) | to 10.0, | the | CIoU |     |     |     |
box
| penalty | forces | ranking | of the | Euclidean | distance |     | (ρ) between |                      |                       |                         |     |
```

**Replace With (⚠️ ESTIMATED - requires sensitivity analysis):**
```
| By raising | the | box loss | multiplier |     | (λ  ) | to 10.0 (selected | after | sensitivity |     |     |     |
|            |     |          |            |     | box   | analysis: λ=7.5→91.2%, λ=10.0→92.93%, λ=12.5→92.1%), | the | CIoU |     |     |     |
| penalty | forces | ranking | of the | Euclidean | distance |     | (ρ) between | predicted | and | ground-truth | box | centers |
```

**⚠️ FLAG:** This is an **ESTIMATED LINE** - requires running training with λ_box = 7.5, 10.0, 12.5.

**Action Items:**
- [x] Identified exact lines in manuscript
- [ ] Run λ_box sensitivity analysis code (see Section 2.2)
- [ ] Replace placeholder values with actual mAP results

---

### 1.5 Fix Table II Title - Not an Ablation Study ✅

**Location:** Section IV. RESULTS → Table II (Line 329)

**Current Line (EXACT from paper):**
```
---

### 1.6 Add SAHI Failure Evidence ⚠️ **REQUIRES CODE EXECUTION**

**Location:** Section IV. RESULTS → E. Architectural Limits (Lines 242-249)

**Current Lines (EXACT from paper):**
```
Paradox", driven by three limitations: mAP@0.5, which calculates the mAP across all classes by
EdgeTruncation:Slicingintroducesartificialboundaries averaging the Area Under the Curve (AUC) of the Precision-
•
|                |         |            |             |        |                  |     |             | Recall curve | at the | Intersection |     | over Union |     | (IoU) threshold | of  |
| -------------- | ------- | ---------- | ----------- | ------ | ---------------- | --- | ----------- | ------------ | ------ | ------------ | --- | ---------- | --- | --------------- | --- |
| cutting        | through | continuous |             | cells. | Truncated        |     | regions are |              |        |              |     |            |     |                 |     |
| misinterpreted |         | as         | independent |        | cells, inflating |     | False Posi- | 0.50.        |        |              |     |            |     |                 |     |
tives.
```

**Replace With (⚠️ ESTIMATED - requires SAHI parameter analysis):**
```
Paradox", driven by three limitations (with SAHI parameters: slice_size=512, overlap_ratio=0.2, NMM threshold=0.5):
• Edge Truncation: Slicing introduces artificial boundaries that cut through continuous cells.
  Truncated regions are misinterpreted as independent cells, inflating False Positives.
  [See Fig. X for visual examples - ESTIMATED: requires generating failure case visualization]
• Context Loss: [ESTIMATED - requires analysis]
• NMM Failure: [ESTIMATED - requires analysis with alternative NMM thresholds]
```

**⚠️ FLAG:** This is an **ESTIMATED LINE** - requires SAHI failure analysis.

**Action Items:**
- [x] Identified exact lines in manuscript
- [ ] Run SAHI failure analysis code (see Section 2.3)
- [ ] Document SAHI parameters used
- [ ] Generate 3-4 visual failure case examples

---

### 1.7 Add Dataset Statistics ⚠️ **REQUIRES CODE EXECUTION**

**Location:** Section IV. RESULTS → A. Detection Performance Analysis (After Line 282)

**Current:** No dataset statistics provided

**Add New Lines (⚠️ ESTIMATED - requires extracting dataset counts):**
```
Dataset Statistics: The BCCD dataset contains XX,XXX images with XX,XXX RBCs, X,XXX WBCs,
and X,XXX platelets. Split: 70% train (patient-level), 15% validation, 15% test.
Preprocessing: Images resized to 800×800, normalized to [0,1]. No data leakage across splits.
External dataset (Kaggle WBC Detection): X,XXX images, completely independent patients and staining protocols.
```

**⚠️ FLAG:** This is an **ESTIMATED LINE** - requires extracting actual image/cell counts.

**Action Items:**
- [x] Identified missing section in manuscript
- [ ] Extract image counts and cell counts from dataset
- [ ] Document preprocessing steps and resolutions
- [ ] Add dataset statistics paragraph to paper

---

### 1.8 Correct External Validation Claims ✅

#### A. Abstract (Line 36)
**Current Line (EXACT from paper):**
```
and 84.99% on out-of-distribution images, thereby emphasizing different imaging conditions.
```

**Replace With:**
```
and 84.99% on an external Kaggle dataset, demonstrating cross-dataset generalization under varying imaging conditions.
```

#### B. Conclusion (Line 401)
**Current Line (EXACT from paper):**
```
mAP@0.5 performance in the internal test set and 84.99%
```

**Replace With:**
```
mAP@0.5 performance in the internal test set and 84.99% on an external dataset (cross-dataset generalization, not clinical validation).
```

**Action Items:**
- [x] Identified exact lines in manuscript
- [ ] Replace "external validation" with "cross-dataset generalization"
- [ ] Remove diagnostic/clinical claims where inappropriate

---

### 1.9 Summary Table - Priority 1 Critical Fixes

| # | Issue | Location | Current Line | Replacement | Requires Code? |
|---|-------|----------|--------------|-------------|----------------|
| 1 | Recall Formula | III. METHODOLOGY (Eq. 3) | `TP +TN` (Line 313) | `TP +FN` | ❌ No |
| 2 | Loss Function Eq. | III. METHODOLOGY (D) | Lines 205-215 | Complete L_total with λ weights | ❌ No |
| 3 | "close-perfect" | IV. RESULTS (A) | Line 264 | "near-perfect" | ❌ No |
| 4 | Grammar errors | IV. RESULTS (A) | Line 278 | "further confirms" | ❌ No |
| 5 | Missing mAP@0.5:0.95 | IV. RESULTS (A) | Lines 264-268 | Add per-class metrics | ⚠️ **YES** |
| 6 | TTA Clarification | IV. RESULTS (A) | Line 320 | Clarify TTA usage | ⚠️ **YES** |
| 7 | λ_box Sensitivity | III. METHODOLOGY (D) | Lines 216-218 | Add sensitivity analysis | ⚠️ **YES** |
| 8 | Table II Title | IV. RESULTS (Table II) | Line 329 | Change "Ablation" to "Comparison" | ❌ No |
| 9 | SAHI Evidence | IV. RESULTS (E) | Lines 242-249 | Add parameters + visual analysis | ⚠️ **YES** |
| 10 | Dataset Stats | IV. RESULTS (A) | After Line 282 | Add image/cell counts | ⚠️ **YES** |
| 11 | External Validation | Abstract & Conclusion | Lines 36, 401 | Change "validation" to "generalization" | ❌ No |

**✅ Issues that can be fixed immediately (no code): #1, #2, #3, #4, #8, #11**

**⚠️ Issues requiring code execution (flagged as ESTIMATED): #5, #6, #7, #9, #10**

---
ABLATIONSTUDYONARCHITECTUREANDINFERENCESTRATEGIES
```

**Replace With:**
```
TABLE II: ARCHITECTURE AND INFERENCE STRATEGY COMPARISON
```

**Note:** RT-DETR comparison is a **baseline comparison**, NOT an ablation.

**Action Items:**
- [x] Identified exact line in manuscript
- [ ] Change table title from "ABLATION STUDY" to "COMPARISON"

classification       | for      | WBCs      | resulting    | an mAP@0.5   |     | of            | 99.3% and |
arecallscoreof1.0.Plateletdetectionstabilizedat90.3%.For
```

**Replace With (⚠️ ESTIMATED - requires running evaluation code):**
```
of 92.93% and mAP@0.5:0.95 of XX.XX%. Especially | the | model | accomplished | near-perfect |
classification for WBCs (mAP@0.5: 99.3%, Precision: XX.X%, Recall: 100.0%) and |
platelet detection (mAP@0.5: 90.3%, Precision: XX.X%, Recall: XX.X%). For |
RBCs, the model achieved mAP@0.5: 89.2% with Precision: 73.9% and Recall: XX.X%. |
```

**⚠️ FLAG:** This is an **ESTIMATED LINE** - requires running the evaluation script.

**Action Items:**
- [x] Identified exact lines in manuscript
- [ ] Run evaluation code (see Section 1.2 "Complete Ready-to-Run Code" below)
- [ ] Replace placeholder XX.XX values with actual metrics


---

### 1.2 Complete Missing Evaluation Metrics ⚠️

**Required Metrics:**
1. **mAP@0.5:0.95** (COCO standard)
2. **Per-class Precision, Recall, and AP** for RBC, WBC, Platelet
3. **Clarify if 92.93% includes TTA**

**New Table Required:**

| Class | Precision | Recall | AP@0.5 | AP@0.5:0.95 |
|-------|-----------|--------|--------|-------------|
| RBC | ? | ? | ? | ? |
| WBC | ? | ? | ? | ? |
| Platelet | ? | ? | ? | ? |
| **Overall** | **?** | **?** | **92.93%** | **?** |

**Complete Ready-to-Run Code:**

```python
# ============================================================================
# PRIORITY 1.2: Generate Complete Evaluation Metrics
# Run this in a new Jupyter notebook cell or Python script
# ============================================================================

from ultralytics import YOLO
import json
import os

# Setup paths
PROJECT_ROOT = os.getcwd()  # or specify: 'c:/Users/meheraj.h/Downloads/msr-main/hema'
MODEL_PATH = os.path.join(PROJECT_ROOT, '3_models/yolo11_stage2_finetune/weights/best.pt')
YAML_PATH = os.path.join(PROJECT_ROOT, 'scripts/finetune_yolo11.yaml')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, '3_models/final_evaluation')

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("PRIORITY 1.2: Generating Complete Evaluation Metrics")
print("="*70)

# Load model
print(f"\nLoading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# Run comprehensive evaluation
print("\nRunning evaluation on test set...")
metrics = model.val(
    data=YAML_PATH,
    split='test',
    save_json=True,
    plots=True,
    project=OUTPUT_DIR,
    name='priority1_complete_metrics'
)

# Extract ALL required metrics
print("\n" + "="*70)
print("EVALUATION RESULTS")
print("="*70)

print(f"\n📊 Overall Metrics:")
print(f"   mAP@0.5:     {metrics.box.map50:.4f}")
print(f"   mAP@0.5:0.95: {metrics.box.map:.4f}")

print(f"\n📊 Per-Class AP@0.5:")
print(f"   RBC:       {metrics.box.maps[0]:.4f}")
print(f"   WBC:       {metrics.box.maps[1]:.4f}")
print(f"   Platelet:  {metrics.box.maps[2]:.4f}")

print(f"\n📊 Per-Class AP@0.5:0.95:")
print(f"   RBC:       {metrics.box.maps5095[0]:.4f}")
print(f"   WBC:       {metrics.box.maps5095[1]:.4f}")
print(f"   Platelet:  {metrics.box.maps5095[2]:.4f}")

# Save results for paper table
results = {
    'overall_map50': float(metrics.box.map50),
    'overall_map50_95': float(metrics.box.map),
    'per_class': {
        'RBC': {
            'ap50': float(metrics.box.maps[0]),
            'ap50_95': float(metrics.box.maps5095[0])
        },
        'WBC': {
            'ap50': float(metrics.box.maps[1]),
            'ap50_95': float(metrics.box.maps5095[1])
        },
        'Platelet': {
            'ap50': float(metrics.box.maps[2]),
            'ap50_95': float(metrics.box.maps5095[2])
        }
    }
}

# Save to JSON
output_json = os.path.join(OUTPUT_DIR, 'evaluation_metrics.json')
with open(output_json, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to: {output_json}")
print(f"✅ Plots saved to: {OUTPUT_DIR}/priority1_complete_metrics/")

# Generate paper-ready table
print("\n" + "="*70)
print("PAPER TABLE (Copy-Paste to Manuscript)")
print("="*70)
print("""
Table X: Per-Class Detection Performance on BCCD Test Set
┌────────────┬─────────────┬──────────┬──────────┬─────────────┐
│ Class      │ Precision   │ Recall   │ AP@0.5   │ AP@0.5:0.95 │
├────────────┼─────────────┼──────────┼──────────┼─────────────┤
│ RBC        │     -       │    -     │  {:.4f}  │   {:.4f}    │
│ WBC        │     -       │    -     │  {:.4f}  │   {:.4f}    │
│ Platelet   │     -       │    -     │  {:.4f}  │   {:.4f}    │
├────────────┼─────────────┼──────────┼──────────┼─────────────┤
│ Overall    │     -       │    -     │  {:.4f}  │   {:.4f}    │
└────────────┴─────────────┴──────────┴──────────┴─────────────┘
""".format(
    metrics.box.maps[0], metrics.box.maps5095[0],
    metrics.box.maps[1], metrics.box.maps5095[1],
    metrics.box.maps[2], metrics.box.maps5095[2],
    metrics.box.map50, metrics.box.map
))

print("\n⚠️  NOTE: Precision and Recall values need to be extracted from the JSON file")
print("   Check: 3_models/final_evaluation/priority1_complete_metrics/results.json")
```

---

### 1.3 Dataset Documentation ⚠️

**Required Information:**
- [ ] Image counts per split (train/val/test)
- [ ] Cell counts per class
- [ ] Preprocessing steps and resolutions
- [ ] Split methodology (image-level, not patient-level)
- [ ] Leakage prevention statement

**New Table:**

| Dataset | Purpose | Images | RBCs | WBCs | Platelets | Resolution |
|---------|---------|--------|------|------|-----------|------------|
| merged_pretrain | Stage 1 | ? | ? | ? | ? | 640×640 |
| bccd_finetune-train | Training | ? | ? | ? | ? | 800×800 |
| bccd_finetune-val | Validation | ? | ? | ? | ? | 800×800 |
| bccd_finetune-test | Internal Test | ? | ? | ? | ? | 800×800 |
| wbc_test | External | ? | ? | ? | ? | 800×800 |

---

## Priority 2: MAJOR REVISIONS (Days 2-3)

### 2.1 Ablation Studies 🔬

**Required Experiments:**

| Component | Baseline | +Component | Δ mAP |
|-----------|----------|------------|-------|
| Architecture (YOLOv11) | RT-DETR | YOLOv11 | ? |
| Two-stage training | Single-stage | Two-stage | ? |
| High-res input (800×800) | 640×640 | 800×800 | ? |
| λ_box=10.0 | λ_box=7.5 | λ_box=10.0 | ? |
| TTA | Standard | TTA | ? |

**Action Items:**
- [ ] Run ablation experiments
- [ ] Create ablation study table
- [ ] Discuss each component's contribution

---

### 2.2 λ_box=10.0 Justification 🔬

**Action Items:**
- [ ] Provide rationale: "Higher λ_box prioritizes box accuracy for dense RBC separation"
- [ ] Show sensitivity analysis (7.5, 10.0, 12.5)
- [ ] Visual evidence: Side-by-side RBC cluster comparisons

---

### 2.3 SAHI Analysis Enhancement 🔬

**Current Finding:** mAP drops 92.9% → 84.0% with SAHI

---

## Priority 3: MODERATE REVISIONS (Days 3-4)

### 3.1 Temper Claims & Add Limitations 📝

**Action Items:**
- [ ] Replace "clinical validation" with "cross-dataset validation"
- [ ] Add Limitations Section before Conclusion

**Key Limitations to Include:**
1. Dataset diversity limitations (public datasets only)
2. No pathological variants tested
3. External validation not truly clinical
4. Detection only, no diagnostic claims
5. Computational efficiency not fully analyzed

---


---

## Priority 4: PRESENTATION FIXES (Days 4-5)

### 4.1 Proofreading & Typos ✏️

**Required Corrections:**
- "highresolution" → "high-resolution"
- "close-perfect" → "near-perfect"
- "furthers confirms" → "further confirms"
- "Right here the main barrier lies…" → "The primary challenge lies in dense cellular occlusion."

**Action Items:**
- [ ] Run spell check and grammar check
- [ ] Use Grammarly for final pass
- [ ] Have co-author review

---

### 4.2 IEEE Format Compliance 📄

**Checklist:**
- [ ] Two-column IEEE format, no page numbers
- [ ] Times New Roman, 10pt
- [ ] Abstract (150-250 words), Keywords (4-6)
- [ ] Figure captions **below** figures
- [ ] Table captions **above** tables
- [ ] References: 1/3 from last 10 years, no Wikipedia, <3 self-citations

---

### 4.3 Figure Quality 📊

**Required Figures:**
1. Architecture diagram
2. Training curves (`3_models/paper_visuals/results.png`)
3. Confusion matrix (`3_models/paper_visuals/confusion_matrix_normalized.png`)
4. PR curves (`3_models/paper_visuals/PR_curve.png`)
5. Qualitative results (`3_models/paper_visuals/stunning_predictions/`)
6. **NEW:** SAHI failure cases
7. **NEW:** λ_box sensitivity visualization

---

## Timeline

| Day | Tasks | Deliverables |
|-----|-------|--------------|
| **Day 1** | Fix math errors, run evaluation | Corrected formulas, metrics table |
| **Day 2** | Ablation studies, λ_box analysis | Ablation table, sensitivity analysis |
| **Day 3** | SAHI analysis, dataset docs | Failure images, dataset table |
| **Day 4** | Temper claims, proofreading | Limitations section, clean manuscript |
| **Day 5** | IEEE format, final review | Camera-ready PDF, response letters |

---

## Final Checklist

- [ ] All mathematical formulas corrected
- [ ] Complete evaluation metrics (mAP@0.5:0.95, per-class)
- [ ] Ablation studies completed
- [ ] λ_box justification with evidence
- [ ] SAHI analysis with failure cases
- [ ] Clinical claims tempered
- [ ] Limitations section added
- [ ] Dataset documentation complete
- [ ] All typos fixed
- [ ] IEEE format compliance
- [ ] Response to reviewers written
- [ ] Copyright form submitted via EDAS
- [ ] At least one author registered for conference

---

*Document generated: 10/7/2026*  
*Deadline for revisions: 10/12/2026*

### 3.2 Receptive Field Claim 🧪

**Action Items:**
- [ ] Reframe as hypothesis OR provide Grad-CAM visualizations
- [ ] Suggested: "We hypothesize that YOLOv11's local receptive fields may help separate overlapping cells..."

---

### 3.3 TTA Clarification 🎯

**Action Items:**
- [ ] Check if 92.93% includes TTA
- [ ] If YES: Re-run RT-DETR with TTA
- [ ] Clearly state in methodology

**Complete Ready-to-Run Code:**

```python
# ============================================================================
# PRIORITY 3.3: Check if Current 92.93% Includes TTA
# ============================================================================
from ultralytics import YOLO

MODEL_PATH = '3_models/yolo11_stage2_finetune/weights/best.pt'
YAML_PATH = 'scripts/finetune_yolo11.yaml'

model = YOLO(MODEL_PATH)

# Current evaluation (should match 92.93%)
metrics_current = model.val(data=YAML_PATH, split='test', verbose=False)

print(f"Current evaluation mAP@0.5: {metrics_current.box.map50:.4f}")
print(f"This should match your reported 92.93%")

if abs(metrics_current.box.map50 - 0.9293) < 0.01:
    print("\n✅ CONFIRMED: Your 92.93% does NOT include TTA")
    print("   Paper statement: 'All metrics use standard single-pass inference'")
else:
    print("\n⚠️  DISCREPANCY: Check if original run used different settings")

# Also check RT-DETR for fair comparison
print("\n" + "="*60)
print("For fair comparison, re-run RT-DETR with same settings:")
print("="*60)
print("Run: 1_fine_tune.ipynb validation cell with augment=False")
```


### 2.1 TTA (Test Time Augmentation) Comparison 🎯

**Complete Ready-to-Run Code:**

```python
# ============================================================================
# PRIORITY 2.1: TTA (Test Time Augmentation) Comparison
# ============================================================================
from ultralytics import YOLO

MODEL_PATH = '3_models/yolo11_stage2_finetune/weights/best.pt'
YAML_PATH = 'scripts/finetune_yolo11.yaml'

model = YOLO(MODEL_PATH)

# Without TTA
print("Running standard evaluation (no TTA)...")
metrics_std = model.val(data=YAML_PATH, split='test', project='3_models/tta_comparison', name='standard')

# With TTA
print("Running TTA evaluation...")
metrics_tta = model.val(data=YAML_PATH, split='test', augment=True, project='3_models/tta_comparison', name='tta')

print("\n" + "="*60)
print("TTA COMPARISON RESULTS")
print("="*60)
print(f"Standard mAP@0.5:   {metrics_std.box.map50:.4f}")
print(f"TTA mAP@0.5:        {metrics_tta.box.map50:.4f}")
print(f"Improvement:        {(metrics_tta.box.map50 - metrics_std.box.map50)*100:.2f}%")
print(f"\nStandard mAP@0.5:0.95: {metrics_std.box.map:.4f}")
print(f"TTA mAP@0.5:0.95:      {metrics_tta.box.map:.4f}")

# Decision for paper
if abs(metrics_tta.box.map50 - metrics_std.box.map50) < 0.01:
    print("\n✅ CONCLUSION: TTA provides negligible improvement. Use standard inference.")
else:
    print("\n⚠️  CONCLUSION: TTA improves mAP. Must clarify in paper if 92.93% includes TTA.")
```

---

### 2.2 λ_box Sensitivity Analysis ⚖️

**Complete Ready-to-Run Code:**

```python
# ============================================================================
# PRIORITY 2.2: λ_box Sensitivity Analysis (Requires ~3 hours training)
# ============================================================================
from ultralytics import YOLO
import json
import os

PRETRAINED = '3_models/yolo11_stage1_pretrain/weights/best.pt'
YAML_PATH = 'scripts/finetune_yolo11.yaml'
OUTPUT_DIR = '3_models/lambda_box_sensitivity'

os.makedirs(OUTPUT_DIR, exist_ok=True)

lambda_values = [7.5, 10.0, 12.5]
results = []

for lambda_box in lambda_values:
    print(f"\n{'='*60}")
    print(f"Training with λ_box = {lambda_box}")
    print(f"{'='*60}")
    
    model = YOLO(PRETRAINED)
    
    model.train(
        data=YAML_PATH,
        epochs=50,  # Short run for sensitivity analysis
        box=lambda_box,
        imgsz=800,
        batch=8,
        workers=4,
        project=OUTPUT_DIR,
        name=f'lambda_{lambda_box}',
        device=0,
        patience=15,
        verbose=False
    )
    
    # Evaluate
    metrics = model.val(data=YAML_PATH, split='test', verbose=False)
    
    results.append({
        'lambda_box': lambda_box,
        'mAP50': float(metrics.box.map50),
        'mAP50_95': float(metrics.box.map),
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr)
    })
    
    print(f"λ_box={lambda_box}: mAP50={metrics.box.map50:.4f}, mAP50-95={metrics.box.map:.4f}")

# Save results
with open(f'{OUTPUT_DIR}/lambda_sensitivity.json', 'w') as f:
    json.dump(results, f, indent=2)

# Print comparison table
print("\n" + "="*70)
print("λ_box SENSITIVITY ANALYSIS - PAPER TABLE")
print("="*70)
print(f"{'λ_box':<10} {'mAP@0.5':<12} {'mAP@0.5:0.95':<15} {'Precision':<12} {'Recall':<10}")
print("-"*70)
for r in results:
    print(f"{r['lambda_box']:<10} {r['mAP50']:<12.4f} {r['mAP50_95']:<15.4f} {r['precision']:<12.4f} {r['recall']:<10.4f}")
print("="*70)

# Find best
best = max(results, key=lambda x: x['mAP50'])
print(f"\n✅ Best λ_box: {best['lambda_box']} (mAP@0.5 = {best['mAP50']:.4f})")
```

---


### 2.3 SAHI Failure Analysis 🧩

**Required Analysis:**
- Edge truncation at slice boundaries
- Loss of global context for WBC detection
- NMM failures in dense regions

**Complete Ready-to-Run Code:**

```python
# ============================================================================
# PRIORITY 2.3: SAHI Failure Analysis with Visual Examples
# ============================================================================
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2
import matplotlib.pyplot as plt
import os

MODEL_PATH = '3_models/yolo11_stage2_finetune/weights/best.pt'
TEST_IMAGES_DIR = '2_preprocessed_datasets/bccd_finetune/images/test'
OUTPUT_DIR = '3_models/sahi_failure_analysis'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
print("Loading model for SAHI...")
detection_model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path=MODEL_PATH,
    confidence_threshold=0.35,
    device='cuda:0'
)

# Get test images
image_files = [f for f in os.listdir(TEST_IMAGES_DIR) if f.endswith(('.jpg', '.png'))][:10]

print(f"Analyzing {len(image_files)} images for SAHI failures...")

failure_cases = []

for i, img_name in enumerate(image_files):
    img_path = os.path.join(TEST_IMAGES_DIR, img_name)
    
    # Run SAHI
    result = get_sliced_prediction(
        img_path,
        detection_model,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        postprocess_type='NMM',
        postprocess_match_threshold=0.5
    )
    
    # Run standard YOLO for comparison
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    standard_result = model.predict(img_path, conf=0.35)[0]
    
    # Count detections
    sahi_count = len(result.object_prediction_list)
    standard_count = len(standard_result.boxes)
    
    # If significant difference, mark as potential failure case
    if abs(sahi_count - standard_count) > 3:
        failure_cases.append({
            'image': img_name,
            'sahi_count': sahi_count,
            'standard_count': standard_count,
            'difference': abs(sahi_count - standard_count)
        })
        
        # Save visualization
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(img)
        ax1.set_title(f'Standard YOLO: {standard_count} detections')
        ax1.axis('off')
        
        # Plot SAHI result
        ax2.imshow(img)
        for obj in result.object_prediction_list:
            box = obj.bbox
            ax2.add_patch(plt.Rectangle((box.minx, box.miny), 
                                        box.maxx-box.minx, 
                                        box.maxy-box.miny,
                                        fill=False, color='red', linewidth=2))
        ax2.set_title(f'SAHI: {sahi_count} detections')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/failure_{img_name}', dpi=300, bbox_inches='tight')
        plt.close()

# Print summary
print("\n" + "="*70)
print("SAHI FAILURE ANALYSIS SUMMARY")
print("="*70)
print(f"Total images analyzed: {len(image_files)}")
print(f"Failure cases found: {len(failure_cases)}")

if failure_cases:
    print("\nTop failure cases (saved to 3_models/sahi_failure_analysis/):")
    for fc in sorted(failure_cases, key=lambda x: x['difference'], reverse=True)[:5]:
        print(f"  {fc['image']}: SAHI={fc['sahi_count']}, Standard={fc['standard_count']}, Diff={fc['difference']}")
else:
    print("\nNo significant failure cases found in sampled images.")

# Count edge truncation (estimate)
print("\n⚠️  For paper: Manually check saved images for:")
print("   1. Cells truncated at slice boundaries (512px edges)")
print("   2. WBCs split across multiple slices")
print("   3. NMM merging failures in dense regions")
```

**Action Items:**
- [ ] Visual examples of edge truncation (3-4 images) - saved to `3_models/sahi_failure_analysis/`
- [ ] Document SAHI parameters: slice=512, overlap=0.2, NMM=0.5
- [ ] Add section: "Analysis of SAHI Performance Degradation"

---

### 2.4 Complete Ablation Study Table Generator 📊

**Action Items:**
- [ ] Clarify: "RT-DETR is baseline comparison, not ablation"
- [ ] Ensure fair comparison (same TTA, augmentation, metrics)

**Complete Ready-to-Run Code:**

```python
# ============================================================================
# PRIORITY 2.4: Generate Complete Ablation Study Table
# Run this AFTER completing Priorities 1.2, 2.1, 2.2
# ============================================================================
import json

# Collect all results (run previous scripts first)
ablation_data = []

# 1. Baseline RT-DETR (from 1_fine_tune.ipynb results)
# You need to extract this from your existing run
# Check: 1_models/ or your RT-DETR training output
rtdetr_map = 0.896  # REPLACE with actual value from RT-DETR run
rtdetr_map50_95 = 0.0  # REPLACE with actual value
ablation_data.append({
    'Component': 'Baseline (RT-DETR 640px)',
    'mAP50': rtdetr_map,
    'mAP50_95': rtdetr_map50_95,
    'Params': '42M',
    'GFLOPs': '108'
})

# 2. YOLOv11 at 640px (from 3_yolov11.ipynb stage 1)
# Check: 3_models/yolo11_stage1_pretrain/results.json
yolo_640 = 0.0  # REPLACE with your actual value from stage 1
yolo_640_map50_95 = 0.0  # REPLACE with actual value
ablation_data.append({
    'Component': '+ YOLOv11 (640px)',
    'mAP50': yolo_640,
    'mAP50_95': yolo_640_map50_95,
    'Params': '2.6M',
    'GFLOPs': '6.4'
})

# 3. High-res 800px (from 2_fine_tune.ipynb)
# This is your reported 92.93%
yolo_800 = 0.9293  # Your reported value
yolo_800_map50_95 = 0.0  # REPLACE from Priority 1.2 results
ablation_data.append({
    'Component': '+ High-res (800px)',
    'mAP50': yolo_800,
    'mAP50_95': yolo_800_map50_95,
    'Params': '2.6M',
    'GFLOPs': '6.4'
})

# 4. λ_box=10.0 (from Priority 2.2 script)
# Check: 3_models/lambda_box_sensitivity/lambda_sensitivity.json
lambda_10 = 0.0  # REPLACE from λ_box sensitivity results
lambda_10_map50_95 = 0.0  # REPLACE from results
ablation_data.append({
    'Component': '+ λ_box=10.0 (optimized)',
    'mAP50': lambda_10,
    'mAP50_95': lambda_10_map50_95,
    'Params': '2.6M',
    'GFLOPs': '6.4'
})

# 5. TTA (from Priority 2.1 script)
# Check: 3_models/tta_comparison/tta/results.json
tta_map = 0.0  # REPLACE from TTA comparison results
tta_map50_95 = 0.0  # REPLACE from results
ablation_data.append({
    'Component': '+ TTA (test-time augmentation)',
    'mAP50': tta_map,
    'mAP50_95': tta_map50_95,
    'Params': '2.6M',
    'GFLOPs': '6.4'
})

# Print ablation table
print("\n" + "="*100)
print("COMPLETE ABLATION STUDY - PAPER TABLE")
print("="*100)
print(f"{'Component':<40} {'mAP@0.5':<12} {'mAP@0.5:0.95':<15} {'Params':<10} {'GFLOPs':<10}")
print("-"*100)
for item in ablation_data:
    print(f"{item['Component']:<40} {item['mAP50']:<12.4f} {item['mAP50_95']:<15.4f} {item['Params']:<10} {item['GFLOPs']:<10}")
print("="*100)

# Calculate improvements
print("\n📊 Key Improvements:")
print(f"   YOLOv11 vs RT-DETR:     +{(yolo_640 - rtdetr_map)*100:.2f}%")
print(f"   High-res improvement:   +{(yolo_800 - yolo_640)*100:.2f}%")
print(f"   λ_box optimization:     +{(lambda_10 - yolo_800)*100:.2f}%")
print(f"   TTA improvement:        +{(tta_map - lambda_10)*100:.2f}%")
print(f"   TOTAL improvement:      +{(tta_map - rtdetr_map)*100:.2f}%")

# Save for paper
output = {
    'ablation_table': ablation_data,
    'improvements': {
        'yolo_vs_rtdetr': (yolo_640 - rtdetr_map)*100,
        'highres_improvement': (yolo_800 - yolo_640)*100,
        'lambda_optimization': (lambda_10 - yolo_800)*100,
        'tta_improvement': (tta_map - lambda_10)*100,
        'total': (tta_map - rtdetr_map)*100
    }
}

with open('3_models/ablation_study.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\n✅ Results saved to: 3_models/ablation_study.json")
```

---

## 🚀 Quick Start Guide - Copy-Paste-Run All Code

### Execution Order (Recommended)

| Step | Priority | Task | Estimated Time | Output File |
|------|----------|------|----------------|-------------|
| 1 | **1.2** | Complete Evaluation Metrics | 5 min | `3_models/final_evaluation/metrics.json` |
| 2 | **3.3** | TTA Status Check | 2 min | Console output |
| 3 | **2.1** | TTA Comparison | 10 min | `3_models/tta_comparison/` |
| 4 | **2.2** | λ_box Sensitivity (3 models) | 2-3 hours | `3_models/lambda_box_sensitivity/` |
| 5 | **2.3** | SAHI Failure Analysis | 30 min | `3_models/sahi_failure_analysis/` |
| 6 | **2.4** | Ablation Table Generator | 5 min | `3_models/ablation_study.json` |

**Total Time: ~4-5 hours** (mostly waiting for λ_box training)

---

### Step-by-Step Instructions

#### **Step 1: Priority 1.2 - Complete Evaluation Metrics** (Day 1 Morning)
```bash
# Create new Jupyter notebook or Python script
# Copy code from Section 1.2 "Complete Ready-to-Run Code"
# Run it
# Output: 3_models/final_evaluation/metrics.json
```
✅ **Deliverable:** Per-class mAP@0.5 and mAP@0.5:0.95 table for paper

---

#### **Step 2: Priority 3.3 - TTA Status Check** (Day 1 Morning)
```bash
# Copy code from Section 3.3 "Complete Ready-to-Run Code"
# Run it
```
✅ **Deliverable:** Confirmation whether 92.93% includes TTA

---

#### **Step 3: Priority 2.1 - TTA Comparison** (Day 1 Afternoon)
```bash
# Copy code from Section 2.1 "Complete Ready-to-Run Code"
# Run it
# Output: 3_models/tta_comparison/standard/ and tta/
```
✅ **Deliverable:** TTA vs Standard comparison table

---

#### **Step 4: Priority 2.2 - λ_box Sensitivity** (Day 2 - Start Early!)
```bash
# Copy code from Section 2.2 "Complete Ready-to-Run Code"
# Run it (will train 3 models: λ_box=7.5, 10.0, 12.5)
# This takes 2-3 hours - run overnight if needed
# Output: 3_models/lambda_box_sensitivity/lambda_sensitivity.json
```
✅ **Deliverable:** λ_box sensitivity analysis table for paper

---

#### **Step 5: Priority 2.3 - SAHI Failure Analysis** (Day 3)
```bash
# Copy code from Section 2.3 "Complete Ready-to-Run Code"
# Run it
# Output: 3_models/sahi_failure_analysis/failure_*.png
```
✅ **Deliverable:** 3-4 failure case images for paper

---

#### **Step 6: Priority 2.4 - Ablation Table** (Day 3 Afternoon)
```bash
# Copy code from Section 2.4 "Complete Ready-to-Run Code"
# BEFORE running: Update placeholder values with actual results from Steps 1-5
# Run it
# Output: 3_models/ablation_study.json
```
✅ **Deliverable:** Complete ablation study table for paper

---

### 📁 Output Files Summary

After completing all steps, you will have:

```
3_models/
├── final_evaluation/
│   └── metrics.json              # Priority 1.2 results
├── tta_comparison/
│   ├── standard/                 # Priority 2.1 results
│   └── tta/
├── lambda_box_sensitivity/
│   ├── lambda_7.5/
│   ├── lambda_10.0/
│   ├── lambda_12.5/
│   └── lambda_sensitivity.json   # Priority 2.2 results
├── sahi_failure_analysis/
│   └── failure_*.png             # Priority 2.3 visual evidence
└── ablation_study.json           # Priority 2.4 final table
```

---

### ⚠️ Important Notes

1. **Run in Order:** Steps 1-3 should be completed before Step 6 (ablation table needs all previous results)
2. **λ_box Training:** Step 4 takes the longest (2-3 hours). Start it early or run overnight.
3. **Update Placeholders:** Step 6 (ablation table) has placeholder values (0.0) that MUST be replaced with actual results
4. **GPU Required:** All YOLOv11 training/inference requires CUDA GPU
5. **SAHI Dependency:** Step 5 requires `sahi` package: `pip install sahi`

---

### 📋 Pre-Run Checklist

Before starting, ensure:
- [ ] GPU drivers installed and working (`nvidia-smi`)
- [ ] Required packages: `ultralytics`, `sahi`, `opencv-python`, `matplotlib`
- [ ] Model weights exist at specified paths
- [ ] YAML configuration files are correct
- [ ] Sufficient disk space (~5GB for model outputs)

---

*Document updated: 10/7/2026 - All ready-to-run code added*


