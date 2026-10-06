## **✅ Test Results Summary:**

### **প্রথম ৪টা (5-fold K-fold + Ensemble):**
| Ablation | Accuracy | QWK | Macro F1 | Notes |
|----------|----------|-----|----------|-------|
| `baseline` | 84.15% | 0.9145 | 0.6625 | ✅ 5 folds ensemble |
| `msda_only` | **84.97%** | **0.9188** | **0.6907** | ✅ 5 folds ensemble |
| `hff_only` | 84.43% | 0.9125 | 0.6716 | ✅ 5 folds ensemble |
| `proposed` | **84.97%** | **0.9192** | 0.6759 | ✅ 5 folds ensemble |

### **পরের ৭টা (Single-fold — শুধু fold 0):**
| Ablation | Accuracy | QWK | Macro F1 | Notes |
|----------|----------|-----|----------|-------|
| `arch_baseline` | **84.97%** | **0.9253** | 0.6610 | ✅ 1 fold (fold 0 only) |
| `arch_msda` | 83.88% | 0.9047 | 0.6478 | ✅ 1 fold |
| `arch_hff` | 82.24% | 0.9069 | 0.6515 | ✅ 1 fold |
| `arch_msda_hff` | 83.61% | 0.9118 | 0.6422 | ✅ 1 fold |
| `arch_attnpool` | 84.43% | 0.9159 | 0.6516 | ✅ 1 fold |
| `arch_auxhead` | 84.15% | 0.9089 | 0.6438 | ✅ 1 fold |
| `arch_full` | 82.24% | 0.8992 | 0.6401 | ✅ 1 fold |



# Phase 3 — Training-Strategy Ablation Results

## Overview

Phase 3 evaluates training-strategy techniques to improve Severe (class 3) and Proliferative (class 4) class performance. Each step builds on the previous one (cumulative approach).

**Baseline for comparison:** `proposed` (5-fold ensemble) from Phase 2 architecture ablation.

---

## Step A — Logit Adjustment ✅ COMPLETED

**Configuration:**
```yaml
use_logit_adjustment: True
logit_adjustment_tau: 1.0
use_drw: False
class_weight_strategy: "inverse"
loss_type: "focal"
```

**Training:** Single-fold (fold 0), H100 GPU
**Test:** 5-fold ensemble, A100 GPU

### Results:

| Metric | Baseline (`proposed` 5-fold) | Step A (Logit) | Change |
|--------|------------------------------|----------------|--------|
| **Accuracy** | 84.97% | **85.25%** | **+0.28%** ✅ |
| **QWK** | 0.9192 | **0.9201** | **+0.0009** ✅ |
| **Precision** | 0.7431 | **0.7529** | **+0.0098** ✅ |
| **Recall** | 0.6483 | **0.6557** | **+0.0074** ✅ |
| **Macro F1** | 0.6759 | **0.6834** | **+0.0075** ✅ |

### Class-wise F1 Comparison:

| Class | Baseline | Step A | Change |
|-------|----------|--------|--------|
| 0: No DR | 0.98 | 0.98 | 0.00 |
| 1: Mild | 0.55 | 0.52 | -0.03 ⚠️ |
| 2: Moderate | 0.78 | 0.78 | 0.00 |
| 3: **Severe** | **0.36** | **0.42** | **+0.06** 🎉 |
| 4: **Proliferative** | **0.71** | **0.71** | **0.00** ➖ |

### Decision: ✅ **KEEP** Logit Adjustment

**Rationale:**
- Severe F1 improved by **16.7% relative** (0.36 → 0.42) — primary goal achieved
- Proliferative F1 maintained (0.71)
- Overall metrics slightly improved across the board
- No negative trade-offs

---

## Step B — + DRW (Dynamic Re-Weighting) ⏳ PENDING

**Configuration:**
```yaml
use_logit_adjustment: True    # Keep from Step A
logit_adjustment_tau: 1.0
use_drw: True
drw_start_frac: 0.6
class_weight_strategy: "inverse"
loss_type: "focal"
```

**Status:** Awaiting training

---

## Step C — + Class-Balanced Loss ⏳ PENDING

**Configuration:**
```yaml
use_logit_adjustment: True    # Keep from Step A
use_drw: True                 # Keep from Step B
class_weight_strategy: "effective_num"
cb_beta: 0.9999
loss_type: "focal"
```

**Status:** Awaiting Step B completion

---

## Step D — LDAM Loss ⏳ PENDING

**Configuration:**
```yaml
use_logit_adjustment: False   # LDAM doesn't pair with logit adjustment
use_drw: True                 # LDAM requires DRW
class_weight_strategy: "inverse"
loss_type: "ldam"
ldam_max_margin: 0.5
ldam_scale: 30.0
```

**Status:** Awaiting Step C completion

---

## Summary Table (All Steps)

| Step | Accuracy | QWK | Macro F1 | Severe F1 | Proliferative F1 | Decision |
|------|----------|-----|----------|-----------|------------------|----------|
| Baseline (`proposed`) | 84.97% | 0.9192 | 0.6759 | 0.36 | 0.71 | — |
| **Step A (Logit)** | **85.25%** | **0.9201** | **0.6834** | **0.42** | **0.71** | ✅ KEEP |
| Step B (+DRW) | ___ | ___ | ___ | ___ | ___ | ⏳ Pending |
| Step C (+CB) | ___ | ___ | ___ | ___ | ___ | ⏳ Pending |
| Step D (LDAM) | ___ | ___ | ___ | ___ | ___ | ⏳ Pending |

**Winner:** TBD after all steps complete

---

## Next Steps

1. **Step B:** Train with DRW enabled (keep Logit Adjustment)
2. **Compare:** Severe/Proliferative F1 vs Step A
3. **Decision:** Keep Step A+B combination, or revert to Step A only
4. **Continue:** Steps C and D
5. **Final:** Decoupled re-training on best combination
