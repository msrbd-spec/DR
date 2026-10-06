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

# Claude Response:
Systematic approach-ta thik ache (ablation track kora, table maintain kora) — kintu **3-ta real issue** paicchi je trust kora-r age fix/verify kora dorkar. Line-by-line:

## 1. `arch_full` sabcheye kharap, `arch_baseline` sabcheye valo — eta flag kora dorkar, accept kora na

Single-fold table-e: `arch_full` (82.24%, QWK 0.8992) = **sabcheye kharap**, `arch_baseline` (84.97%, QWK 0.9253) = **sabcheye valo**. Eta pura paper-er premise-ke contradict kore (MSDA+HFF-i to headline novelty).

**Important clue:** `proposed` 5-fold ensemble = 84.97%, kintu `arch_full` single-fold = 82.24% — **2.73-point gap**, jekhane duitar config literally identical (MSDA+HFF+AttnPool+Aux+Ordinal shob ON). Mane complex architecture-ta **ensemble chara highly unstable** hote pare — fold-0-er particular random split-e kharap pore geche, kintu 5-fold average-e thik hoye jay. Eta ekta **legitimate, reportable finding** (architecture high-variance, needs ensembling) — kintu "architecture kharap" bole conclude kora **premature**।

**Next step (age eta koro, Phase 3-e jawar age na):**
- `arch_baseline` ar `arch_full` — duitar training log-e check koro **ki epoch-e early-stop hoyeche**, ar kono crash/warning silently hoyeche kina
- Possible: ekta 2nd seed/fold diye dutai repeat koro — jodi gap close hoye jay, eta noise chilo; na hoile eta real pattern, paper-e "architecture components benefit from ensembling" hisebe explicitly report korte hobe

## 2. `proposed` P1-fix-er por SLIGHTLY kharap hoyeche (85.25% → 84.97%, QWK 0.9276 → 0.9192)

Eta tomar **sabcheye valo model**-e hoyeche, sabcheye concerning. Possible karon ami age bolechilam: `freeze_backbone_stages: 2` + `mix_prob: 0.4` + CoarseDropout — ei combined regularization-e model-er converge korte **beshi epoch lagte pare**, kintu `patience: 25` same roye geche. **Check koro:** notun `proposed` run-er log-e best-checkpoint-save-hoya epoch number ta — jodi purono run-er (54 epoch) cheye onek aage thame geche, eta-i culprit. Fix: `patience` 25→35-40 koro, rerun koro.

## 3. **Sabcheye critical bug — Phase 3 Step A-er test protocol mismatched**

Table-e likhecho: *"Training: Single-fold (fold 0)... Test: 5-fold ensemble"*. **Eita problem.** Jodi sudhu fold-0 checkpoint-e Logit Adjustment diye retrain korecho, kintu test "5-fold ensemble" mode-e run disho — tahole test actually load korche **fold-0 (notun, logit-adjustment-soho) + fold-1,2,3,4 (purono, logit-adjustment chara)** — ekta **Frankenstein mixed ensemble**। Tomar "+0.28% accuracy, Severe F1 +0.06" result eta theke ashte pare, **actual Logit Adjustment-er effect theke na**।

**Fix:** Phase 3-er protocol consistently single-fold rakhte hobe **duitai** — train `--fold 0` (already korecho) ar test-o single-fold mode-e (`ensemble_folds: False` config-e set kore, ba ensemble logic bypass kore just fold-0 checkpoint evaluate kora)। 5-fold ensemble test shudhu **final winning combination**-er jonno (Phase 8-er age), proti ablation step-er jonno na — commands.md-e eta-i intended chilo, miscommunicate hoye gechilo mone hocche।

**Step A-ke redo/re-verify koro ei fix soho, tarpor-i Step B-e jaw** — noile B/C/D-o same bug carry kore jabe.

---

**Summary: taratari egochho, kintu ei 3-ta methodological gap fix na kore egiye gele je number-gula pacho shegula misleading hote pare.** Fix korar por-i decide koro Step B-e jaba naki na.

---
**`run_helper.md`** — unchanged, no code modified this session.
