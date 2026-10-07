# commands.md — Complete Sequential Run Guide (REVISED: everything 5-fold)

## Standing rule (prevents the Phase-3-Step-A bug from recurring)
**Before running any `test`/`external_validation` command, confirm all 5
fold checkpoints exist for that exact config** (`checkpoints/{ablation}/fold0.pth`
through `fold4.pth`, or `checkpoints/{ablation}_decoupled/fold0.pth`..`fold4.pth`
for decoupled runs). If even one fold is missing/stale (e.g. only fold 0
retrained with a new loss setting, folds 1-4 still from an older config),
the ensemble will silently mix inconsistent models and the result is
invalid. This is why every phase below trains all 5 folds before testing
— no more single-fold shortcuts anywhere in this document.

## Checkpoint/log layout (after `03_ablation_harness.md`'s addendum is implemented)
```
checkpoints/{ablation}/fold0.pth ... fold4.pth
checkpoints/{ablation}_decoupled/fold0.pth ... fold4.pth
checkpoints/ssl_pretrained_backbone.pth          (top-level, not per-ablation)
checkpoints/eyepacs_supervised_backbone.pth      (top-level)
logs/{ablation}/{mode}_{timestamp}.log
```

---

## Phase 0 — Fix the merge bug + folder-structure change, sanity check
```bash
python -c "import ast; ast.parse(open('main.py').read())" && echo "main.py OK"
python test_model.py
```

## Phase 1 — SSL backbone (SKIP — already done with old code)
```bash
# python main.py --mode detect_lesions     # SKIP
# python main.py --mode pretrain            # SKIP
```
```yaml
# configs/config.yaml — confirm (should already be set):
ssl_pretrained_path: "checkpoints/ssl_pretrained_backbone.pth"
```

---

## Phase 2 — Architecture ablation (11 configs, all 5-fold)
**Config check:**
```yaml
use_sam: False
use_logit_adjustment: False
use_drw: False
class_weight_strategy: "inverse"
loss_type: "focal"
use_fda: False
use_rep_proj: False
freeze_backbone_stages: 2
mix_prob: 0.4
```
```bash
# Original 4
python main.py --mode train --ablation baseline
python main.py --mode train --ablation msda_only
python main.py --mode train --ablation hff_only
python main.py --mode train --ablation proposed

# P3's 7-row architecture ablation — all 5 folds each
for ablation in arch_baseline arch_msda arch_hff arch_msda_hff arch_attnpool arch_auxhead arch_full; do
  for fold in 0 1 2 3 4; do
    python main.py --mode train --ablation $ablation --fold $fold
  done
done
```
**Wait for ALL of the above to finish, then test everything:**
```bash
python main.py --mode test --ablation baseline
python main.py --mode test --ablation msda_only
python main.py --mode test --ablation hff_only
python main.py --mode test --ablation proposed
for ablation in arch_baseline arch_msda arch_hff arch_msda_hff arch_attnpool arch_auxhead arch_full; do
  python main.py --mode test --ablation $ablation
done
```
`arch_full` and `proposed` share identical flags — their results should
match closely. A persistent mismatch = pipeline bug to investigate first.

**Also run the `mix_prob` diagnostic here (see `03_ablation_harness.md`
addendum) to explain the `proposed` P1-regression, in parallel with the
above (independent, doesn't block anything):**
```yaml
mix_prob: 0.1   # then separately try 0.0
```
```bash
python main.py --mode train --ablation proposed --fold 0   # diagnostic only, single-fold is fine here — not a result you'll report, just a log comparison
```
**Revert `mix_prob: 0.4` before continuing to Phase 3** once you've drawn
a conclusion from the diagnostic log.

→ **STOP. Review all 11 results + the diagnostic log before Phase 3.**

---

## Phase 3 — Training-strategy (P2) ablation — now ALL 5-FOLD, one flag at a time
Each step: edit config → train all 5 folds → test (full ensemble) →
record → next step. Do not skip ahead.

**Step A — Logit Adjustment:**
```yaml
use_logit_adjustment: True
logit_adjustment_tau: 1.0
```
```bash
for fold in 0 1 2 3 4; do python main.py --mode train --ablation proposed --fold $fold; done
python main.py --mode test --ablation proposed
```
**Re-do this step** if you previously tested it under the mixed-ensemble
bug — the number from that run is not trustworthy.

**Step B — + DRW:**
```yaml
use_drw: True
drw_start_frac: 0.6
```
```bash
for fold in 0 1 2 3 4; do python main.py --mode train --ablation proposed --fold $fold; done
python main.py --mode test --ablation proposed
```

**Step C — + Class-Balanced Loss:**
```yaml
class_weight_strategy: "effective_num"
cb_beta: 0.9999
```
```bash
for fold in 0 1 2 3 4; do python main.py --mode train --ablation proposed --fold $fold; done
python main.py --mode test --ablation proposed
```

**Step D — LDAM (keep `use_drw: True` from Step B):**
```yaml
loss_type: "ldam"
ldam_max_margin: 0.5
ldam_scale: 30.0
```
```bash
for fold in 0 1 2 3 4; do python main.py --mode train --ablation proposed --fold $fold; done
python main.py --mode test --ablation proposed
```

**Step E — Decoupled retraining** (on whichever of A-D won), all 5 folds:
```bash
for fold in 0 1 2 3 4; do python main.py --mode decoupled_retrain --ablation proposed --fold $fold; done
python main.py --mode test --ablation proposed_decoupled
```

→ Lock in the best A-E combination for every phase below.

---

## Phase 4 — SAM, all 5 folds, on top of the Phase 3 winner
```yaml
use_sam: True
sam_rho: 0.05
```
```bash
for fold in 0 1 2 3 4; do python main.py --mode train --ablation proposed --fold $fold; done
python main.py --mode test --ablation proposed
```
→ Keep only if it beats Phase 3's result.

---

## Phase 5 — FDA + external validation, all 5 folds
```yaml
use_fda: True
fda_pool_frac: 0.2
fda_prob: 0.3
fda_beta: 0.03
```
```bash
for fold in 0 1 2 3 4; do python main.py --mode train --ablation proposed --fold $fold; done
python main.py --mode external_validation --ablation proposed
```
**Comparison (FDA off), all 5 folds:**
```yaml
use_fda: False
```
```bash
for fold in 0 1 2 3 4; do python main.py --mode train --ablation proposed --fold $fold; done
python main.py --mode external_validation --ablation proposed
```
→ Keep whichever gave the better external accuracy.

---

## Phase 6 — RepConv/HFF reparameterization, all 5 folds (fresh checkpoints)
```yaml
use_rep_proj: True
```
```bash
for fold in 0 1 2 3 4; do python main.py --mode train --ablation proposed --fold $fold; done
python main.py --mode test --ablation proposed
```
→ Efficiency comparison (params/FLOPs/inference time), accuracy should
stay similar. Revert to `False` after logging unless keeping it.

---

## Phase 7 — EyePACS-supervised SSL row
```bash
python main.py --mode pretrain_eyepacs_supervised
```
```yaml
ssl_pretrained_path: null   # temporary, for the "ImageNet pretrain" comparison row
```
```bash
for fold in 0 1 2 3 4; do python main.py --mode train --ablation proposed --fold $fold; done
python main.py --mode test --ablation proposed
```
```yaml
ssl_pretrained_path: "checkpoints/ssl_pretrained_backbone.pth"   # revert immediately after
```

---

## Phase 8 — Final tables/charts
```bash
python main.py --mode generate_results
```

---

## Decision point: is P6 (diffusion) needed?
**See the honest assessment below before starting P6.**