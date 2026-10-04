# run_helper.md — Full annotated run sequence (P1-P7)

Logic behind the flag ordering: architecture ablation (Phase 2) must have
ALL P2/P4/P5 flags OFF so it isolates MSDA/HFF/heads only. P2's loss/
sampling techniques (Phase 3) and P1's SAM (Phase 4) are tested on TOP of
the best architecture config, one at a time, so each one's individual
contribution is measurable. FDA (Phase 5) is tested specifically around
the external-validation step since that's what it targets. RepConv
(Phase 6) needs a dedicated fresh run (checkpoint-incompatible).

Every flag is in `configs/config.yaml` unless noted otherwise. Edit the
file, save, then run the command under it. Flags not mentioned in a phase
should stay at whatever the previous phase left them at ONLY if explicitly
told "leave as-is" — otherwise assume default (False/"inverse"/"focal")
unless this file says to change it.

---

## Phase 0 — Fix the bug, sanity check
No flags involved.
```bash
python -c "import ast; ast.parse(open('main.py').read())" && echo "main.py OK"
python test_model.py
```

## Phase 1 — SSL backbone (already done with old code, skip)
```bash
# python main.py --mode detect_lesions     # SKIP — already have the .npz files
# python main.py --mode pretrain            # SKIP — already have checkpoints/ssl_pretrained_backbone.pth
```

## Phase 2 — Architecture ablation sweep
**Config check (should already be defaults — confirm, don't need to edit unless they differ):**
```yaml
use_sam: False
use_logit_adjustment: False
use_drw: False
class_weight_strategy: "inverse"
loss_type: "focal"
use_fda: False
use_rep_proj: False
```
```bash
# Original 4 (K-fold)
python main.py --mode train --ablation baseline
python main.py --mode train --ablation msda_only
python main.py --mode train --ablation hff_only
python main.py --mode train --ablation proposed

# P3's 7-row architecture ablation (single-fold)
python main.py --mode train --ablation arch_baseline --fold 0
python main.py --mode train --ablation arch_msda --fold 0
python main.py --mode train --ablation arch_hff --fold 0
python main.py --mode train --ablation arch_msda_hff --fold 0
python main.py --mode train --ablation arch_attnpool --fold 0
python main.py --mode train --ablation arch_auxhead --fold 0
python main.py --mode train --ablation arch_full --fold 0

# Test all of them
python main.py --mode test --ablation baseline
python main.py --mode test --ablation msda_only
python main.py --mode test --ablation hff_only
python main.py --mode test --ablation proposed
```
→ Whichever architecture wins (expected: `proposed`/`arch_full`), use that
`--ablation` name as the base for every phase below.

## Phase 3 — Training-strategy (P2) ablation, one flag at a time
Each step: edit `config.yaml`, rerun `train --ablation proposed --fold 0`
(re-test with `test --ablation proposed` after each). Do NOT combine steps
except where marked.

**Step A — Logit Adjustment only:**
```yaml
use_logit_adjustment: True
logit_adjustment_tau: 1.0
```
**Step B — + DRW (keep Step A's setting too if it helped, otherwise reset `use_logit_adjustment: False` first — your call based on Step A's result):**
```yaml
use_drw: True
drw_start_frac: 0.6
```
**Step C — + Class-Balanced Loss:**
```yaml
class_weight_strategy: "effective_num"
cb_beta: 0.9999
```
**Step D — LDAM (+DRW), canonical pairing — set loss_type, keep use_drw: True from Step B:**
```yaml
loss_type: "ldam"
ldam_max_margin: 0.5
ldam_scale: 30.0
```
**Step E — Decoupled retraining (run after whichever of A-D gave the best base checkpoint):**
```bash
python main.py --mode decoupled_retrain --ablation proposed --fold 0
```
→ Lock in whichever combination of A-D gave the best Severe/Proliferative
F1 — that becomes your config baseline for Phases 4-6.

---
# RUN AT LEAST TILL THIS, INCLUDING THE EXTERNAL VALIDATION WITH/WITHOUT FDA
---


## Phase 4 — SAM (P1), on top of the Phase 3 winner
```yaml
use_sam: True
sam_rho: 0.05
```
```bash
python main.py --mode train --ablation proposed --fold 0
python main.py --mode test --ablation proposed
```
→ Keep `use_sam: True` only if it beats the Phase 3 winner; otherwise set
back to `False`.

## Phase 5 — FDA (P4), right before external validation
```yaml
use_fda: True
fda_pool_frac: 0.2
fda_prob: 0.3
fda_beta: 0.03
```
```bash
python main.py --mode train --ablation proposed --fold 0
python main.py --mode external_validation --ablation proposed
```
→ Compare against a `use_fda: False` external-validation run of the same
config to confirm it actually helps before keeping it on.

## Phase 6 — RepConv/HFF reparameterization (P5) — separate, one-off row
**Fresh checkpoint required — incompatible with all prior checkpoints.**
```yaml
use_rep_proj: True
```
```bash
python main.py --mode train --ablation proposed --fold 0
python main.py --mode test --ablation proposed
```
→ This is an efficiency comparison (params/FLOPs/inference time), not
expected to change accuracy. Set back to `False` after logging the result
unless you want it in the final model.

## Phase 7 — EyePACS-supervised SSL row (no flag, separate CLI mode)
```bash
python main.py --mode pretrain_eyepacs_supervised
```

## Phase 8 — Final tables/charts
```bash
python main.py --mode generate_results
```

---

## If 97.5%/90% still not reached after all of the above
See `plan/06_diffusion_pipeline.md` + `plan/06b_diffusion_pipeline_fix.md`.