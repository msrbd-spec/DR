# commands.md — Complete Sequential Run Guide (P1-P7)

Follow top to bottom. Each phase states: what config to check/change
BEFORE running, what command(s) to run, and what must finish before you
can start the NEXT phase. Phases marked "independent" can run in any
order relative to each other (not dependent on one another), but each
phase's OWN commands must finish in the order listed within that phase.

---

## Phase 0 — Fix the merge bug, sanity check
**Blocks everything below until done.**
```bash
python -c "import ast; ast.parse(open('main.py').read())" && echo "main.py OK"
python test_model.py
```

## Phase 1 — SSL backbone pretraining
**Run once. Blocks every `train`/`test` command below** (they all default
to loading `checkpoints/ssl_pretrained_backbone.pth` via `config.yaml`'s
`ssl_pretrained_path`).
```yaml
# configs/config.yaml — confirm this is set (should be default already):
ssl_pretrained_path: "checkpoints/ssl_pretrained_backbone.pth"
```
```bash
python main.py --mode detect_lesions
python main.py --mode pretrain
```
Wait for BOTH to fully complete before Phase 2. Do not run again later —
this is the only time you run these two.

---

## Phase 2 — Architecture ablation (isolates MSDA/HFF/heads only)
**Config check before starting — all must be at these defaults:**
```yaml
use_sam: False
use_logit_adjustment: False
use_drw: False
class_weight_strategy: "inverse"
loss_type: "focal"
use_fda: False
use_rep_proj: False
freeze_backbone_stages: 2    # P1 default, leave as-is
mix_prob: 0.4                # P1 default, leave as-is
```
```bash
# Original 4 (K-fold) — independent of each other, any order
python main.py --mode train --ablation baseline
python main.py --mode train --ablation msda_only
python main.py --mode train --ablation hff_only
python main.py --mode train --ablation proposed

# P3's 7-row architecture ablation (single-fold) — independent of each other
python main.py --mode train --ablation arch_baseline --fold 0
python main.py --mode train --ablation arch_msda --fold 0
python main.py --mode train --ablation arch_hff --fold 0
python main.py --mode train --ablation arch_msda_hff --fold 0
python main.py --mode train --ablation arch_attnpool --fold 0
python main.py --mode train --ablation arch_auxhead --fold 0
python main.py --mode train --ablation arch_full --fold 0
```
**Wait for ALL 11 of the above train commands to finish**, then test all
11 (each test command only needs its own train command finished, not the
others — but simplest is to wait for all training first):
```bash
python main.py --mode test --ablation baseline
python main.py --mode test --ablation msda_only
python main.py --mode test --ablation hff_only
python main.py --mode test --ablation proposed
python main.py --mode test --ablation arch_baseline
python main.py --mode test --ablation arch_msda
python main.py --mode test --ablation arch_hff
python main.py --mode test --ablation arch_msda_hff
python main.py --mode test --ablation arch_attnpool
python main.py --mode test --ablation arch_auxhead
python main.py --mode test --ablation arch_full
```
**Note on the 7 `arch_*` rows:** these were trained single-fold
(`--fold 0`), but `test` defaults to expecting a full 5-fold ensemble. If
`n_folds`/`ensemble_folds` aren't adjusted, the test command for these 7
will fail or silently use only whatever checkpoints exist. Before testing
these 7 specifically, either set `ensemble_folds: False` in
`config.yaml`, or confirm the test code path correctly falls back to
single-model evaluation when only 1 fold checkpoint exists — verify this
once on `arch_baseline` before running the rest.

→ **STOP HERE and look at the results** before Phase 3. Confirm which
architecture config (expected: `proposed`/`arch_full`) is winning — that
name replaces `proposed` in every command from Phase 3 onward.

---

## Phase 3 — Training-strategy (P2) ablation — one flag at a time
**Do this BEFORE concluding P6 is needed — this is the main lever for the
Severe/Proliferative weakness in your confusion matrices.**

Each step: edit config → train (single-fold) → test → record result →
move to next step. Do not skip to Step D without doing A-C first (LDAM
is meant to pair with DRW from Step B).

**Step A — Logit Adjustment only:**
```yaml
use_logit_adjustment: True
logit_adjustment_tau: 1.0
```
```bash
python main.py --mode train --ablation proposed --fold 0
python main.py --mode test --ablation proposed
```

**Step B — + DRW** (keep Step A's setting if it helped; your call):
```yaml
use_drw: True
drw_start_frac: 0.6
```
```bash
python main.py --mode train --ablation proposed --fold 0
python main.py --mode test --ablation proposed
```

**Step C — + Class-Balanced Loss:**
```yaml
class_weight_strategy: "effective_num"
cb_beta: 0.9999
```
```bash
python main.py --mode train --ablation proposed --fold 0
python main.py --mode test --ablation proposed
```

**Step D — LDAM (keep `use_drw: True` from Step B):**
```yaml
loss_type: "ldam"
ldam_max_margin: 0.5
ldam_scale: 30.0
```
```bash
python main.py --mode train --ablation proposed --fold 0
python main.py --mode test --ablation proposed
```

**Step E — Decoupled retraining** (run only after whichever of A-D gave
the best result above; loads that checkpoint):
```bash
python main.py --mode decoupled_retrain --ablation proposed --fold 0
python main.py --mode test --ablation proposed_decoupled
```

→ **Lock in whichever combination gave the best Severe/Proliferative F1.**
Leave `config.yaml` set to that combination for every phase below.

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
→ Keep `use_sam: True` only if it beats Phase 3's result; otherwise set
back to `False`.

---

## Phase 5 — FDA (P4) + external validation
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
**Comparison run** (confirm FDA actually helps before keeping it):
```yaml
use_fda: False
```
```bash
python main.py --mode train --ablation proposed --fold 0
python main.py --mode external_validation --ablation proposed
```
→ Keep whichever (`use_fda: True` or `False`) gave the better external
accuracy.

---

## Phase 6 — RepConv/HFF reparameterization (P5) — one-off efficiency row
**Fresh checkpoint required — not compatible with any prior checkpoint.**
```yaml
use_rep_proj: True
```
```bash
python main.py --mode train --ablation proposed --fold 0
python main.py --mode test --ablation proposed
```
→ This is a params/FLOPs/inference-time comparison, accuracy should stay
about the same. Set back to `False` after logging the result unless you
want it in the final model (then also re-run Phases 3-5 with it on, since
it's a fresh checkpoint — your call based on time budget).

---

## Phase 7 — EyePACS-supervised SSL row (independent, no dependency on
anything above except Phase 0)
```bash
python main.py --mode pretrain_eyepacs_supervised
```
**ImageNet-only row** (for the SSL ablation table's baseline comparison —
temporary edit, revert after):
```yaml
ssl_pretrained_path: null
```
```bash
python main.py --mode train --ablation proposed --fold 0
python main.py --mode test --ablation proposed
```
```yaml
# revert immediately after:
ssl_pretrained_path: "checkpoints/ssl_pretrained_backbone.pth"
```

---

## Phase 8 — Final tables/charts
**Run only after every phase above you intend to include in the paper
has finished.**
```bash
python main.py --mode generate_results
```

---

## Decision point: is P6 (diffusion) needed?
Look at your best result after Phases 2-7. If internal test accuracy and
external (Messidor-2) accuracy are still meaningfully short of 97.5%/90%
— proceed to `plan/06_diffusion_pipeline.md` +
`plan/06b_diffusion_pipeline_fix.md`. P6 only retrains the final winning
config ONE more time with real+synthetic data added — none of Phases
0-8 above need to be rerun; they remain your "no synthetic augmentation"
baseline row for P6's own ablation table.