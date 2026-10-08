# commands.md — Full run order (every experiment = ONE command, no loops, no config.yaml edits)

How it works (after `plan/03_ablation_harness.md` is implemented):
- `--ablation X` picks the architecture. `--tag T` names the run (`X__T`) so nothing is overwritten.
- `--set k=v ...` changes config for that run only. The final config is saved as `checkpoints/<run>/config_snapshot.yaml`.
- `train` / `decoupled_retrain` without `--fold` run all 5 folds in the one command.
- `test` / `external_validation` / `xai` automatically restore the config of the run they evaluate. They need all 5 folds trained first.
- `--inherit T` = start from the saved config of run `X__T`, then apply `--set`. This replaces "lock the winner in config.yaml".
- Independent commands can run in parallel on different GPUs: `CUDA_VISIBLE_DEVICES=1 python main.py ...`
- Never start a `test`/`external_validation` for a run until its `train` command has fully finished.
- `configs/config.yaml` is never edited. Defaults must stay: use_sam false, use_logit_adjustment false, use_drw false,
  class_weight_strategy inverse, loss_type focal, use_fda false, use_rep_proj false, freeze_backbone_stages 2, mix_prob 0.4.

---
## Phase -1 — Agent work first (once)
Give the agent `plan/03_ablation_harness.md` (prompt in chat), then run:
```bash
python -c "import ast; ast.parse(open('main.py').read())" && echo "main.py OK"
python test_model.py
```
Also confirm `07c_merge_fix.md` is applied (main.py must parse).

## Phase 1 — SSL backbone: SKIP (already done)
`detect_lesions` and `pretrain` are not rerun. `checkpoints/ssl_pretrained_backbone.pth` is the "Full SSL" backbone.

---
## Phase 2 — Architecture ablation (10 runs; `arch_full` is dropped because it equals `proposed`)
Train (each command trains all 5 folds):
```bash
python main.py --mode train --ablation baseline
python main.py --mode train --ablation msda_only
python main.py --mode train --ablation hff_only
python main.py --mode train --ablation proposed
python main.py --mode train --ablation arch_baseline
python main.py --mode train --ablation arch_msda
python main.py --mode train --ablation arch_hff
python main.py --mode train --ablation arch_msda_hff
python main.py --mode train --ablation arch_attnpool
python main.py --mode train --ablation arch_auxhead
```
Test (after the matching train finished):
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
```
Optional diagnostic for the post-P1 drop of `proposed` (single fold, never reported; compare its log with `proposed` fold 0):
```bash
python main.py --mode train --ablation proposed --tag mix01 --fold 0 --set mix_prob=0.1
python main.py --mode train --ablation proposed --tag mix00 --fold 0 --set mix_prob=0.0
```
-> STOP and review Phase 2 results. `proposed` here is the baseline for Phases 3 and 4.

---
## Phase 3 — SSL ablation (all rows use the default recipe, same as `proposed`; "Full SSL" = `proposed` from Phase 2)
Row "ImageNet pretrain":
```bash
python main.py --mode train --ablation proposed --tag ssl_imagenet --set ssl_pretrained_path=null
python main.py --mode test --ablation proposed --tag ssl_imagenet
```
Row "EyePACS supervised":
```bash
python main.py --mode pretrain_eyepacs_supervised
python main.py --mode train --ablation proposed --tag ssl_eyepacs_sup --set ssl_pretrained_path=checkpoints/eyepacs_supervised_backbone.pth
python main.py --mode test --ablation proposed --tag ssl_eyepacs_sup
```
Row "Contrastive only" (own save dir, so the resume bug cannot skip training):
```bash
cp results/ssl_loss_history.npz results/ssl_loss_history_full.npz
python main.py --mode pretrain --set ssl_use_contrastive=true ssl_use_multitask=false ssl_save_dir=checkpoints/ssl_contrastive_only
python main.py --mode train --ablation proposed --tag ssl_contrastive --set ssl_pretrained_path=checkpoints/ssl_contrastive_only/ssl_pretrained_backbone.pth
python main.py --mode test --ablation proposed --tag ssl_contrastive
```
Row "Multi-task only":
```bash
python main.py --mode pretrain --set ssl_use_contrastive=false ssl_use_multitask=true ssl_save_dir=checkpoints/ssl_multitask_only
python main.py --mode train --ablation proposed --tag ssl_multitask --set ssl_pretrained_path=checkpoints/ssl_multitask_only/ssl_pretrained_backbone.pth
python main.py --mode test --ablation proposed --tag ssl_multitask
cp results/ssl_loss_history_full.npz results/ssl_loss_history.npz
```

---
## Phase 4 — Training-strategy ablation (cumulative, on top of `proposed`; run in this order)
```bash
# Step A: Logit Adjustment
python main.py --mode train --ablation proposed --tag stepA --set use_logit_adjustment=true
python main.py --mode test --ablation proposed --tag stepA

# Step B: + DRW
python main.py --mode train --ablation proposed --tag stepB --set use_logit_adjustment=true use_drw=true
python main.py --mode test --ablation proposed --tag stepB

# Step C: + Class-Balanced Loss
python main.py --mode train --ablation proposed --tag stepC --set use_logit_adjustment=true use_drw=true class_weight_strategy=effective_num
python main.py --mode test --ablation proposed --tag stepC

# Step D: LDAM + DRW (no logit adjustment)
python main.py --mode train --ablation proposed --tag stepD --set loss_type=ldam use_drw=true
python main.py --mode test --ablation proposed --tag stepD
```
If a step made Severe/Proliferative F1 or QWK worse, edit the next step's `--set` to drop that flag.
Pick the best tag as WINNER (below `stepB` is only an example). Decoupled re-training reuses the winner's saved config automatically:
```bash
python main.py --mode decoupled_retrain --ablation proposed --tag stepB
python main.py --mode test --ablation proposed --tag stepB__decoupled
```
If no step beat `proposed`, skip `--inherit` in Phases 5-7 and use plain `--ablation proposed`.

---
## Phase 5 — SAM (on top of the winner)
```bash
python main.py --mode train --ablation proposed --inherit stepB --tag sam --set use_sam=true
python main.py --mode test --ablation proposed --tag sam
```

## Phase 6 — FDA + external validation (Messidor-2)
Replace `stepB` with the best tag so far (`stepB`, `stepB__decoupled`'s base, or `sam`).
```bash
python main.py --mode train --ablation proposed --inherit stepB --tag fda --set use_fda=true
python main.py --mode external_validation --ablation proposed --tag fda
python main.py --mode external_validation --ablation proposed --tag stepB
```
(The last line is the "FDA off" comparison: no retraining needed.) Also run external validation for the final model of each earlier phase you may report, e.g. `--tag sam` or `--tag stepB__decoupled`.

## Phase 7 — RepConv / HFF reparameterization (efficiency row; fresh checkpoints)
```bash
python main.py --mode train --ablation proposed --inherit stepB --tag rep --set use_rep_proj=true
python main.py --mode test --ablation proposed --tag rep
```

## Phase 8 — Final tables and figures
1. Edit `configs/tables.yaml`: replace `<WINNER_STEP>` and `<WINNER>` with real run names, set `main_run` to the final model.
2. Run:
```bash
python main.py --mode xai --ablation proposed --tag stepB
python main.py --mode generate_results
```
(use the final model's tag for `xai`). Outputs: `results/tables/*.csv`, `results/figures/*`.

---
## Decision point — P6 (diffusion)
Only after Phases 2-7. Look at QWK, adjacent-grade accuracy, Severe/Proliferative F1 and external accuracy first.
Raw 5-class accuracy near the ~84.8% APTOS inter-rater level is already close to the label-noise ceiling.
If a real gap remains, use `plan/06_diffusion_pipeline.md` + `plan/06b_diffusion_pipeline_fix.md`; no earlier run needs retraining.