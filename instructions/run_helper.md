# run_helper.md — RetiNA-Net experiment commands

Status: commands below assume P1-P7 (see /plan/00_INDEX.md through
07_results_automation.md) have been implemented against the repo. Update
this file again once implementation actually lands and command names are
confirmed against the real code.

## Environment
```bash
conda create -n dr_iccit python=3.12 -y
conda activate dr_iccit
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## P1 — sanity check after overfitting-fix changes
```bash
python test_model.py
python main.py --mode train --ablation proposed --fold 0   # short regression check
```

## Original 4 ablations (K-fold, unchanged)
```bash
python main.py --mode train --ablation baseline
python main.py --mode train --ablation msda_only
python main.py --mode train --ablation hff_only
python main.py --mode train --ablation proposed
```

## P3 — architecture ablation (7 new single-fold rows; row 8 reuses `proposed`)
```bash
python main.py --mode train --ablation arch_baseline --fold 0
python main.py --mode train --ablation arch_msda --fold 0
python main.py --mode train --ablation arch_hff --fold 0
python main.py --mode train --ablation arch_msda_hff --fold 0
python main.py --mode train --ablation arch_attnpool --fold 0
python main.py --mode train --ablation arch_auxhead --fold 0
python main.py --mode train --ablation arch_full --fold 0
```

## P3 — SSL ablation (5 rows)
```bash
python main.py --mode detect_lesions
python main.py --mode pretrain                         # self-supervised (contrastive+multitask)
python main.py --mode pretrain_eyepacs_supervised       # new: missing "EyePACS supervised" row
# "Contrastive only" / "Multi-task only": toggle ssl_use_contrastive /
# ssl_use_multitask in configs/config_ssl.yaml, rerun --mode pretrain
# "ImageNet pretrain": set ssl_pretrained_path: null in configs/config.yaml
```

## Testing (ensemble + multi-scale TTA) — works for any ablation name above
```bash
python main.py --mode test --ablation proposed
python main.py --mode test --ablation arch_attnpool
```

## External validation (Messidor-2)
```bash
python main.py --mode external_validation --ablation proposed
```

## P5 — RepConv/HFF reparameterization ablation
```yaml
# configs/config.yaml: set use_rep_proj: True, then fresh training run
# (NOT compatible with existing checkpoints — see 05_repconv_reparam.md)
```
```bash
python main.py --mode train --ablation proposed --fold 0
python main.py --mode test --ablation proposed
```

## P6 — diffusion pipeline (stretch item — only after P1-P4 measured)
```bash
python main.py --mode train_vae --config configs/config_diffusion.yaml
python main.py --mode train_diffusion_unet --config configs/config_diffusion.yaml
python main.py --mode generate_synthetic --config configs/config_diffusion.yaml
python main.py --mode generation_ablation --config configs/config_diffusion.yaml
```

## XAI heatmaps
```bash
python main.py --mode xai --ablation proposed
```

## P7 — final tables/charts generation (auto-populates from logged results)
```bash
python main.py --mode generate_results
```

## Architecture smoke test
```bash
python test_model.py
```
