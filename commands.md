```bash
# 1. Sanity check the fix
python -c "import ast; ast.parse(open('main.py').read())" && echo "main.py OK"
python test_model.py

# Pre-process data for lesion detection and pretraining
python main.py --mode detect_lesions
python main.py --mode pretrain

# 2. Original 4 ablations (K-fold)
python main.py --mode train --ablation baseline
python main.py --mode train --ablation msda_only
python main.py --mode train --ablation hff_only
python main.py --mode train --ablation proposed

# 3. P3 — 7-row architecture ablation (single-fold)
python main.py --mode train --ablation arch_baseline --fold 0
python main.py --mode train --ablation arch_msda --fold 0
python main.py --mode train --ablation arch_hff --fold 0
python main.py --mode train --ablation arch_msda_hff --fold 0
python main.py --mode train --ablation arch_attnpool --fold 0
python main.py --mode train --ablation arch_auxhead --fold 0
python main.py --mode train --ablation arch_full --fold 0

# 4. Testing (ensemble + multi-scale TTA) — any ablation name above
python main.py --mode test --ablation proposed
python main.py --mode test --ablation baseline
python main.py --mode test --ablation msda_only
python main.py --mode test --ablation hff_only

# 5. P2 — decoupled classifier re-training (after a base model exists)
python main.py --mode decoupled_retrain --ablation proposed --fold 0

# 6. P3 — SSL ablation rows
python main.py --mode detect_lesions
python main.py --mode pretrain
python main.py --mode pretrain_eyepacs_supervised

# 7. External validation (Messidor-2)
python main.py --mode external_validation --ablation proposed

# 8. P7 — generate all tables/charts from logged results
python main.py --mode generate_results
```

For P2/P4/P5 toggles (`use_sam`, `use_logit_adjustment`, `use_drw`, `class_weight_strategy`, `use_fda`, `use_rep_proj`), edit `configs/config.yaml` before rerunning `train`/`test` — each is off by default so existing results stay unaffected.

---