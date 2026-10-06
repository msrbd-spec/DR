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
