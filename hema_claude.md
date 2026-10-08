Tomar plan-er sathe amar previous analysis merge kore ekta notun consolidated plan baniyechi. Preference onujayi file na banay chat-ei dilam, tai nicher poro block-ta copy kore `REVISION_PLAN.md` hishebe save koro. Ekta jinis agei bole rakhi: leakage fix-er por number kome jete pare, ar seta-i report korte hobe. Reviewer-ra exactly eta-i dekhte chaichhe.

````markdown
# HemaVision-Net: Major-Revision Master Plan (DATA'26, Paper #31)

Deadline: **12 Oct 2026** (from your plan). Today is 8 Oct, so about 4 days remain. GPU time is the bottleneck, so start Cell C3 tonight.

---

## 0. Corrections to the old Review_Resolution_Plan.md

| # | Old plan said | Problem | Fixed here |
|---|---|---|---|
| 1 | "EXACT lines" quoted with line numbers | They were copied from the EDAS PDF text, not from `paper.md` (LaTeX). Find-and-replace will not match. | Section 3 quotes the real `paper.md` text |
| 2 | λ_cls = 1.0, λ_dfl = 1.0 | Your training args show `box=10, cls=0.5, dfl=1.5` | Eq. uses 10 / 0.5 / 1.5 |
| 3 | "Check whether 92.93% includes TTA" (script says it does NOT) | It **does**. No-TTA gave 0.9205 and the TTA run gave 0.9293. The per-class numbers in the paper also come from the TTA run, while Table II calls the row "Standard". | Report both separately, headline = single pass |
| 4 | Placeholder numbers (λ=7.5→91.2%, ...) | These look like results but are invented | Removed. Numbers are filled only from JSON outputs |
| 5 | Dataset text "70/15/15, patient-level, no leakage, 800×800" | Your code does a random 10% **image-level** move. Roboflow augmented siblings (`BloodImage_00168` appears 3 times in test) are very likely in train too. | C1 audits leakage, C2 re-splits by source image |
| 6 | External set "completely independent" | About 800 of the 1931 external images are named `BloodImage_*` (BCCD's own naming). It is WBC-only (1958 instances). | C2 removes them and the text says WBC-only |
| 7 | `metrics.box.maps` as AP50, `maps5095` | `maps` is per-class mAP50-95 and `maps5095` does not exist | C3 uses `ap50`, `ap`, `all_ap`, `p`, `r` |
| 8 | λ_box sweep: 50 epochs, no custom augmentation, 1 seed | Not comparable to the reported model | Same args as the main run, 100 epochs, 2–3 seeds |
| 9 | SAHI script compares detection counts only | The old SAHI run used conf 0.35 + pycocotools, which truncates the PR curve. Standard numbers use conf 0.001. The "1 slices" log means nothing was sliced. Recall 65.2% was AR@[.5:.95], not recall@0.5. Slice sizes also differed (RT-DETR 256, YOLO 512). | C6 sweep with plain-model baseline in the same evaluator and a conf-threshold check |
| 10 | RT-DETR vs YOLO = "CNN beats transformer" | Confounded: RT-DETR-640 used mixup 0.2, box 7.5 and 640 px. Your own RT-DETR-800 run scored 0.928 mAP50 / 0.697 mAP50-95 on test. | C3 trains RT-DETR with the same recipe |
| 11 | Table I: MixUp default 0.1 | Ultralytics default is 0.0 (your log: `mixup=0.0`). `degrees=90` and `flipud=0.5` are the real changes. | New Table I |
| 12 | RT-DETR "42M params" | Your log: 32.8M (31.99M fused) | C5 measures it |
| 13 | Figure copy failed for `PR_curve.png` | Ultralytics names it `BoxPR_curve.png`, `BoxF1_curve.png` | C8 uses the right names |
| 14 | Missing | Novelty framing, efficiency table, other YOLO baselines, FP/FN analysis, occlusion stratification, reproducibility, loss-name fix, clinical-wording sweep | Added |

Kept from the old plan: the recall fix, the loss-label fix, the typo fixes, the Table II rename, "external validation" → "cross-dataset", the limitations section, the IEEE checklist, and the final checklist.

---

## 1. Reviewer coverage matrix

| Reviewer ask | Where it is solved |
|---|---|
| R1-1 / R2-1 novelty, which modules changed | E3, E4, E6, E7 (frame as recipe; architecture is stock YOLO11n) |
| R1-2 more external datasets | C2, C3 (`EXTRA_EXTERNAL`); text says "one external dataset, WBC only" (E12). If you add another dataset, add its yaml. |
| R1-3 efficiency | C5 → `tab_efficiency` |
| R1-4 / R2-2 component ablation | C3 → `tab_ablation` |
| R1-5 / R2-5 recall formula | E10 |
| R1-6 / R2-6 TTA role, 92.93 meaning, mAP50-95, per-class | C3, C4 → `tab_main`, `tab_perclass`; E11, E13 |
| R1-7 / R2-8 clinical claims | E18 (every occurrence listed) |
| R1-8 RBC FP/FN, failure cases | C7, E14b |
| R1-9 / R2-9 SAHI | C6, E9 |
| R1-10 / R2-7 reproducibility, counts, leakage | C1, C2, E5, E17 |
| R2-3 λ_box justification | C3 (box 7.5 / 10 / 12.5, AP75), E8 |
| R2-4 CIoU labelled DFL | E8 |
| R2-10 receptive-field claim | E6, E14 (hypothesis wording) |
| R2-11 limitations | E15 |
| R2-12 proofreading | C9 |
| Other YOLO-based detectors | C3 baselines (YOLOv8n, YOLOv10n, YOLO11s) |

---

## 2. Execution order

| When | Task |
|---|---|
| Tonight | C1, C2 (about 30 min). Then start C3. Run RT-DETR in notebook A (`RUN_ONLY=["rtdetr_640","rtdetr_800"]`) and the rest in notebook B on the same GPU. YOLO11n needs about 3 GB, RT-DETR-800 about 5.4 GB. |
| Day 2 | C5 efficiency, C6 SAHI, C7 error analysis (all need `hv_main_s0` from C3) |
| Day 3 | C4 tables, C8 figures, C9 patch the manuscript, edits E1–E18 |
| Day 4 | Proofreading, IEEE format, response letter, copyright form, conference registration |

Estimated GPU time (from your logs): hv_main 3 runs ≈ 1 h, ablations ≈ 3.5 h, baselines ≈ 1.8 h, RT-DETR ≈ 2.3 h. If time runs out, set `ABL_SEEDS=[0]` and say "single run" in the paper.

---

## 3. Code cells (each is standalone; run from the project root)

### C1: Leakage and dataset audit (read-only)

```python
import os, re, json
from collections import Counter
from PIL import Image

ROOT = os.getcwd()
BC  = os.path.join(ROOT, "2_preprocessed_datasets", "bccd_finetune")
EXT = os.path.join(ROOT, "2_preprocessed_datasets", "wbc_test", "images", "val")
IMG_EXT = (".jpg", ".jpeg", ".png")
names = {0: "RBC", 1: "WBC", 2: "Platelet"}
src_id = lambda f: re.split(r"\.rf\.", f)[0]      # BloodImage_00168_jpg.rf.xxx.jpg -> BloodImage_00168_jpg

ids, files_by = {}, {}
for s in ["train", "val", "test"]:
    fs = [f for f in os.listdir(os.path.join(BC, "images", s)) if f.lower().endswith(IMG_EXT)]
    files_by[s] = fs; ids[s] = {src_id(f) for f in fs}
    cnt = Counter()
    for f in fs:
        lp = os.path.join(BC, "labels", s, os.path.splitext(f)[0] + ".txt")
        if os.path.exists(lp):
            for ln in open(lp):
                if ln.strip(): cnt[names[int(float(ln.split()[0]))]] += 1
    print(f"{s:5s}: {len(fs):4d} images | {len(ids[s]):4d} source images | {dict(cnt)}")

for a, b in [("train", "test"), ("train", "val"), ("val", "test")]:
    print(f"{a} ∩ {b}: {len(ids[a] & ids[b])} shared source images")
leaky = [f for f in files_by["test"] if src_id(f) in ids["train"]]
print(f"TEST images whose source image also sits in TRAIN: {len(leaky)}/{len(files_by['test'])}")

sizes = Counter(Image.open(os.path.join(BC, "images", "test", f)).size for f in files_by["test"])
print("BCCD image sizes (W,H):", sizes)

ext_files = [f for f in os.listdir(EXT) if f.lower().endswith(IMG_EXT)]
ext_bccd = [f for f in ext_files if f.startswith("BloodImage_")]
print(f"External: {len(ext_files)} images, {len(ext_bccd)} have BCCD-style names (BloodImage_*)")
print("External BCCD-style source ids also present in BCCD splits:",
      len({src_id(f) for f in ext_bccd} & set().union(*ids.values())))
```

Read the result like this:
- If "TEST images whose source also sits in TRAIN" is greater than 0, the old 92.93% was inflated and C2 is mandatory.
- Note the printed image size. You need it for E7 (`{{IMG_W}}×{{IMG_H}}`) and for judging SAHI.

### C2: Source-grouped re-split, clean external set, manifest

```python
import os, re, json, shutil, random, yaml
import numpy as np
from collections import defaultdict, Counter
from PIL import Image

ROOT = os.getcwd(); SEED = 42
SRC = os.path.join(ROOT, "2_preprocessed_datasets", "bccd_finetune")
DST = os.path.join(ROOT, "2_preprocessed_datasets", "bccd_grouped")
ESRC = os.path.join(ROOT, "2_preprocessed_datasets", "wbc_test")
EDST = os.path.join(ROOT, "2_preprocessed_datasets", "wbc_test_clean")
OUT = os.path.join(ROOT, "3_models", "revision"); os.makedirs(OUT, exist_ok=True)
IMG_EXT = (".jpg", ".jpeg", ".png"); names = {0: "RBC", 1: "WBC", 2: "Platelet"}
src_id = lambda f: re.split(r"\.rf\.", f)[0]

# ---------- 1) group split of all BCCD images by source image (80/10/10) ----------
items = []
for sp in ["train", "val", "test"]:
    idir = os.path.join(SRC, "images", sp); ldir = os.path.join(SRC, "labels", sp)
    for f in os.listdir(idir):
        if f.lower().endswith(IMG_EXT):
            items.append((os.path.join(idir, f), os.path.join(ldir, os.path.splitext(f)[0] + ".txt"), f, src_id(f)))
groups = defaultdict(list)
for it in items: groups[it[3]].append(it)
keys = sorted(groups); random.Random(SEED).shuffle(keys)
n = len(keys); n_test = round(0.10 * n); n_val = round(0.10 * n)
split_of = {k: ("test" if i < n_test else "val" if i < n_test + n_val else "train") for i, k in enumerate(keys)}

if os.path.exists(DST): shutil.rmtree(DST)
for sp in ["train", "val", "test"]:
    os.makedirs(os.path.join(DST, "images", sp)); os.makedirs(os.path.join(DST, "labels", sp))
manifest = {"seed": SEED, "rule": "group by Roboflow source image, 80/10/10", "counts": {}}
cnts = {sp: Counter() for sp in ["train", "val", "test"]}
for k, lst in groups.items():
    sp = split_of[k]
    for ip, lp, f, _ in lst:
        shutil.copy2(ip, os.path.join(DST, "images", sp, f))
        dl = os.path.join(DST, "labels", sp, os.path.splitext(f)[0] + ".txt")
        if os.path.exists(lp): shutil.copy2(lp, dl)
        else: open(dl, "w").close()
        cnts[sp]["images"] += 1
        for ln in open(dl):
            if ln.strip(): cnts[sp][names[int(float(ln.split()[0]))]] += 1
for sp in cnts:
    cnts[sp]["source_images"] = sum(1 for k in keys if split_of[k] == sp)
    manifest["counts"][sp] = dict(cnts[sp]); print(sp, dict(cnts[sp]))
w, h = Image.open(items[0][0]).size; manifest["img_w"], manifest["img_h"] = w, h

yaml.dump({"path": DST, "train": "images/train", "val": "images/val", "test": "images/test",
           "nc": 3, "names": names}, open(os.path.join(ROOT, "scripts", "finetune_grouped.yaml"), "w"), default_flow_style=False)

# ---------- 2) external set without BCCD-named images ----------
if os.path.exists(EDST): shutil.rmtree(EDST)
os.makedirs(os.path.join(EDST, "images", "val")); os.makedirs(os.path.join(EDST, "labels", "val"))
kept = removed = inst = 0
for f in os.listdir(os.path.join(ESRC, "images", "val")):
    if not f.lower().endswith(IMG_EXT): continue
    if f.startswith("BloodImage_"): removed += 1; continue
    shutil.copy2(os.path.join(ESRC, "images", "val", f), os.path.join(EDST, "images", "val", f))
    lp = os.path.join(ESRC, "labels", "val", os.path.splitext(f)[0] + ".txt")
    dl = os.path.join(EDST, "labels", "val", os.path.splitext(f)[0] + ".txt")
    if os.path.exists(lp): shutil.copy2(lp, dl); inst += sum(1 for l in open(lp) if l.strip())
    else: open(dl, "w").close()
    kept += 1
manifest["external"] = {"kept": kept, "removed_bccd_named": removed, "instances": inst}
print("external kept", kept, "removed", removed, "instances", inst)
yaml.dump({"path": EDST, "train": "images/val", "val": "images/val", "nc": 3, "names": names},
          open(os.path.join(ROOT, "scripts", "external_clean.yaml"), "w"), default_flow_style=False)

# ---------- 3) near-duplicate check external vs BCCD (average hash, Hamming <= 4) ----------
def ahash(p):
    a = np.asarray(Image.open(p).convert("L").resize((8, 8)), dtype=np.float32)
    return (a > a.mean()).flatten()
B = np.array([ahash(it[0]) for it in items])
E = np.array([ahash(os.path.join(EDST, "images", "val", f)) for f in os.listdir(os.path.join(EDST, "images", "val"))])
near = 0
for i in range(0, len(E), 200):
    d = (E[i:i+200, None, :] != B[None, :, :]).sum(-1)
    near += int((d.min(1) <= 4).sum())
manifest["external"]["near_duplicates_of_bccd"] = near
print("external images with a near-duplicate in BCCD:", near, "(flip/rotate copies can escape this check)")
json.dump(manifest, open(os.path.join(OUT, "split_manifest.json"), "w"), indent=2)
print("saved", os.path.join(OUT, "split_manifest.json"))
```

If `near_duplicates_of_bccd` is large, say so in E12 and also drop those images.

### C3: Unified train + eval queue (resumable, long)

Everything uses the same data, protocol (`conf=0.001, iou=0.7`, same `val()` call) and recipe args as your original run. Results go to `3_models/revision/results/*.json`.

```python
import os, json, time, gc, traceback, torch
import pandas as pd
from ultralytics import YOLO, RTDETR

ROOT = os.getcwd(); OUT = os.path.join(ROOT, "3_models", "revision")
RES = os.path.join(OUT, "results"); os.makedirs(RES, exist_ok=True)
Y_BC  = os.path.join(ROOT, "scripts", "finetune_grouped.yaml")
Y_EXT = os.path.join(ROOT, "scripts", "external_clean.yaml")
Y_PRE = os.path.join(ROOT, "scripts", "pretrain_yolo11.yaml")
S1_Y11N = os.path.join(ROOT, "3_models", "yolo11_stage1_pretrain", "weights", "best.pt")
S1_RTD  = os.path.join(ROOT, "3_models", "stage1_pretrained", "weights", "best.pt")
EXTRA_EXTERNAL = []          # e.g. [("raabin", "/path/to/raabin.yaml")] -> evaluated as external_<name>
MAIN_SEEDS, ABL_SEEDS = [0, 1, 2], [0, 1]
RUN_ONLY = None              # e.g. ["rtdetr_640","rtdetr_800"] to split work across two notebooks

def cfg(tag, family="yolo", arch="yolo11n.pt", stage1=S1_Y11N, imgsz=800, box=10.0, mixup=0.0, batch=8, seeds=MAIN_SEEDS):
    return dict(tag=tag, family=family, arch=arch, stage1=stage1, imgsz=imgsz, box=box, mixup=mixup, batch=batch, seeds=seeds)

CONFIGS = [
    cfg("hv_main"),
    cfg("abl_box7.5", box=7.5, seeds=ABL_SEEDS),
    cfg("abl_box12.5", box=12.5, seeds=ABL_SEEDS),
    cfg("abl_res640", imgsz=640, seeds=ABL_SEEDS),
    cfg("abl_nostage1", stage1=None, seeds=ABL_SEEDS),
    cfg("abl_mixup0.1", mixup=0.1, seeds=ABL_SEEDS),
    cfg("base_yolov8n", arch="yolov8n.pt", stage1="auto", seeds=[0]),
    cfg("base_yolov10n", arch="yolov10n.pt", stage1="auto", seeds=[0]),
    cfg("base_yolo11s", arch="yolo11s.pt", stage1="auto", seeds=[0]),
    cfg("rtdetr_640", family="rtdetr", arch="rtdetr-l.pt", stage1=S1_RTD, imgsz=640, box=7.5, batch=8, seeds=[0]),
    cfg("rtdetr_800", family="rtdetr", arch="rtdetr-l.pt", stage1=S1_RTD, imgsz=800, box=7.5, batch=4, seeds=[0]),
]
COMMON = dict(epochs=100, patience=25, workers=4, device=0, lr0=0.001, lrf=0.01, mosaic=1.0, degrees=90.0,
              fliplr=0.5, flipud=0.5, cls=0.5, deterministic=True, exist_ok=True, verbose=False)

def get_stage1(c):
    stem = os.path.splitext(c["arch"])[0]
    w = os.path.join(OUT, "stage1", stem, "weights", "best.pt")
    if not os.path.exists(w):
        YOLO(c["arch"]).train(data=Y_PRE, epochs=50, patience=15, imgsz=640, batch=16, workers=4, device=0, seed=0,
                              deterministic=True, project=os.path.join(OUT, "stage1"), name=stem, exist_ok=True, verbose=False)
    return w

def mdict(m):
    b = m.box; idx = [int(i) for i in b.ap_class_index]
    d = {"map50": float(b.map50), "map5095": float(b.map), "map75": float(b.map75), "p": float(b.mp), "r": float(b.mr), "per_class": {}}
    for j, c in enumerate(idx):
        d["per_class"][m.names[c]] = {"p": float(b.p[j]), "r": float(b.r[j]), "ap50": float(b.ap50[j]),
                                      "ap5095": float(b.ap[j]), "ap75": float(b.all_ap[j, 5])}
    d["speed_ms"] = {k: float(v) for k, v in m.speed.items()}
    return d

def evaluate(Model, w, imgsz, data, split, name, tta):
    mdl = Model(w)
    kw = dict(data=data, split=split, imgsz=imgsz, conf=0.001, iou=0.7, batch=8, plots=True, verbose=False,
              project=os.path.join(OUT, "eval"), name=name, exist_ok=True)
    if tta: kw["augment"] = True
    r = mdict(mdl.val(**kw)); del mdl; torch.cuda.empty_cache(); return r

def run_one(c, seed):
    name = f"{c['tag']}_s{seed}"; resf = os.path.join(RES, name + ".json")
    if os.path.exists(resf): print("skip", name); return
    Model = RTDETR if c["family"] == "rtdetr" else YOLO
    start = get_stage1(c) if c["stage1"] == "auto" else (c["stage1"] or c["arch"])
    t0 = time.time()
    Model(start).train(data=Y_BC, imgsz=c["imgsz"], box=c["box"], mixup=c["mixup"], batch=c["batch"], seed=seed,
                       project=os.path.join(OUT, "runs"), name=name, **COMMON)
    tmin = (time.time() - t0) / 60
    rundir = os.path.join(OUT, "runs", name); best = os.path.join(rundir, "weights", "best.pt")
    out = {"tag": c["tag"], "seed": seed, "cfg": {k: v for k, v in c.items() if k != "seeds"}, "weights": best,
           "train_min": tmin, "n_epochs": int(len(pd.read_csv(os.path.join(rundir, "results.csv"))))}
    jobs = [("internal_std", Y_BC, "test", False), ("internal_tta", Y_BC, "test", True),
            ("external_std", Y_EXT, "val", False), ("external_tta", Y_EXT, "val", True)]
    jobs += [(f"external_{n}", y, "val", False) for n, y in EXTRA_EXTERNAL]
    for key, data, split, tta in jobs:
        if tta and c["family"] == "rtdetr": continue
        try: out[key] = evaluate(Model, best, c["imgsz"], data, split, f"{name}_{key}", tta)
        except Exception as e: print("eval failed", name, key, e); out[key] = None
    json.dump(out, open(resf, "w"), indent=2)
    print(f"done {name}: internal mAP50={out['internal_std']['map50']:.4f}  ({tmin:.1f} min)")
    gc.collect(); torch.cuda.empty_cache()

for c in CONFIGS:
    if RUN_ONLY and c["tag"] not in RUN_ONLY: continue
    for s in c["seeds"]:
        try: run_one(c, s)
        except Exception: print("FAILED", c["tag"], s); traceback.print_exc()
```

Notes for the paper (put in E17):
- `lr0=0.001` is ignored because `optimizer=auto` picks AdamW with lr 0.001429 (see your logs).
- RT-DETR here uses the same augmentation, mixup 0 and the same stage-1 weights.
- Re-running the cell skips finished runs, so a crash is harmless.

### C4: Aggregate → LaTeX tables (run after C3, C5, C6)

```python
import os, json, glob, numpy as np
ROOT = os.getcwd(); OUT = os.path.join(ROOT, "3_models", "revision")
RES = os.path.join(OUT, "results"); TAB = os.path.join(OUT, "tables"); os.makedirs(TAB, exist_ok=True)
runs = {}
for f in glob.glob(os.path.join(RES, "*.json")):
    d = json.load(open(f)); runs.setdefault(d["tag"], []).append(d)

def vals(tag, ev, fn):
    out = []
    for r in runs.get(tag, []):
        try:
            if r.get(ev): out.append(fn(r[ev]))
        except KeyError: pass
    return out
def ms(xs, pct=True, dp=1):
    if not xs: return "--"
    a = np.array(xs, float) * (100 if pct else 1)
    return f"{a[0]:.{dp}f}" if len(a) == 1 else f"{a.mean():.{dp}f}$\\pm${a.std(ddof=1):.{dp}f}"
mean = lambda xs: float(np.mean(xs)) if xs else float("nan")
DISP = {"hv_main": "HemaVision-Net (YOLO11n)", "rtdetr_640": "RT-DETR-L (640 px)", "rtdetr_800": "RT-DETR-L (800 px)",
        "base_yolov8n": "YOLOv8n", "base_yolov10n": "YOLOv10n", "base_yolo11s": "YOLO11s",
        "abl_res640": "Fine-tune at 640 px", "abl_box7.5": "$\\lambda_{box}=7.5$", "abl_box12.5": "$\\lambda_{box}=12.5$",
        "abl_mixup0.1": "MixUp 0.1", "abl_nostage1": "No stage-1 pretraining"}

def write_table(fn, cap, label, header, rows):
    body = " \\\\ \\hline\n".join(" & ".join(r) for r in rows)
    s = ("\\begin{table}[htbp]\n\\caption{%s}\n\\label{%s}\n\\centering\\scriptsize\n\\resizebox{\\columnwidth}{!}{\n"
         "\\begin{tabular}{|l|%s|}\n\\hline\n%s \\\\ \\hline\n%s \\\\ \\hline\n\\end{tabular}}\n\\end{table}\n") % (
         cap, label, "c" * (len(header) - 1), " & ".join("\\textbf{%s}" % h for h in header), body)
    open(os.path.join(TAB, fn), "w").write(s); print("\n== ", fn, "\n", "\n".join(" | ".join(r) for r in rows))

G = lambda k: (lambda e: e[k])
# --- main comparison ---
rows = []
for tag, ev, lab in [("hv_main", "internal_std", "HemaVision-Net (YOLO11n), single pass"), ("hv_main", "internal_tta", "HemaVision-Net (YOLO11n), + TTA"),
                     ("rtdetr_640", "internal_std", DISP["rtdetr_640"]), ("rtdetr_800", "internal_std", DISP["rtdetr_800"]),
                     ("base_yolov8n", "internal_std", "YOLOv8n"), ("base_yolov10n", "internal_std", "YOLOv10n"), ("base_yolo11s", "internal_std", "YOLO11s")]:
    if tag in runs:
        rows.append([lab, str(len(runs[tag])), ms(vals(tag, ev, G("p"))), ms(vals(tag, ev, G("r"))), ms(vals(tag, ev, G("map50"))), ms(vals(tag, ev, G("map5095")))])
write_table("tab_main.tex", "Detection results on the source-grouped BCCD test split (mean$\\pm$std over $n$ seeds; same protocol for all models)",
            "tab:main", ["Model", "n", "P", "R", "mAP@0.5", "mAP@0.5:0.95"], rows)
# --- per class ---
rows = [[c] + [ms(vals("hv_main", "internal_std", (lambda c, k: lambda e: e["per_class"][c][k])(c, k))) for k in ["p", "r", "ap50", "ap5095"]] for c in ["RBC", "WBC", "Platelet"]]
write_table("tab_perclass.tex", "Per-class results of HemaVision-Net on the BCCD test split (single pass)", "tab:perclass",
            ["Class", "P", "R", "AP@0.5", "AP@0.5:0.95"], rows)
# --- ablation ---
base50 = mean(vals("hv_main", "internal_std", G("map50"))); rows = []
for tag in ["hv_main", "abl_res640", "abl_box7.5", "abl_box12.5", "abl_mixup0.1", "abl_nostage1"]:
    if tag in runs:
        d = (mean(vals(tag, "internal_std", G("map50"))) - base50) * 100
        rows.append([DISP[tag], str(len(runs[tag])), ms(vals(tag, "internal_std", G("map50"))), ms(vals(tag, "internal_std", G("map5095"))),
                     ms(vals(tag, "internal_std", G("map75"))), ms(vals(tag, "internal_std", lambda e: e["per_class"]["RBC"]["ap75"])), f"{d:+.1f}"])
rows.append(["+ TTA at test time", str(len(runs.get("hv_main", []))), ms(vals("hv_main", "internal_tta", G("map50"))), ms(vals("hv_main", "internal_tta", G("map5095"))),
             ms(vals("hv_main", "internal_tta", G("map75"))), ms(vals("hv_main", "internal_tta", lambda e: e["per_class"]["RBC"]["ap75"])),
             f"{(mean(vals('hv_main', 'internal_tta', G('map50'))) - base50) * 100:+.1f}"])
write_table("tab_ablation.tex", "Component ablation (one change at a time relative to the full recipe)", "tab:ablation",
            ["Variant", "n", "mAP@0.5", "mAP@0.5:0.95", "mAP@0.75", "RBC AP@0.75", "$\\Delta$mAP@0.5"], rows)
# --- external ---
rows = [[DISP[t], ms(vals(t, "external_std", lambda e: e["per_class"]["WBC"]["p"])), ms(vals(t, "external_std", lambda e: e["per_class"]["WBC"]["r"])),
         ms(vals(t, "external_std", lambda e: e["per_class"]["WBC"]["ap50"])), ms(vals(t, "external_std", lambda e: e["per_class"]["WBC"]["ap5095"]))]
        for t in ["hv_main", "rtdetr_640", "rtdetr_800", "base_yolov8n", "base_yolov10n", "base_yolo11s"] if t in runs]
write_table("tab_external.tex", "Cross-dataset evaluation on the Kaggle WBC detection set (WBC class only; BCCD-named images removed)", "tab:external",
            ["Model", "P", "R", "AP@0.5", "AP@0.5:0.95"], rows)
# --- efficiency / sahi if present ---
ef = os.path.join(OUT, "efficiency.json")
if os.path.exists(ef):
    rows = [[DISP.get(r["tag"], r["tag"]), str(r["imgsz"]), f"{r['params_M']:.2f}", f"{r['gflops']:.1f}", f"{r['ms_fp32']:.1f}",
             f"{r['ms_fp16']:.1f}", f"{r['ms_cpu']:.0f}", f"{r['vram_mb']:.0f}"] for r in json.load(open(ef))]
    write_table("tab_efficiency.tex", "Model size and speed (batch 1, network forward only, RTX 4070 SUPER; CPU = Ryzen 7 7700)", "tab:efficiency",
                ["Model", "Input", "Params (M)", "GFLOPs", "GPU fp32 ms", "GPU fp16 ms", "CPU ms", "Peak VRAM MB"], rows)
sj = os.path.join(OUT, "sahi_sweep.json")
if os.path.exists(sj):
    S = json.load(open(sj)); b = S["baseline"]
    rows = [["single pass", "--", "--", "1", f"{b['map50']*100:.1f}", f"{b['map50_c35']*100:.1f}", f"{b['recall50']*100:.1f}"]]
    for r in sorted(S["configs"], key=lambda r: -r["map50"]):
        rows.append([f"SAHI {r['slice']}", f"{r['overlap']}", r["post"], str(r["n_slices"]), f"{r['map50']*100:.1f}", f"{r['map50_c35']*100:.1f}", f"{r['recall50']*100:.1f}"])
    write_table("tab_sahi.tex", "SAHI sweep versus single-pass inference (pycocotools, conf$\\geq$0.001; last-but-one column keeps only detections with conf$\\geq$0.35)",
                "tab:sahi", ["Setting", "Overlap", "Merge", "Slices", "mAP@0.5", "mAP@0.5 (conf$\\geq$.35)", "Recall@0.5"], rows)

# --- decision rule for the narrative ---
a, b = vals("hv_main", "internal_std", G("map50")), vals("rtdetr_800", "internal_std", G("map50"))
if a and b:
    sd = np.std(a, ddof=1) if len(a) > 1 else 0
    print(f"\nDECISION: HV {mean(a)*100:.1f} vs RT-DETR-800 {mean(b)*100:.1f} (HV seed std {sd*100:.1f})")
    print("-> If RT-DETR >= HV or within 1 std: write 'comparable accuracy at a fraction of the compute' and DELETE every 'CNN handles occlusion better' claim.")
    print("-> If HV is higher by more than 2 std: you may say HV is more accurate under this protocol, still as an observation, not a mechanism.")
```

### C5: Efficiency (params, GFLOPs, latency, VRAM)

```python
import os, time, json, copy, gc, torch
from ultralytics import YOLO, RTDETR
from ultralytics.utils.torch_utils import get_flops

ROOT = os.getcwd(); OUT = os.path.join(ROOT, "3_models", "revision")
TAGS = {"hv_main": YOLO, "base_yolov8n": YOLO, "base_yolov10n": YOLO, "base_yolo11s": YOLO, "rtdetr_640": RTDETR, "rtdetr_800": RTDETR}
CPU_THREADS = 8
torch.set_num_threads(CPU_THREADS)

@torch.no_grad()
def bench(net, x, iters, cuda):
    for _ in range(5): net(x)
    if cuda: torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters): net(x)
    if cuda: torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1000

rows = []
for tag, M in TAGS.items():
    w = os.path.join(OUT, "runs", f"{tag}_s0", "weights", "best.pt")
    if not os.path.exists(w): print("missing", w); continue
    for sz in (640, 800):
        net = M(w).model.float().eval()
        try: net.fuse(verbose=False)
        except Exception: pass
        params = sum(p.numel() for p in net.parameters()) / 1e6
        try: gf = float(get_flops(net, sz))
        except Exception: gf = float("nan")
        g = copy.deepcopy(net).cuda()
        x = torch.randn(1, 3, sz, sz).cuda()
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        ms32 = bench(g, x, 100, True); vram = torch.cuda.max_memory_allocated() / 2**20
        gh = copy.deepcopy(g).half(); ms16 = bench(gh, x.half(), 100, True)
        c = copy.deepcopy(net).cpu(); ms_cpu = bench(c, torch.randn(1, 3, sz, sz), 10, False)
        rows.append(dict(tag=tag, imgsz=sz, params_M=params, gflops=gf, ms_fp32=ms32, ms_fp16=ms16, ms_cpu=ms_cpu, vram_mb=vram))
        print(rows[-1]); del net, g, gh, c, x; gc.collect(); torch.cuda.empty_cache()
json.dump(rows, open(os.path.join(OUT, "efficiency.json"), "w"), indent=2)
```

### C6: SAHI sweep with a proper baseline

```python
import os, io, json, contextlib, itertools, math, copy, cv2
import numpy as np
from collections import Counter
from PIL import Image
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = os.getcwd(); OUT = os.path.join(ROOT, "3_models", "revision")
W = os.path.join(OUT, "runs", "hv_main_s0", "weights", "best.pt")
TI = os.path.join(ROOT, "2_preprocessed_datasets", "bccd_grouped", "images", "test")
TL = os.path.join(ROOT, "2_preprocessed_datasets", "bccd_grouped", "labels", "test")
files = sorted(f for f in os.listdir(TI) if f.lower().endswith((".jpg", ".jpeg", ".png")))
dims = {f: Image.open(os.path.join(TI, f)).size for f in files}
print("image sizes:", Counter(dims.values()))
W0, H0 = dims[files[0]]

# ---- ground truth in COCO format ----
imgs, anns, aid = [], [], 1
for i, f in enumerate(files):
    w, h = dims[f]; imgs.append({"id": i, "file_name": f, "width": w, "height": h})
    lp = os.path.join(TL, os.path.splitext(f)[0] + ".txt")
    if os.path.exists(lp):
        for ln in open(lp):
            p = ln.split()
            if len(p) >= 5:
                c = int(float(p[0])); x, y, bw, bh = [float(v) for v in p[1:5]]
                anns.append({"id": aid, "image_id": i, "category_id": c, "bbox": [(x - bw/2)*w, (y - bh/2)*h, bw*w, bh*h],
                             "area": bw*w*bh*h, "iscrowd": 0}); aid += 1
gt = COCO(); gt.dataset = {"images": imgs, "annotations": anns, "categories": [{"id": 0, "name": "RBC"}, {"id": 1, "name": "WBC"}, {"id": 2, "name": "Platelet"}]}
with contextlib.redirect_stdout(io.StringIO()): gt.createIndex()

def coco_eval(dets, thr=0.0):
    dets = [dict(d) for d in dets if d["score"] >= thr]
    if not dets: return dict(map50=0.0, map5095=0.0, recall50=0.0)
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(gt, gt.loadRes(dets), "bbox"); ev.evaluate(); ev.accumulate(); ev.summarize()
    r = ev.eval["recall"][0, :, 0, 2]; r = r[r > -1]
    return dict(map50=float(ev.stats[1]), map5095=float(ev.stats[0]), recall50=float(r.mean()) if len(r) else 0.0)

# ---- single-pass baseline through the SAME evaluator ----
model = YOLO(W); base_dets = []
for i, f in enumerate(files):
    r = model.predict(os.path.join(TI, f), conf=0.001, iou=0.7, verbose=False)[0]
    for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
        base_dets.append({"image_id": i, "category_id": int(c), "bbox": [float(b[0]), float(b[1]), float(b[2]-b[0]), float(b[3]-b[1])], "score": float(s)})
B = coco_eval(base_dets); B["map50_c35"] = coco_eval(base_dets, 0.35)["map50"]
print("single pass:", B)

# ---- SAHI sweep ----
sizes = sorted({max(64, int(round(fr * min(W0, H0) / 32) * 32)) for fr in (0.5, 0.65, 0.8)})
overlaps = [0.1, 0.2, 0.3]
posts = [("GREEDYNMM", "IOS", 0.5), ("NMS", "IOU", 0.5)]
dm = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=W, confidence_threshold=0.001, device="cuda:0")
def n_slices(s, o):
    st = max(1, int(s * (1 - o))); f = lambda L: 1 if L <= s else 1 + math.ceil((L - s) / st)
    return f(W0) * f(H0)

res, keep = [], {}
for s, o, (pt, pm, pth) in itertools.product(sizes, overlaps, posts):
    dets = []
    for i, f in enumerate(files):
        r = get_sliced_prediction(os.path.join(TI, f), dm, slice_height=s, slice_width=s, overlap_height_ratio=o, overlap_width_ratio=o,
                                  postprocess_type=pt, postprocess_match_metric=pm, postprocess_match_threshold=pth, verbose=0)
        for p in r.object_prediction_list:
            bb = p.bbox
            dets.append({"image_id": i, "category_id": int(p.category.id), "bbox": [bb.minx, bb.miny, bb.maxx - bb.minx, bb.maxy - bb.miny], "score": float(p.score.value)})
    m = coco_eval(dets); m["map50_c35"] = coco_eval(dets, 0.35)["map50"]
    m.update(slice=s, overlap=o, post=pt, n_slices=n_slices(s, o)); res.append(m); keep[(s, o, pt)] = dets
    print(m)
json.dump({"baseline": B, "configs": res}, open(os.path.join(OUT, "sahi_sweep.json"), "w"), indent=2)

# ---- visual evidence: 3 images where SAHI adds the most boxes (conf>=0.25) ----
cfgk = (sizes[len(sizes)//2 - 1 if len(sizes) > 1 else 0], 0.2, "GREEDYNMM"); sd = keep[cfgk]
cnt = lambda D, i: sum(1 for d in D if d["image_id"] == i and d["score"] >= 0.25)
diff = sorted(range(len(files)), key=lambda i: -(cnt(sd, i) - cnt(base_dets, i)))[:3]
os.makedirs(os.path.join(OUT, "sahi_vis"), exist_ok=True)
for i in diff:
    pan = []
    for D, ttl in [(base_dets, "single pass"), (sd, f"SAHI {cfgk[0]}px")]:
        im = cv2.imread(os.path.join(TI, files[i])).copy()
        for d in D:
            if d["image_id"] == i and d["score"] >= 0.25:
                x, y, w, h = map(int, d["bbox"]); cv2.rectangle(im, (x, y), (x+w, y+h), (0, 0, 255), 1)
        st = max(1, int(cfgk[0] * 0.8))
        for k in range(st, im.shape[1], st): cv2.line(im, (k, 0), (k, im.shape[0]), (255, 200, 0), 1)
        for k in range(st, im.shape[0], st): cv2.line(im, (0, k), (im.shape[1], k), (255, 200, 0), 1)
        cv2.putText(im, ttl, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2); pan.append(im)
    cv2.imwrite(os.path.join(OUT, "sahi_vis", f"cmp_{files[i][:30]}.jpg"), np.hstack(pan))
print("visuals in", os.path.join(OUT, "sahi_vis"))
```

Read the sweep like this:
- If `n_slices` is 1 everywhere, the earlier SAHI run did nothing and the "edge truncation" explanation was unsupported.
- If the single-pass `map50_c35` is close to the old 84%, the old "paradox" was a confidence-threshold artifact.
- Variant A or B in E9 is chosen from this table.

### C7: Error analysis (FP/FN types, occlusion groups, failure cases)

```python
import os, json, cv2
import numpy as np
from collections import Counter, defaultdict
from ultralytics import YOLO

ROOT = os.getcwd(); OUT = os.path.join(ROOT, "3_models", "revision")
W = os.path.join(OUT, "runs", "hv_main_s0", "weights", "best.pt")
TI = os.path.join(ROOT, "2_preprocessed_datasets", "bccd_grouped", "images", "test")
TL = os.path.join(ROOT, "2_preprocessed_datasets", "bccd_grouped", "labels", "test")
CONF_T, IOU_T = 0.25, 0.5; NAMES = {0: "RBC", 1: "WBC", 2: "Platelet"}

def iou_m(a, b):
    if len(a) == 0 or len(b) == 0: return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)

model = YOLO(W); files = sorted(f for f in os.listdir(TI) if f.lower().endswith((".jpg", ".jpeg", ".png")))
tp, fp, fn = Counter(), Counter(), Counter(); fp_t, fn_t = defaultdict(Counter), defaultdict(Counter)
grp = {g: [0, 0] for g in ["isolated", "partial", "heavy"]}     # [matched, total] RBC GT by overlap
per_img = []; cache = []
for f in files:
    img = cv2.imread(os.path.join(TI, f)); h, w = img.shape[:2]; gt = []
    lp = os.path.join(TL, os.path.splitext(f)[0] + ".txt")
    for ln in (open(lp) if os.path.exists(lp) else []):
        p = ln.split()
        if len(p) >= 5:
            c = int(float(p[0])); x, y, bw, bh = [float(v) for v in p[1:5]]
            gt.append([c, (x - bw/2)*w, (y - bh/2)*h, (x + bw/2)*w, (y + bh/2)*h])
    gt = np.array(gt) if gt else np.zeros((0, 5)); gb, gc = gt[:, 1:5], gt[:, 0].astype(int)
    r = model.predict(os.path.join(TI, f), conf=CONF_T, iou=0.7, verbose=False)[0]
    pb = r.boxes.xyxy.cpu().numpy(); pc = r.boxes.cls.cpu().numpy().astype(int); ps = r.boxes.conf.cpu().numpy()
    o = np.argsort(-ps); pb, pc, ps = pb[o], pc[o], ps[o]
    M = iou_m(pb, gb); gm = np.zeros(len(gt), bool); pt = np.zeros(len(pb), bool)
    for i in range(len(pb)):
        cand = [j for j in range(len(gt)) if gc[j] == pc[i] and not gm[j] and M[i, j] >= IOU_T]
        if cand: j = max(cand, key=lambda j: M[i, j]); gm[j] = True; pt[i] = True
    for i in range(len(pb)):
        c = NAMES[pc[i]]
        if pt[i]: tp[c] += 1; continue
        fp[c] += 1
        same = [M[i, j] for j in range(len(gt)) if gc[j] == pc[i]]; diff = [M[i, j] for j in range(len(gt)) if gc[j] != pc[i]]
        if same and max(same) >= IOU_T: t = "duplicate"
        elif diff and max(diff) >= IOU_T: t = "class confusion"
        elif same and max(same) >= 0.1: t = "poor localization"
        else: t = "background"
        fp_t[c][t] += 1
    ridx = [j for j in range(len(gt)) if gc[j] == 0]; G = iou_m(gb[ridx], gb[ridx]) if ridx else np.zeros((0, 0))
    if len(ridx): np.fill_diagonal(G, 0)
    for j in range(len(gt)):
        c = NAMES[gc[j]]
        if gm[j]: pass
        else:
            fn[c] += 1; col = M[:, j] if len(pb) else np.array([])
            if len(col) and any(col[i] >= IOU_T and pc[i] != gc[j] for i in range(len(pb))): t = "misclassified"
            elif len(col) and col.max() >= 0.1: t = "poor localization"
            else: t = "missed"
            fn_t[c][t] += 1
        if gc[j] == 0:
            ov = G[ridx.index(j)].max() if len(ridx) > 1 else 0
            g = "isolated" if ov < 0.1 else "partial" if ov < 0.3 else "heavy"
            grp[g][1] += 1; grp[g][0] += int(gm[j])
    nr = int((gc == 0).sum()); frp = int(sum(1 for i in range(len(pb)) if not pt[i] and pc[i] == 0)); fnr = int(sum(1 for j in range(len(gt)) if not gm[j] and gc[j] == 0))
    per_img.append(dict(f=f, n_rbc=nr, fp=frp, fn=fnr)); cache.append((f, gb, gc, gm, pb, pc, pt))

res = {"conf": CONF_T, "iou": IOU_T, "per_class": {}, "rbc_recall_by_overlap": {}, "rbc_by_density": {}}
for c in NAMES.values():
    P = tp[c] / max(1, tp[c] + fp[c]); R = tp[c] / max(1, tp[c] + fn[c])
    res["per_class"][c] = dict(TP=tp[c], FP=fp[c], FN=fn[c], P=P, R=R, fp_types=dict(fp_t[c]), fn_types=dict(fn_t[c]))
    print(f"{c:9s} TP={tp[c]} FP={fp[c]} FN={fn[c]}  P={P:.3f} R={R:.3f}\n   FP types {dict(fp_t[c])}\n   FN types {dict(fn_t[c])}")
for g, (m, t) in grp.items():
    res["rbc_recall_by_overlap"][g] = dict(matched=m, total=t, recall=m / max(1, t)); print(f"RBC recall, {g}: {m}/{t} = {m / max(1, t):.3f}")
q = np.quantile([d["n_rbc"] for d in per_img], [1/3, 2/3])
for name, lo, hi in [("sparse", -1, q[0]), ("medium", q[0], q[1]), ("dense", q[1], 1e9)]:
    s = [d for d in per_img if lo < d["n_rbc"] <= hi]; t = sum(d["n_rbc"] for d in s); f_ = sum(d["fn"] for d in s); p_ = sum(d["fp"] for d in s)
    res["rbc_by_density"][name] = dict(images=len(s), rbc=t, FN=f_, FP=p_, recall=1 - f_ / max(1, t)); print(f"density {name}: {len(s)} imgs, RBC={t}, FN={f_}, FP={p_}")
json.dump(res, open(os.path.join(OUT, "error_analysis.json"), "w"), indent=2)

# failure cases: top-6 images by RBC FP+FN. green=TP, red=FP, yellow=missed GT
os.makedirs(os.path.join(OUT, "failure_cases"), exist_ok=True)
for k in sorted(range(len(per_img)), key=lambda k: -(per_img[k]["fp"] + per_img[k]["fn"]))[:6]:
    f, gb, gc, gm, pb, pc, pt = cache[k]; im = cv2.imread(os.path.join(TI, f))
    for j in range(len(gb)):
        if not gm[j]: cv2.rectangle(im, tuple(map(int, gb[j][:2])), tuple(map(int, gb[j][2:])), (0, 255, 255), 2)
    for i in range(len(pb)):
        cv2.rectangle(im, tuple(map(int, pb[i][:2])), tuple(map(int, pb[i][2:])), (0, 200, 0) if pt[i] else (0, 0, 255), 1)
    cv2.imwrite(os.path.join(OUT, "failure_cases", f"fail_{k}_{f[:25]}.jpg"), im)
print("saved failure cases + error_analysis.json")
```

The overlap groups use RBC GT-box IoU: isolated < 0.1, partial 0.1–0.3, heavy ≥ 0.3. Define this in E14b.

### C8: Collect figures with the right names

```python
import os, shutil
ROOT = os.getcwd(); OUT = os.path.join(ROOT, "3_models", "revision"); FIG = os.path.join(OUT, "paper_figs"); os.makedirs(FIG, exist_ok=True)
ev = os.path.join(OUT, "eval", "hv_main_s0_internal_std"); rn = os.path.join(OUT, "runs", "hv_main_s0")
pairs = [(os.path.join(ev, "BoxPR_curve.png"), "internal_test_BoxPR_curve.png"),
         (os.path.join(ev, "confusion_matrix_normalized.png"), "confusion_matrix_normalized.png"),
         (os.path.join(ev, "BoxF1_curve.png"), "internal_test_BoxF1_curve.png"),
         (os.path.join(rn, "results.png"), "training_loss_results.png")]
for s, d in pairs:
    if os.path.exists(s): shutil.copy(s, os.path.join(FIG, d)); print("ok  ", d)
    else: print("MISSING", s)
print("Also copy the 'failure_cases' and 'sahi_vis' folders and re-render localized_bb.jpg from the NEW weights (predict on the grouped test set).")
```

### C9: Patch the manuscript (typos + token fill + leftover scan)

```python
import os, re, json, glob, numpy as np
ROOT = os.getcwd(); OUT = os.path.join(ROOT, "3_models", "revision"); RES = os.path.join(OUT, "results")
PAPER = "paper.md"                       # your LaTeX source
runs = {}
for f in glob.glob(os.path.join(RES, "*.json")):
    d = json.load(open(f)); runs.setdefault(d["tag"], []).append(d)
def vals(t, ev, fn):
    o = []
    for r in runs.get(t, []):
        try:
            if r.get(ev): o.append(fn(r[ev]))
        except KeyError: pass
    return o
def ms(xs, dp=1):
    if not xs: return None
    a = np.array(xs) * 100
    return f"{a[0]:.{dp}f}" if len(a) == 1 else f"{a.mean():.{dp}f} $\\pm$ {a.std(ddof=1):.{dp}f}"
tok = {}
def put(k, v):
    if v is not None: tok["{{%s}}" % k] = str(v)
for k, t, ev, f in [("HV_MAP50", "hv_main", "internal_std", "map50"), ("HV_MAP5095", "hv_main", "internal_std", "map5095"),
                    ("HV_TTA_MAP50", "hv_main", "internal_tta", "map50"), ("HV_TTA_MAP5095", "hv_main", "internal_tta", "map5095"),
                    ("RT640_MAP50", "rtdetr_640", "internal_std", "map50"), ("RT800_MAP50", "rtdetr_800", "internal_std", "map50"),
                    ("RT800_MAP5095", "rtdetr_800", "internal_std", "map5095")]:
    put(k, ms(vals(t, ev, lambda e, f=f: e[f])))
put("EXT_MAP50", ms(vals("hv_main", "external_std", lambda e: e["per_class"]["WBC"]["ap50"])))
put("EXT_P", ms(vals("hv_main", "external_std", lambda e: e["per_class"]["WBC"]["p"])))
put("EXT_R", ms(vals("hv_main", "external_std", lambda e: e["per_class"]["WBC"]["r"])))
mf = os.path.join(OUT, "split_manifest.json")
if os.path.exists(mf):
    M = json.load(open(mf))
    for k, sp in [("N_TRAIN", "train"), ("N_VAL", "val"), ("N_TEST", "test")]: put(k, M["counts"][sp]["images"])
    put("N_EXT", M["external"]["kept"]); put("N_EXT_REMOVED", M["external"]["removed_bccd_named"]); put("IMG_W", M["img_w"]); put("IMG_H", M["img_h"])
ef = os.path.join(OUT, "efficiency.json")
if os.path.exists(ef):
    E = {(r["tag"], r["imgsz"]): r for r in json.load(open(ef))}
    for k, t in [("HV", "hv_main"), ("RT", "rtdetr_800")]:
        r = E.get((t, 800))
        if r: put(f"{k}_PARAMS", f"{r['params_M']:.1f}"); put(f"{k}_GFLOPS", f"{r['gflops']:.1f}"); put(f"{k}_MS", f"{r['ms_fp32']:.1f}")
sj = os.path.join(OUT, "sahi_sweep.json")
if os.path.exists(sj):
    S = json.load(open(sj)); best = max(S["configs"], key=lambda r: r["map50"])
    put("SAHI_BASE_MAP50", f"{S['baseline']['map50']*100:.1f}"); put("SAHI_BEST_MAP50", f"{best['map50']*100:.1f}")
    put("SAHI_BEST_CFG", f"{best['slice']} px, overlap {best['overlap']}, {best['post']}"); put("SAHI_N_CFG", len(S["configs"]))

TYPOS = {"analtsis": "analysis", "approcahes": "approaches", "approches": "approaches", "boradly": "broadly", "detectiob": "detection",
 "bsased": "based", "myltiple cell": "multiple cells", "laacks": "lacks", "standalized": "standardized", "maP,": "mAP,", "comparision": "comparison",
 "dificult": "difficult", "revemt": "recent", "BCRT=DETR": "BCRT-DETR", "RT=DETR": "RT-DETR", "transdormer": "transformer", ", ut lacks": ", but lacks",
 "acrross": "across", "maintaning": "maintaining", "vealuated": "evaluated", "generalibility": "generalizability", "capabiloyt": "capability",
 "realtwd": "related", "performacne": "performance", "comdition": "condition", "approcahc": "approach", "generatlized": "generalized",
 "convilotional": "convolutional", "redudancy": "redundancy", "efficency": "efficiency", "Similary": "Similarly", "un detection": "in detection",
 "peripehral": "peripheral", "reply on": "rely on", "generaziliblty": "generalizability", "99.79%.these": "99.79%. These", "mobilenetv2": "MobileNetV2",
 "transform models": "transformer models", "bottlenext": "bottleneck", "highresolution": "high-resolution", "close-perfect": "near-perfect",
 "furthers confirms": "further confirms", "specially for": "especially for", "accross": "across", "microscoopy": "microscopy", "successfulyy": "successfully",
 "Precisiom": "Precision", "standand": "standard", "forcasted": "forecasted", "false-postive": "false-positive"}
txt = open(PAPER, encoding="utf-8").read(); n_typo = 0
for a, b in TYPOS.items():
    c = txt.count(a); n_typo += c; txt = txt.replace(a, b)
print("typo replacements:", n_typo)
for k, v in tok.items(): txt = txt.replace(k, v)
open("paper_fixed.tex", "w", encoding="utf-8").write(txt)
print("unfilled tokens:", sorted(set(re.findall(r"\{\{[A-Z0-9_]+\}\}", txt))))
for w in ["clinical", "clinic", "diagnos", "validation", "perfect", "paradox", "ablation", "C2f", "PANet", "1920", "TP+TN", "robust", "guarantee", "completely"]:
    for m in re.finditer(w, txt, flags=re.I):
        print(f"[{w}] ...{txt[max(0, m.start()-50):m.end()+50].replace(chr(10), ' ')}...")
```

Then manually check the author name behind `chen2025bcrt`. The text says "X.Cheng" but the key says Chen.

---

## 4. Edits to `paper.md` (LaTeX source)

Each edit gives the section, the current text and the replacement. `{{TOKEN}}` markers are filled by C9. Decision-dependent sentences are marked **[IF …]**.

### E1: Abstract (replace everything between `\begin{abstract}` and `\end{abstract}`)

Current:
```
The peripheral blood smear (PBS) analysis plays a crucial role in diagnosing hematological disorders, but its manual interpretation is both tedious and suffers from interobserver variations, whereas deep learning techniques have shown great promise, however, despite their excellent performance on benchmarked clean datasets with isolated cells such as the RT DETR model, CNN and transformer-based networks demonstrate poor clinical applicability, where transformers experience high false positives with RBC precision as low as 57.1% based on confusion matrices and low classification accuracy, to solve this problem, we introduce HemaVisionNet with a YOLOv11 architecture that detects and classifies Red Blood Cells, White Blood Cells, and Platelets in crowded PBS samples with a dual-training approach using multiple pretrained data sets and fine-tuning with a high-resolution image size of 800 by 800 pixels, along with optimized data augmentation and higher bounding box loss, we discover the SAHI Paradox that slicing inferences decrease recall to 65.2%, whereas our model obtains an mAP of 0.5 for BCCD at 92.93% and 84.99% on out-of-distribution images, thereby emphasizing its novelty and significance.
```
Replace with:
```latex
Counting red cells, white cells and platelets in peripheral blood smears is slow by hand and varies between observers, and automatic detectors struggle when cells touch or overlap. We study a training and inference recipe built on the unmodified YOLOv11n detector: two-stage training (pretraining on TXL-PBC and CBC, then fine-tuning on BCCD), 800 px input, a box-loss weight of 10, and rotation and flip augmentation without MixUp. On a BCCD test split grouped by source image, single-pass inference reaches {{HV_MAP50}}\% mAP@0.5 and {{HV_MAP5095}}\% mAP@0.5:0.95 (mean $\pm$ std over three seeds). On a second Kaggle dataset, white-blood-cell detection reaches {{EXT_MAP50}}\% mAP@0.5. Under the same protocol RT-DETR-L reaches {{RT800_MAP50}}\% mAP@0.5 with {{RT_PARAMS}} M parameters against {{HV_PARAMS}} M for YOLOv11n. We ablate each component, test slicing-aided inference (SAHI) over {{SAHI_N_CFG}} settings, and analyse errors on overlapped RBCs. The work covers detection of three cell types only; it makes no diagnostic claim.
```
**[IF RT-DETR ≥ HV or within 1 std]** keep the RT-DETR sentence exactly as written. **[IF HV clearly better]** you may add "higher" there. **[IF SAHI baseline ≈ SAHI best]** add: "SAHI gave no gain over single-pass inference."

### E2: Keywords

Current: `... High-resolution microscopy, Cross-institutional validation. \end{IEEEkeywords}`
Replace: `... High-resolution microscopy, Cross-dataset evaluation. \end{IEEEkeywords}`

### E3: Introduction, last paragraph and contributions

Current:
```
To solve these issues, this study proposes a high-resolution CNN-based framework named HemaVision-Net, which is designed for RBC, WBC and platelet detection in complex blood smear images. The proposed framework emphasizes localized feature learning to ensure more accurate detection in dense cellular environments. In addition, a two-stage training strategy is used to improve feature generalization and robustness. In addition, the limitations of transformer-based approaches and slicing-based inference methods in hematological image analysis are also analyzed in this study.
```
Replace:
```latex
To address these issues, this study evaluates a training and inference recipe, which we call HemaVision-Net, for detecting RBCs, WBCs and platelets in crowded blood smear images. The detector is an unmodified YOLOv11n; what we change is how it is trained (two-stage pretraining and fine-tuning, 800 px input, a larger box-loss weight, rotation and flip augmentation) and how it is evaluated. We compare it with RT-DETR-L and other YOLO variants under one protocol, ablate each component, and test slicing-aided inference (SAHI) across slice sizes, overlaps and merging rules.
```
Contributions: replace the whole `\begin{itemize} ... \end{itemize}` that follows "The main contributions of this research are as follows:" with:
```latex
\begin{itemize}
\item A training and inference recipe for crowded blood smears on an unmodified YOLOv11n, with every changed setting listed (Table~\ref{tab:hyperparams}).
\item A controlled comparison with RT-DETR-L and three other YOLO models on a source-grouped split, reporting mAP@0.5, mAP@0.5:0.95, parameters, FLOPs and latency.
\item A one-factor-at-a-time ablation of resolution, box-loss weight, MixUp, two-stage pretraining and test-time augmentation.
\item An error analysis of RBC detection by degree of cell overlap, and a SAHI sweep over slice size, overlap and merging rule against a single-pass baseline.
\item A cross-dataset check on an independent Kaggle WBC set (WBC class only).
\end{itemize}
```
Also edit the sentence "A comparative discussion of CNN and transform models" if it survives elsewhere (C9 fixes the typo).

### E4: Methodology opening

Current:
```
Since standard global context architecture struggle with extreme cell density, we propose HemaVision-Net : a customized two stage Convolutional Neural Network (CNN) optimized for high resolution domain adaption and penalized bounding box regression.Furthermore, we conduct an architectural ablation study comparing Vision Transformer(RT-DETR) against our CNN, analyzing the failure points of Slicing Aided Hyper Inference(SAHI) Fig.~\ref{fig:workflow} illustrated the complete methodology workflow. illustrates the complete methodology workflow.
```
Replace:
```latex
We use YOLOv11n as an unmodified detector and study how a two-stage training schedule, higher input resolution and a larger box-loss weight affect detection in crowded smears. We compare it with RT-DETR-L under the same recipe and test slicing-aided inference (SAHI). Fig.~\ref{fig:workflow} shows the workflow.
```

### E5: Dataset subsection (fine-tuning and external)

Current:
```
For domain adaption, we used the BCCD dataset, featuring severe RBC occlusion mimicking real world smears. A 10% was isolated for internal testing. Finally , the WBC Object Detection Dataset (Kaggle) was kept completely unseen during training which serves as our benchmark for cross-institutional robustness.
```
Replace:
```latex
For fine-tuning we used the BCCD dataset (Roboflow export, {{IMG_W}}$\times${{IMG_H}} px), which contains many touching and overlapping RBCs. The export contains augmented copies of the same source image, so we grouped all images by source image and split the groups 80/10/10 into train, validation and test ({{N_TRAIN}}/{{N_VAL}}/{{N_TEST}} images; seed 42), which keeps copies of one image inside one split. BCCD carries no patient identifiers, so the split cannot be made patient-level. Table~\ref{tab:data} lists image and cell counts. The Kaggle WBC detection dataset was never used for training or model selection. It contains only WBC labels; we removed {{N_EXT_REMOVED}} images with BCCD-style file names, leaving {{N_EXT}} images.
```
Add a data table (counts from `split_manifest.json`, printed by C2) with columns split, images, source images, RBC, WBC, platelets, `\label{tab:data}`. Include rows for the pretraining set (1182 train / 312 val images, from your logs) and the external set.

### E6: Architecture subsection

Current:
```
Vision transformer applied universal self attention, our study with RT-DETR illustrated universal context which becomes a bottlenext in compactly packed smears and it significantly dropped precision. to solve this issue, we designed HemaVision-Net in native YOLOv11 architecture. Fig.~\ref{fig:architecture} presents the internal architecture of the HemaVision-Net framework By utilizing C2f modules and a Path Aggregation Network (PANet) neck, our CNN leverages strictly localized receptive field to distinctly separate touching RBC boundaries without distraction from the global image context.
```
Replace:
```latex
HemaVision-Net uses YOLOv11n as released (2.58 M parameters, 6.3 GFLOPs): a convolutional backbone with C3k2 blocks, SPPF and a C2PSA self-attention block, a PAN-style neck, and a decoupled detection head over three scales. We changed no module. Fig.~\ref{fig:architecture} shows the network. We expected the convolutional design to handle touching RBCs well; this is a hypothesis, and Section~\ref{sec:cmp} tests it against RT-DETR-L trained with the same recipe.
```
Caption (`fig:architecture`): replace with "YOLOv11n as used in HemaVision-Net (no structural modification; the contribution is the training recipe)." Make sure Fig. 2 itself does not show C2f or custom heads; redraw it if it does. Add `\label{sec:cmp}` to the comparison subsection.

### E7: Two-stage training subsection and Table I

Current:
```
Standard detectors train at $640 \times 640$ pixels. Compressing a $1920 \times 1080$ smear merges adjacent RBCs into distinguishable blobs, forcing the model to approximate boundaries. To solve this , we implemented a two stage strategy.
```
Replace:
```latex
Standard detectors train at $640 \times 640$ pixels. Downscaling can merge touching RBCs, so we also fine-tune at 800 px. BCCD images are {{IMG_W}}$\times${{IMG_H}} px, so 800 px involves upsampling; whether it helps is tested in Table~\ref{tab:ablation}. We use a two-stage schedule.
```
Current:
```
Crucially, MixUp augmentation was disabled ($\alpha = 0.0$) because, while generally beneficial, it produces biologically implausible overlapping artifacts (``ghost cells'') in hematology.
```
Replace:
```latex
MixUp stays at $\alpha = 0$ (the Ultralytics default); we expect it to create unrealistic overlaps and test $\alpha = 0.1$ in the ablation.
```
Table I: replace the whole `\begin{table} ... \label{tab:hyperparams} \end{table}` block with:
```latex
\begin{table}[htbp]
\caption{Settings changed from the Ultralytics defaults}
\centering\footnotesize
\begin{tabular}{|l|c|c|}
\hline
\textbf{Setting} & \textbf{Default} & \textbf{This work} \\ \hline
Fine-tuning input size (px) & 640 & 800 \\ \hline
Box-loss gain $\lambda_{box}$ & 7.5 & 10.0 \\ \hline
Rotation (degrees) & 0 & 90 \\ \hline
Vertical flip probability & 0 & 0.5 \\ \hline
\end{tabular}
\label{tab:hyperparams}
\end{table}
```
Add after the table: "Unchanged: mosaic 1.0, MixUp 0, $\lambda_{cls}=0.5$, $\lambda_{dfl}=1.5$. The optimizer was set to \texttt{auto}, which selected AdamW with lr 0.001429 (the requested \texttt{lr0} is ignored in this mode)."

### E8: Loss function subsection

Current:
```
To enforce strict boundary adherence, the total loss ($L_{\text{total}}$) combines classification loss ($L_{\text{cls}}$), Distribution Focal Loss (L_{\text{DFL} ) and Complete Intersection over Union (L_{\text{DFL} ). Eq.~\ref{eq:ciou} formulates this relationship:
```
Replace:
```latex
The total loss combines a box term based on Complete IoU ($L_{\text{CIoU}}$), a binary cross-entropy classification term ($L_{\text{cls}}$) and the Distribution Focal Loss ($L_{\text{DFL}}$). The CIoU term is defined in Eq.~\ref{eq:ciou}:
```
Keep the `eq:ciou` equation. Directly after it and before "By raising ...", insert:
```latex
\begin{equation}
L_{\text{total}} = \lambda_{box} L_{\text{CIoU}} + \lambda_{cls} L_{\text{cls}} + \lambda_{dfl} L_{\text{DFL}}, \quad \lambda_{box}=10,\ \lambda_{cls}=0.5,\ \lambda_{dfl}=1.5
\label{eq:total}
\end{equation}
```
Current:
```
By raising the box loss multiplier ($\lambda_{box}$) to 10.0, the CIoU penalty forces ranking of the Euclidean distance ($\rho$) between anticipated and reality centers.This efficiently dintincts merged RBC clusters, updating placement accuracy.
```
Replace:
```latex
A larger $\lambda_{box}$ weights box regression more heavily against classification, which we expected to help separate touching RBCs. We tested this with $\lambda_{box}\in\{7.5,10,12.5\}$ (Table~\ref{tab:ablation}) and report mAP@0.75 and RBC AP@0.75, because a tighter-box effect should show at the stricter IoU.
```
Keep the sentence about Fig.~\ref{fig:loss_curve}, but change "which signifies this improving process" to "which shows the optimization."

### E9: SAHI subsection (retitle and rewrite)

Title: `\subsection{Architectural Limits: Exposing the ``SAHI Paradox''}` becomes `\subsection{Slicing-Aided Inference (SAHI)}`.

Replace everything from "Slicing-Aided Hyper Inference (SAHI) divides..." through the end of the `\end{itemize}` plus the paragraph "Consequently, ... varying cellular densities." with the text below. Pick **Variant A or B** from the C6 table.

Variant A (SAHI still worse under the fair evaluator):
```latex
Slicing-Aided Hyper Inference (SAHI) splits an image into overlapping tiles, runs the detector on each and merges the results. We tested {{SAHI_N_CFG}} settings (three slice sizes, three overlaps, greedy NMM and NMS merging) with the same model, the same pycocotools evaluator and a confidence threshold of 0.001. Single-pass inference reached {{SAHI_BASE_MAP50}}\% mAP@0.5; the best SAHI setting ({{SAHI_BEST_CFG}}) reached {{SAHI_BEST_MAP50}}\% (Table~\ref{tab:sahi}). Fig.~\ref{fig:sahi} shows images where tiling adds boxes along tile borders. We did not separate the causes. Truncated cells at tile borders and merge errors are plausible explanations, but we have not tested them directly, and the result applies to this model, this dataset and these settings only. HemaVision-Net therefore uses single-pass inference.
```
Variant B (gap vanishes once conf and evaluator are matched): say so plainly ("the drop we first observed came from a 0.35 confidence cut-off and a different evaluator") and delete the term "paradox" everywhere (title, Intro, Discussion, Conclusion).

Fig.~\ref{fig:predictions} caption currently ends "...using using HemaVision-Net without". Replace the ending with "using HemaVision-Net with single-pass inference." Delete the sentence "...establishes a robust framework for real-world clinical applicability across varying cellular densities."

### E10: Evaluation metrics

Current: `\text{Recall}=\frac{TP}{TP+TN}`
Replace: `\text{Recall}=\frac{TP}{TP+FN}`

Current:
```
The achievement of the suggested model was analyzed using standand COCO metrics, including Precisiom (P), Recall (R) and mean average precision (mAP).
```
Replace:
```latex
Detection quality is reported as precision (P), recall (R), mAP@0.5 and mAP@0.5:0.95 from the Ultralytics validator (v8.3.166). All models use the same call: confidence 0.001, NMS IoU 0.7 (RT-DETR is NMS-free), single pass unless marked TTA. P and R are taken at the confidence that maximizes F1. Results are mean $\pm$ std over seeds where $n>1$.
```
Current: `...at the Intersection over Union (IoU) threshold of 0.50.`
Replace: `...at an IoU threshold of 0.50; mAP@0.5:0.95 averages this over IoU thresholds from 0.50 to 0.95 in steps of 0.05.`

### E11: Internal test results

Replace the whole paragraph beginning "The proposed HemaVision-Net framework was analyzed on the separated 10% internal test split..." through "...furthers confirms this robust class performance." with:
```latex
HemaVision-Net was evaluated on the source-grouped BCCD test split ({{N_TEST}} images). Single-pass inference reached {{HV_MAP50}}\% mAP@0.5 and {{HV_MAP5095}}\% mAP@0.5:0.95; with test-time augmentation (TTA) it reached {{HV_TTA_MAP50}}\% and {{HV_TTA_MAP5095}}\% (Table~\ref{tab:main}); the single-pass figure is our headline result. Per-class results are in Table~\ref{tab:perclass}. RBCs are the hardest class; white blood cells are detected most reliably, and platelets fall in between. Fig.~\ref{fig:pr_curve} and Fig.~\ref{fig:confusion_matrix} show the precision-recall curves and the normalized confusion matrix for the same run.
```
After it add `\input{tables/tab_main}` and `\input{tables/tab_perclass}`. Regenerate Figs. 4–5 from C8. Re-read every number in the text against Table~\ref{tab:perclass} before submitting.

### E12: External evaluation subsection

Retitle: `\subsection{Cross-institutional Robustness (External Validation)}` becomes `\subsection{Cross-dataset Generalization (WBC Only)}`.

Replace the two paragraphs ("To guarantee real world clinical applicability ..." and "Without prior exposure ...") with:
```latex
We evaluated the final model, without retraining, on the Kaggle WBC detection dataset ({{N_EXT}} images after removing {{N_EXT_REMOVED}} BCCD-named images). This set is labelled for WBCs only, so it measures cross-dataset generalization of WBC detection; it says nothing about RBC or platelet detection. HemaVision-Net reached {{EXT_MAP50}}\% AP@0.5 for WBCs (precision {{EXT_P}}\%, recall {{EXT_R}}\%; Table~\ref{tab:external}), lower than on BCCD. The dataset differs in staining and imaging, but we did not verify patients, scanners or staining protocols, so this is a cross-dataset check and not a clinical or cross-institutional validation.
```
Add `\input{tables/tab_external}`. If C2 printed a non-zero near-duplicate count, add a sentence stating it. If you evaluate a second external set (via `EXTRA_EXTERNAL`), say "two external datasets" and add its row.

### E13: Ablation subsection (replace the whole `\subsection{Ablation Study}` block, including Table II)

```latex
\subsection{Controlled Comparison and Component Ablation}\label{sec:cmp}
Table~\ref{tab:main} compares HemaVision-Net with RT-DETR-L and three other YOLO models trained with the same data, schedule, augmentation and evaluation. Table~\ref{tab:ablation} changes one factor at a time relative to the full recipe. Differences smaller than the seed standard deviation should not be read as effects. Table~\ref{tab:efficiency} lists parameters, FLOPs and latency.
\subsubsection{Architecture}
Under identical settings RT-DETR-L reached {{RT800_MAP50}}\% mAP@0.5 and {{RT800_MAP5095}}\% mAP@0.5:0.95 at 800 px, against {{HV_MAP50}}\% and {{HV_MAP5095}}\% for HemaVision-Net, with {{RT_PARAMS}} M versus {{HV_PARAMS}} M parameters and {{RT_GFLOPS}} versus {{HV_GFLOPS}} GFLOPs. [WRITE ONE SENTENCE FROM THE C4 DECISION RULE: "comparable accuracy at a fraction of the compute", or "higher under this protocol".] Our first RT-DETR baseline ({{RT640_MAP50}}\% at 640 px) was trained with different augmentation and resolution, which explains most of the earlier gap.
\subsubsection{Test-time augmentation}
TTA changed mAP@0.5 from {{HV_MAP50}}\% to {{HV_TTA_MAP50}}\% (last row of Table~\ref{tab:ablation}). Unless stated, results in this paper are single pass.
\input{tables/tab_ablation}
\input{tables/tab_efficiency}
```
Delete the old paragraphs ("Architectural Comparison: Transformer vs CNN", "Impact of Test-Time Augmentation (TTA) vs Scaling", "The SAHI Paradox") and old Table II. The 1024 px RT-DETR row and the "SAHI (256)" row were never re-run under the new protocol; do not carry them over.

### E14: Discussion (and E14b: add error analysis)

Replace the whole paragraph ("For dense, repetitive structures such as overlapping RBCs ... without site-specific retraining.") with:
```latex
HemaVision-Net detects the three cell types well in crowded smears at a small model size. Whether convolutional layers are better than attention for touching RBCs remains a hypothesis: in our controlled comparison [STATE THE C4 RESULT]. The ablation (Table~\ref{tab:ablation}) shows which settings moved the numbers beyond seed noise; settings within one standard deviation should be treated as having no demonstrated effect. SAHI gave no gain in our tests (Table~\ref{tab:sahi}), and we make no claim about why. The cross-dataset result covers WBCs only and is lower than the in-dataset result, so generalization to other staining and imaging conditions is only partly shown.
```
E14b: add `\subsection{Error Analysis}` after the ablation, filled from `error_analysis.json` (C7):
```latex
At confidence 0.25 and IoU 0.5, RBC detection had {{TP}} true positives, {{FP}} false positives and {{FN}} false negatives. False positives were mostly [DUPLICATE / POOR LOCALIZATION / CLASS CONFUSION / BACKGROUND, from fp_types]; false negatives were mostly [from fn_types]. We grouped RBCs by the largest IoU with a neighbouring RBC box (isolated $<0.1$, partial 0.1--0.3, heavy $\ge0.3$): recall was {{iso}}, {{part}} and {{heavy}} respectively, and {{dense}} in the densest third of images. Fig.~\ref{fig:fail} shows representative failures (green: matched, red: false positive, yellow: missed).
```
Add the `failure_cases` images as `fig:fail`. Use "occluded" in the paper only when this overlap grouping supports it.

### E15: Limitations (replace the whole subsection)

Current: "Although this study shows solid performance, several limitation remain. ... without offering computational efficiency."
Replace:
```latex
This study has several limits. (i) The datasets are public and small, with limited staining and acquisition variation, and no pathological cell variants. (ii) BCCD has no patient identifiers, so the split is grouped by source image, not by patient. (iii) The only external set is labelled for WBCs, so RBC and platelet generalization is untested. (iv) The task is detection of three cell types; the work does not evaluate diagnosis or disease-level outcomes. (v) 800 px input raises memory and latency (Table~\ref{tab:efficiency}), and BCCD images are upsampled to reach it. (vi) Ablations use two seeds and baselines one, so small differences are not conclusive. (vii) Hyperparameters were not tuned beyond the values we report. Future work includes segmentation for finer morphology and evaluation on multi-site data.
```

### E16: Conclusion (delete from "This study presents a high-resolution convolutional framework" to "...stable performance are required" and replace)

```latex
We evaluated a training and inference recipe on an unmodified YOLOv11n for detecting RBCs, WBCs and platelets in crowded blood smears. On a BCCD test split grouped by source image it reached {{HV_MAP50}}\% mAP@0.5 and {{HV_MAP5095}}\% mAP@0.5:0.95, and {{EXT_MAP50}}\% AP@0.5 for WBCs on a second dataset. In a controlled comparison, RT-DETR-L reached {{RT800_MAP50}}\% mAP@0.5 with {{RT_PARAMS}} M parameters against {{HV_PARAMS}} M. [ONE SENTENCE FROM THE C4 RESULT.] SAHI gave no improvement in our tests. The results cover detection of three cell types on public datasets; further work on diverse multi-site data is needed before any clinical use is considered.
```

### E17: Implementation details (add as a new subsection at the start of Results)

```latex
\subsection{Implementation Details}
All models were trained with Ultralytics 8.3.166, PyTorch 2.7.1 (CUDA 11.8) and Python 3.12.3 on one NVIDIA RTX 4070 SUPER (12 GB) with a Ryzen 7 7700 CPU, with AMP, deterministic mode and seeds 0--2. Fine-tuning used 100 epochs (early-stopping patience 25), batch size 8 (4 for RT-DETR at 800 px), AdamW selected automatically (lr 0.001429, momentum 0.9, weight decay 0.0005), 3 warm-up epochs and mosaic closed for the last 10 epochs. Stage 1 used 50 epochs (patience 15), batch size 16 at 640 px on 1182 training and 312 validation images. RT-DETR-L stage 1 used 150 epochs (patience 20). The best checkpoint by validation fitness was evaluated once on the test split. The split manifest, code and configs are available at [REPOSITORY URL].
```
Upload `split_manifest.json`, the yaml files and the notebooks to a public repository and put the URL in.

### E18: Clinical-wording sweep (every place)

| Location | Current | Replace |
|---|---|---|
| Abstract | "poor clinical applicability" | removed by E1 |
| Keywords | "Cross-institutional validation" | E2 |
| §IV external | "To guarantee real world clinical applicability..." | E12 |
| §IV external | "This minimal degradation across domain shifts confirms that the framework successfully learned fundamental biological morphologies..." | removed by E12 (not supported) |
| SAHI subsection end | "...robust framework for real-world clinical applicability..." | E9 |
| Discussion | "clinically reliable predictions ... highly suitable for deployment in diverse laboratory environments without site-specific retraining" | removed by E14 |
| Discussion | "this proves that native high-resolution inference is scientifically superior" | removed by E14 |
| Conclusion | "considerable potential for real clinical use", "ensures its stability and usability in various institutions" | removed by E16 |

After editing, C9's scan must show no remaining hits for `clinical`, `diagnos`, `validation` or `guarantee` outside the Limitations sentence.

---

## 5. Known gaps this plan cannot close

- **Results may drop** after the group split. Report whatever comes out; the reviewers explicitly asked for this.
- **One external dataset, WBC only.** R1 asked for several independent datasets. The honest wording is already in E12. Add a second dataset via `EXTRA_EXTERNAL` only if you can obtain one that was not used for pretraining.
- **No patient-level split** is possible with BCCD. This is stated in E5 and E15.
- **Few seeds.** Ablations have 2 seeds, baselines 1. Language in E13 reflects this.
- **The "why" of SAHI degradation** (edge truncation, merge errors) stays an untested hypothesis unless the sweep happens to isolate it.
- **Pretraining data overlap:** TXL-PBC and CBC were not checked for overlap with BCCD or the external set. Add one sentence in Limitations if you cannot check it.

## 6. Final checklist

- [ ] C1 and C2 run; manifest saved; split described in E5
- [ ] C3 finished for all configs (`results/*.json` count matches the configs and seeds)
- [ ] C5, C6, C7 run; C4 tables written to `tables/`; C8 figures re-copied
- [ ] E1–E18 applied; C9 shows no unfilled tokens and no leftover hits
- [ ] Recall equation is TP/(TP+FN); loss equation has λ 10 / 0.5 / 1.5; no "DFL" label on CIoU
- [ ] "paradox" removed or justified by the sweep; no "clinical" / "validation" claims left
- [ ] Figures regenerated from the new run (PR curve, confusion matrix, loss curves, localized predictions)
- [ ] IEEE format: two-column, table captions above, figure captions below, abstract 150–250 words, 4–6 keywords
- [ ] Response-to-reviewers letter written from Section 1; copyright form submitted on EDAS; one author registered
````

Kichu jinis mathay rekho:
- C1-er output dekhei bujhte parbe leakage ase kina. Tarpor-e C2 chalao, ar C3 tonight-e start koro, ota-i shobcheye lamba.
- C6 (SAHI) result onujayi E9-er Variant A ba B beche nite hobe. C4-er "DECISION" line onujayi E13, E14, E16-te RT-DETR niye ki likhbe seta thik hobe.
- Plan-e eta hoyni: tomar literature review-te author name `X.Cheng` ar cite key `chen2025bcrt` mile na. C9 shesh hole seta manually check koro.

Chaile pore response-to-reviewers letter-er draft ba E-section-gulor final filled text (number boshanor por) ami banay dite pari.
