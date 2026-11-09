"""Build report/results_table.tex (per-fold and pooled out-of-fold PQ) from runs/evaluation_f*.json."""
import json, glob, os, numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
label = {"resnet34d": "ResNet34 U-Net", "effb3d": "EfficientNet-B3 U-Net", "ensemble": "Both encoders (equal weights)"}
rows = {}
for f in sorted(glob.glob(f"{ROOT}/runs/evaluation_f*.json")):
    fold = int(f.rsplit("_f", 1)[1].split(".")[0])
    for name, r in json.load(open(f)).items():
        rows.setdefault(name.rsplit("_f", 1)[0] if name != "ensemble" else "ensemble", {})[fold] = r
folds = sorted({k for v in rows.values() for k in v})
lines = [r"\begin{tabular}{l" + "c" * len(folds) + "ccc}", r"\toprule",
         "Model & " + " & ".join(f"fold {k}" for k in folds) + r" & pooled PQ & Dice & TP / FP / FN \\", r"\midrule"]
for key in ["resnet34d", "effb3d", "ensemble"]:
    if key not in rows: continue
    v = rows[key]; n = sum(r["n_sets"] for r in v.values())
    pooled = sum(r["pq"] * r["n_sets"] for r in v.values()) / n
    per_set = np.array([r["pq_se"] ** 2 * r["n_sets"] * (r["n_sets"] - 1) for r in v.values()])  # sum of squared deviations
    se = float(np.sqrt(per_set.sum() / n / (n - 1)))
    cells = " & ".join(f"{v[k]['pq']:.3f}" if k in v else "--" for k in folds)
    tp, fp, fn = (sum(r[k] for r in v.values()) for k in ("tp", "fp", "fn"))
    dice = np.mean([r["dice"] for r in v.values()])
    lines.append(f"{label[key]} & {cells} & ${pooled:.3f} \\pm {se:.3f}$ & {dice:.3f} & {tp} / {fp} / {fn} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{ROOT}/report/results_table.tex", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
json.dump({k: {str(f): {"pq": r["pq"], "thr": r["thr"], "min_area": r["min_area"]} for f, r in v.items()} for k, v in rows.items()},
          open(f"{ROOT}/report/results_summary.json", "w"), indent=1)
