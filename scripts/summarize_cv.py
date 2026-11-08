"""Summarise runs/evaluation_f*.json into one table (per fold and pooled) for the report."""
import json, glob, os, sys, numpy as np
ROOT = os.path.join(os.path.dirname(__file__), "..")
rows = {}
for f in sorted(glob.glob(f"{ROOT}/runs/evaluation_f*.json")):
    fold = int(f.rsplit("_f", 1)[1].split(".")[0])
    for name, r in json.load(open(f)).items():
        key = name.rsplit("_f", 1)[0] if name != "ensemble" else "ensemble"
        rows.setdefault(key, []).append((fold, r))
print(f"{'model':12s} {'fold':>4s} {'method':>6s} {'thr':>4s} {'Amin':>4s} {'Dice':>6s} {'PQ':>6s} {'±SE':>6s} {'TP':>5s} {'FP':>5s} {'FN':>5s}")
for key, lst in rows.items():
    for fold, r in sorted(lst):
        print(f"{key:12s} {fold:4d} {r['method']:>6s} {r['thr']:4.1f} {r['min_area']:4d} {r['dice']:6.3f} {r['pq']:6.4f} {r['pq_se']:6.4f} {r['tp']:5d} {r['fp']:5d} {r['fn']:5d}")
    n = sum(r["n_sets"] for _, r in lst); pq = sum(r["pq"] * r["n_sets"] for _, r in lst) / n
    tp, fp, fn = (sum(r[k] for _, r in lst) for k in ("tp", "fp", "fn"))
    print(f"{key:12s} {'all':>4s} {'':>6s} {'':>4s} {'':>4s} {sum(r['dice'] for _, r in lst)/len(lst):6.3f} {pq:6.4f} {'':>6s} {tp:5d} {fp:5d} {fn:5d}   ({n} annotation sets)")
