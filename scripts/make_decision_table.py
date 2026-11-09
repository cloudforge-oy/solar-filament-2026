"""Per-fold pooled PQ of the plain rule vs the decision layer, from the candidate bank and the cross-fitted scores.

Writes report/decision_table.tex and report/decision_summary.json. Needs runs/decision/bank_effb3d.pkl and
runs/decision/oof_scores.pkl (written during `python src/decision.py fit` analysis, see README)."""
import json, os, pickle, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from decision import _choose, pooled_parts, pq_of, LEVELS  # noqa: E402

TAU = float(sys.argv[1]) if len(sys.argv) > 1 else 0.26
with open(f"{ROOT}/runs/decision/bank_effb3d.pkl", "rb") as fh:
    bank = pickle.load(fh)
with open(f"{ROOT}/runs/decision/oof_scores.pkl", "rb") as fh:
    scores = pickle.load(fh)


def base_parts(r):
    b = r["baseline"]
    return (sum(x[0] for x in b), sum(x[1] for x in b), sum(x[2] for x in b), sum(r["n_gt"]) - sum(x[1] for x in b))


def fixed_parts(r, k, A):
    nd = r["nodes"]
    ch = np.nonzero((nd.level_idx.to_numpy() == k) & (nd.area.to_numpy() >= A))[0] if len(nd) else []
    return pooled_parts(r["ious"], r["n_gt"], ch)


rows, out = [], {}
k13 = int(np.argmin(np.abs(LEVELS - 0.65)))
for f in list(range(5)) + ["all"]:
    rs = bank if f == "all" else [r for r in bank if r["fold"] == f]
    b = pq_of([base_parts(r) for r in rs])
    fx = pq_of([fixed_parts(r, k13, 400) for r in rs])
    d = pq_of([pooled_parts(r["ious"], r["n_gt"], _choose(r, scores[r["stem"]], TAU)) for r in rs])
    o = pq_of([pooled_parts(r["ious"], r["n_gt"], _choose(r, (r["ious"] * (r["ious"] > 0.5)).mean(1), 0.2)) for r in rs])
    out[str(f)] = dict(baseline=b[0], fixed=fx[0], decision=d[0], oracle=o[0], tp=d[1], fp=d[2], fn=d[3], base_fp=b[2], base_tp=b[1])
    rows.append(f"{'all' if f == 'all' else f} & {b[0]:.3f} & {fx[0]:.3f} & \\textbf{{{d[0]:.3f}}} & {d[0] - b[0]:+.3f} & {o[0]:.3f} \\\\")
tex = "\n".join([r"\begin{tabular}{lccccc}", r"\toprule",
                 r"Fold & CC rule & best tree level & decision layer & $\Delta$ & oracle $u$ \\", r"\midrule"]
                + rows[:-1] + [r"\midrule", rows[-1], r"\bottomrule", r"\end{tabular}"])
open(f"{ROOT}/report/decision_table.tex", "w").write(tex + "\n")
json.dump(dict(tau=TAU, **out), open(f"{ROOT}/report/decision_summary.json", "w"), indent=1)
print(tex)
