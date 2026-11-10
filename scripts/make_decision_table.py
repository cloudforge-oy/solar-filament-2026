"""Per-fold pooled out-of-fold PQ of the final pipeline: plain rule, decision layer, refiner, candidate oracle.

Writes report/decision_table.tex and report/decision_summary.json from
  runs/decision/bank_<tag>.pkl, runs/decision/scorer_<tag>.pkl, runs/decision/scorer_<tag>_oof_scores.pkl
  runs/refine/<tag>/eval.json            (python src/refine.py eval ... --json runs/refine/<tag>/eval.json)
usage: python scripts/make_decision_table.py [tag] [refiner threshold]
"""
import json, os, pickle, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from decision import _choose, pooled_parts, pq_of  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "v2s_b3"
T = float(sys.argv[2]) if len(sys.argv) > 2 else 0.45


def load(p):
    with open(f"{ROOT}/{p}", "rb") as fh:
        return pickle.load(fh)


bank = load(f"runs/decision/bank_{TAG}.pkl"); scores = load(f"runs/decision/scorer_{TAG}_oof_scores.pkl")
tau = load(f"runs/decision/scorer_{TAG}.pkl")["tau"]
with open(f"{ROOT}/runs/refine/{TAG}/eval.json") as fh:
    ref = json.load(fh)[f"t{T:.2f}"]


def base_parts(r):
    b = r["baseline"]
    return (sum(x[0] for x in b), sum(x[1] for x in b), sum(x[2] for x in b), sum(r["n_gt"]) - sum(x[1] for x in b))


rows, out = [], {}
for f in list(range(5)) + ["all"]:
    rs = bank if f == "all" else [r for r in bank if r["fold"] == f]
    b = pq_of([base_parts(r) for r in rs])
    d = pq_of([pooled_parts(r["ious"], r["n_gt"], _choose(r, scores[r["stem"]], tau)) for r in rs])
    o = pq_of([pooled_parts(r["ious"], r["n_gt"], _choose(r, (r["ious"] * (r["ious"] > 0.5)).mean(1), 0.2)) for r in rs])
    out[str(f)] = dict(baseline=b[0], decision=d[0], refined=ref[str(f)], oracle=o[0], tp=d[1], fp=d[2], fn=d[3], base_tp=b[1], base_fp=b[2])
    rows.append(f"{f} & {b[0]:.3f} & {d[0]:.3f} & \\textbf{{{ref[str(f)]:.3f}}} & {ref[str(f)] - b[0]:+.3f} & {o[0]:.3f} \\\\")
tex = "\n".join([r"\begin{tabular}{lccccc}", r"\toprule",
                 r"Fold & plain rule & decision layer & + refiner & $\Delta$ & oracle $u$ \\", r"\midrule"]
                + rows[:-1] + [r"\midrule", rows[-1], r"\bottomrule", r"\end{tabular}"])
with open(f"{ROOT}/report/decision_table.tex", "w") as fh:
    fh.write(tex + "\n")
with open(f"{ROOT}/report/decision_summary.json", "w") as fh:
    json.dump(dict(tag=TAG, tau=tau, refiner_threshold=T, refined_counts={k: ref[k] for k in ("tp", "fp", "fn", "sq")}, **out), fh, indent=1)
print(tex)
