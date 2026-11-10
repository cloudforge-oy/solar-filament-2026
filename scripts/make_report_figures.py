"""Report figures for the final pipeline.

  report/figures/sweep.pdf     pooled out-of-fold PQ of the decision layer against the selection threshold tau
  report/figures/examples.pdf  fold-0 validation images at the 25/50/75th percentile of PQ: refined instances

Needs runs/decision/scorer_<tag>.pkl, runs/refine/<tag>/{seeds.pkl,p/} and the fold-0 refiner (see README,
"Train from scratch"). GPU required for the examples.
usage: python scripts/make_report_figures.py [tag] [refiner threshold]
"""
import os, pickle, sys
import cv2
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from common import RAW, load_image, normalize  # noqa: E402
from refine import _dec, _parts, _iou_records, load_refiners, refine_probs, assemble  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "v2s_b3"
T = float(sys.argv[2]) if len(sys.argv) > 2 else 0.45
OUT = os.path.join(ROOT, "report", "figures"); os.makedirs(OUT, exist_ok=True)


def load(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------- tau curve
fit = load(f"runs/decision/scorer_{TAG}.pkl")
oof = pd.DataFrame(fit["oof"])
fig, ax = plt.subplots(figsize=(3.4, 2.3))
ax.plot(oof.tau, oof.pq, "o-", ms=3, color="#2a78d6", label="decision layer (stage 1)")
ax.axhline(fit["baseline"], color="#eb6834", ls="--", lw=1.2, label="plain rule (p>0.6, A$\\geq$400)")
ax.axhline(fit["best_fixed"][0], color="#8a8984", ls=":", lw=1.2, label="best single tree level")
ax.axvline(fit["tau"], color="#52514e", lw=0.8)
ax.set_xlabel("selection threshold $\\tau$", fontsize=8); ax.set_ylabel("pooled OOF PQ", fontsize=8)
ax.tick_params(labelsize=7); ax.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
ax.grid(alpha=0.3)
fig.savefig(f"{OUT}/sweep.pdf", bbox_inches="tight"); plt.close(fig)
print("tau curve saved")

# ---------------------------------------------------------------- examples (fold 0, fold-0 refiner)
seeds = [r for r in load(f"runs/refine/{TAG}/seeds.pkl")["images"] if r["fold"] == 0 and r["sel"]]
models = load_refiners([os.path.join(ROOT, "weights", "refiner_f0.pt")])


def refined(r):
    img = load_image(f"{RAW}/train/train_images/{r['stem']}.jpeg")
    p = cv2.imread(os.path.join(ROOT, "runs", "refine", TAG, "p", f"{r['stem']}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
    masks = [_dec(s["rle"]) for s in r["sel"]]
    return img, assemble(masks, refine_probs(models, normalize(img, r["stats"]), p, masks), T)


pq_img = {}
for r in seeds:
    _, masks = refined(r)
    s, tp, fp, fn = _parts(_iou_records(masks, r["gt_paths"]), r["n_gt"])
    pq_img[r["stem"]] = s / max(tp + 0.5 * fp + 0.5 * fn, 1e-9)
order = sorted(seeds, key=lambda r: pq_img[r["stem"]])
picks = [order[len(order) // 4], order[len(order) // 2], order[3 * len(order) // 4]]
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
for row, r in enumerate(picks):
    s = r["stem"]; img, masks = refined(r)
    inst = np.zeros((2048, 2048), np.int32)
    for k, m in enumerate(masks, 1):
        inst[m] = k
    g = cv2.imread(r["gt_paths"][0], cv2.IMREAD_UNCHANGED).astype(np.int32)
    ng, npred = g.max(), inst.max()
    inter = np.zeros((ng + 1, npred + 1), np.int64); np.add.at(inter, (g.ravel(), inst.ravel()), 1)
    ag = np.bincount(g.ravel(), minlength=ng + 1)[:, None]; ap = np.bincount(inst.ravel(), minlength=npred + 1)[None, :]
    iou = inter / np.maximum(ag + ap - inter, 1); iou[0, :] = 0; iou[:, 0] = 0
    tp_pred = set(np.nonzero(iou.max(0) > 0.5)[0]); tp_gt = set(np.nonzero(iou.max(1) > 0.5)[0])
    ys, xs = np.nonzero((g > 0) | (inst > 0)); cy, cx = int(np.median(ys)), int(np.median(xs)); rr = 450
    y0, x0 = np.clip(cy - rr, 0, 2048 - 2 * rr), np.clip(cx - rr, 0, 2048 - 2 * rr); sl = (slice(y0, y0 + 2 * rr), slice(x0, x0 + 2 * rr))
    ov = np.stack([img] * 3, -1)
    for k in range(1, npred + 1):
        ov[inst == k] = (0.2, 0.9, 0.2) if k in tp_pred else (1.0, 0.2, 0.2)
    for k in range(1, ng + 1):
        if k not in tp_gt:
            ov[(g == k) & (inst == 0)] = (0.3, 0.4, 1.0)
    gtv = np.stack([img] * 3, -1); gtv[g > 0] = (1.0, 0.8, 0.1)
    n = len(r["gt_paths"])
    axes[row, 0].imshow(img[sl], cmap="gray", vmin=0, vmax=1)
    axes[row, 0].set_title(f"{s}  (pooled PQ {pq_img[s]:.2f}, {n} annotator{'s' if n > 1 else ''})", fontsize=8)
    axes[row, 1].imshow(gtv[sl]); axes[row, 1].set_title("ground truth (first annotator)", fontsize=8)
    axes[row, 2].imshow(ov[sl]); axes[row, 2].set_title("refined instances vs first annotator: TP green, FP red, FN blue", fontsize=8)
    for a in axes[row]:
        a.axis("off")
fig.tight_layout(); fig.savefig(f"{OUT}/examples.pdf", dpi=150, bbox_inches="tight")
print("examples saved:", [r["stem"] for r in picks])
