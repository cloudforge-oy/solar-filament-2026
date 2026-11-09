"""Report figures for the decision layer.

  report/figures/sweep.pdf     pooled out-of-fold PQ against the selection threshold tau, with the fixed rules
  report/figures/examples.pdf  fold-0 validation images at the 25/50/75th percentile of PQ, decision-layer output

Needs runs/decision/{bank_effb3d.pkl, oof_scores.pkl, scorer_effb3d.pkl} and the out-of-fold maps in
runs/maps/oof_effb3d (see README, "Train from scratch").
usage: python scripts/make_report_figures.py [tau]
"""
import json, os, pickle, sys
import cv2
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from common import RAW, load_image  # noqa: E402
from decision import process_image, select, pooled_parts, pq_of  # noqa: E402

TAU = float(sys.argv[1]) if len(sys.argv) > 1 else 0.26
OUT = os.path.join(ROOT, "report", "figures"); os.makedirs(OUT, exist_ok=True)


def load(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------- tau curve
fit = load("runs/decision/scorer_effb3d.pkl")
oof = pd.DataFrame(fit["oof"])
fig, ax = plt.subplots(figsize=(3.4, 2.3))
ax.plot(oof.tau, oof.pq, "o-", ms=3, color="#2a78d6", label="decision layer")
ax.axhline(fit["baseline"], color="#eb6834", ls="--", lw=1.2, label="CC rule (p>0.6, A$\\geq$400)")
ax.axhline(fit["best_fixed"][0], color="#8a8984", ls=":", lw=1.2, label="best single tree level")
ax.axvline(TAU, color="#52514e", lw=0.8)
ax.set_xlabel("selection threshold $\\tau$", fontsize=8); ax.set_ylabel("pooled OOF PQ", fontsize=8)
ax.tick_params(labelsize=7); ax.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
ax.grid(alpha=0.3)
fig.savefig(f"{OUT}/sweep.pdf", bbox_inches="tight"); plt.close(fig)
print("tau curve saved")

# ---------------------------------------------------------------- examples
bank = {r["stem"]: r for r in load("runs/decision/bank_effb3d.pkl")}
scores = load("runs/decision/oof_scores.pkl")
with open(f"{RAW}/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json") as fh:
    images = pd.DataFrame(json.load(fh)["images"])
va = [s for s, r in bank.items() if r["fold"] == 0 and len(r["nodes"])]
pq_img = {}
for s in va:
    r = bank[s]; ch = select(r["nodes"].parent.to_numpy(), scores[s] - TAU)
    pq_img[s] = pq_of([pooled_parts(r["ious"], r["n_gt"], ch)])[0]
order = sorted(va, key=pq_img.get)
picks = [order[len(order) // 4], order[len(order) // 2], order[3 * len(order) // 4]]
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
for row, s in enumerate(picks):
    ids = images[images.file_name == s + ".jpeg"].id.tolist()
    gt_paths = [os.path.join(ROOT, "data", "masks", "inst", f"{i}.png") for i in ids]
    r = process_image(s, os.path.join(ROOT, "runs", "maps", "oof_effb3d", f"{s}.npy"),
                      f"{RAW}/train/train_images/{s}.jpeg", gt_paths, keep_labels=True)
    nd = r["nodes"]; ch = select(nd.parent.to_numpy(), scores[s] - TAU)
    inst = np.zeros((2048, 2048), np.int32)
    for k, v in enumerate(ch, 1):
        inst[r["labels"][int(nd.level_idx.iloc[v])] == int(nd.lab_id.iloc[v])] = k
    g = cv2.imread(gt_paths[0], cv2.IMREAD_UNCHANGED).astype(np.int32)
    ng, npred = g.max(), inst.max()
    inter = np.zeros((ng + 1, npred + 1), np.int64); np.add.at(inter, (g.ravel(), inst.ravel()), 1)
    ag = np.bincount(g.ravel(), minlength=ng + 1)[:, None]; ap = np.bincount(inst.ravel(), minlength=npred + 1)[None, :]
    iou = inter / np.maximum(ag + ap - inter, 1); iou[0, :] = 0; iou[:, 0] = 0
    tp_pred = set(np.nonzero(iou.max(0) > 0.5)[0]); tp_gt = set(np.nonzero(iou.max(1) > 0.5)[0])
    img = load_image(f"{RAW}/train/train_images/{s}.jpeg")
    ys, xs = np.nonzero((g > 0) | (inst > 0)); cy, cx = int(np.median(ys)), int(np.median(xs)); rr = 450
    y0, x0 = np.clip(cy - rr, 0, 2048 - 2 * rr), np.clip(cx - rr, 0, 2048 - 2 * rr); sl = (slice(y0, y0 + 2 * rr), slice(x0, x0 + 2 * rr))
    ov = np.stack([img] * 3, -1)
    for k in range(1, npred + 1):
        ov[inst == k] = (0.2, 0.9, 0.2) if k in tp_pred else (1.0, 0.2, 0.2)
    for k in range(1, ng + 1):
        if k not in tp_gt:
            ov[(g == k) & (inst == 0)] = (0.3, 0.4, 1.0)
    gtv = np.stack([img] * 3, -1); gtv[g > 0] = (1.0, 0.8, 0.1)
    axes[row, 0].imshow(img[sl], cmap="gray", vmin=0, vmax=1)
    axes[row, 0].set_title(f"{s}  (pooled PQ {pq_img[s]:.2f}, {len(ids)} annotator{'s' if len(ids) > 1 else ''})", fontsize=8)
    axes[row, 1].imshow(gtv[sl]); axes[row, 1].set_title("ground truth (first annotator)", fontsize=8)
    axes[row, 2].imshow(ov[sl]); axes[row, 2].set_title("decision layer vs first annotator: TP green, FP red, FN blue", fontsize=8)
    for a in axes[row]:
        a.axis("off")
fig.tight_layout(); fig.savefig(f"{OUT}/examples.pdf", dpi=150, bbox_inches="tight")
print("examples saved:", picks)
