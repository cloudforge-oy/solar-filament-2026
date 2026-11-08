"""Sweep heat-map and qualitative examples for the report, from the evaluation cache of the ensemble."""
import sys, os, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import ROOT, RAW, load_image, panoptic_quality
from predict import extract_instances

names = sys.argv[1:]  # cache sub-directories to average, e.g. resnet34d_f0_tta effb3d_f0_tta
ev = json.load(open(f"{ROOT}/runs/evaluation.json"))["ensemble" if len(names) > 1 else names[0]]
splits = pd.read_csv(f"{ROOT}/data/splits.csv"); va = splits[splits.fold == 0].stem.tolist()
images_df = pd.DataFrame(json.load(open(f"{RAW}/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"))["images"])
preds = {s: np.mean([np.load(f"{ROOT}/runs/evalcache/{n}/{s}.npy") for n in names], 0).astype(np.float32) / 255 for s in va}
gts = {s: [cv2.imread(f"{ROOT}/data/masks/inst/{i}.png", cv2.IMREAD_UNCHANGED).astype(np.int32)
           for i in images_df[images_df.file_name == s + ".jpeg"].id] for s in va}

# --- sweep heat map
thrs, areas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [100, 200, 300, 400, 550, 700, 900]
grid = np.zeros((len(thrs), len(areas)))
for i, t in enumerate(thrs):
    for j, a in enumerate(areas):
        grid[i, j] = np.mean([panoptic_quality(g, extract_instances(preds[s], ev["method"], t, a, ev["aux"]))[0] for s in va for g in gts[s]])
fig, ax = plt.subplots(figsize=(3.4, 2.4))
im = ax.imshow(grid, cmap="viridis", aspect="auto"); ax.set_xticks(range(len(areas))); ax.set_xticklabels(areas, fontsize=7)
ax.set_yticks(range(len(thrs))); ax.set_yticklabels(thrs, fontsize=7); ax.set_xlabel("$A_{\\min}$ (px)", fontsize=8); ax.set_ylabel("threshold $\\tau$", fontsize=8)
for i in range(len(thrs)):
    for j in range(len(areas)):
        ax.text(j, i, f"{grid[i,j]:.3f}", ha="center", va="center", fontsize=5.5, color="w" if grid[i, j] < grid.max() - 0.04 else "k")
fig.colorbar(im, ax=ax, label="PQ"); fig.savefig(f"{ROOT}/report/figures/sweep.pdf", bbox_inches="tight"); plt.close(fig)
print("sweep best", grid.max(), "at", thrs[int(np.argmax(grid) // len(areas))], areas[int(np.argmax(grid) % len(areas))])

# --- qualitative examples: pick images at the 25th, 50th, 75th percentile of PQ and crop around filaments
pq_img = {s: np.mean([panoptic_quality(g, extract_instances(preds[s], ev["method"], ev["thr"], ev["min_area"], ev["aux"]))[0] for g in gts[s]]) for s in va}
order = sorted(va, key=lambda s: pq_img[s]); picks = [order[len(order) // 4], order[len(order) // 2], order[3 * len(order) // 4]]
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
for row, s in enumerate(picks):
    img = load_image(f"{RAW}/train/train_images/{s}.jpeg"); g = gts[s][0]
    inst = extract_instances(preds[s], ev["method"], ev["thr"], ev["min_area"], ev["aux"])
    ys, xs = np.nonzero((g > 0) | (inst > 0)); cy, cx = int(np.median(ys)), int(np.median(xs)); r = 450
    y0, x0 = np.clip(cy - r, 0, 2048 - 2 * r), np.clip(cx - r, 0, 2048 - 2 * r); sl = (slice(y0, y0 + 2 * r), slice(x0, x0 + 2 * r))
    # TP/FP/FN colouring
    ng, npred = g.max(), inst.max(); inter = np.zeros((ng + 1, npred + 1), np.int64); np.add.at(inter, (g.ravel(), inst.ravel()), 1)
    ag = np.bincount(g.ravel(), minlength=ng + 1)[:, None]; ap = np.bincount(inst.ravel(), minlength=npred + 1)[None, :]
    iou = inter / np.maximum(ag + ap - inter, 1); iou[0, :] = iou[:, 0] = 0
    tp_pred = set(np.nonzero(iou.max(0) > 0.5)[0]); tp_gt = set(np.nonzero(iou.max(1) > 0.5)[0])
    overlay = np.stack([img] * 3, -1)
    for k in range(1, npred + 1):
        overlay[inst == k] = (0.2, 0.9, 0.2) if k in tp_pred else (1.0, 0.2, 0.2)
    for k in range(1, ng + 1):
        if k not in tp_gt: overlay[(g == k) & (inst == 0)] = (0.3, 0.4, 1.0)
    axes[row, 0].imshow(img[sl], cmap="gray", vmin=0, vmax=1); axes[row, 0].set_title(f"{s}  (PQ {pq_img[s]:.2f})", fontsize=8)
    gtv = np.stack([img] * 3, -1); gtv[g > 0] = (1.0, 0.8, 0.1); axes[row, 1].imshow(gtv[sl]); axes[row, 1].set_title("ground truth (one annotator)", fontsize=8)
    axes[row, 2].imshow(overlay[sl]); axes[row, 2].set_title("prediction: TP green, FP red, FN blue", fontsize=8)
    for a in axes[row]: a.axis("off")
fig.tight_layout(); fig.savefig(f"{ROOT}/report/figures/examples.pdf", dpi=150, bbox_inches="tight"); print("figures saved")
