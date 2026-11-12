"""Draw the pipeline diagram for the report (report/figures/pipeline.pdf)."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def box(ax, x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="#333", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5, wrap=True)

def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=0.8, color="#333"))

fig, ax = plt.subplots(figsize=(10, 3.1)); ax.set_xlim(0, 10); ax.set_ylim(0, 3.1); ax.axis("off")
ax.text(0.05, 2.95, "Training", fontsize=9, weight="bold", va="top"); ax.text(0.05, 1.35, "Inference", fontsize=9, weight="bold", va="top")
tr = ["#dbe9f6", "#dbe9f6", "#e3f1dc", "#e3f1dc", "#fbe7c6"]
tx = ["COCO polygons\n(1–3 annotator sets\nper image)", "Rasterise to\ninstance maps\n2048×2048",
      "Consensus targets:\nsoft mask, distance,\ncontact line", "Random native-res\ncrops + flips/rot90\n+ photometric jitter",
      "U-Nets: EffNetV2-S and\nEffNet-B3 encoders,\n5 folds each"]
xs = [0.1, 2.1, 4.1, 6.1, 8.1]
for x, t, c in zip(xs, tx, tr):
    box(ax, x, 1.75, 1.75, 1.0, t, c)
for a, b in zip(xs[:-1], xs[1:]):
    arrow(ax, a + 1.75, 2.25, b, 2.25)
inf = ["H-α test image\n2048×2048: full frame,\n8 views × 10 models", "Component tree:\n16 threshold levels,\nclosing + hole fill",
       "GBM scorer û = ĥ·m̂\n(67 descriptors),\nexact tree DP", "Stage 2: UNet++ refiner\nre-segments each\nselected instance",
       "Pixel-disjoint\nCOCO RLE →\nsubmission.csv"]
ic = ["#fbe7c6", "#f6d8d8", "#f6d8d8", "#e6dcf2", "#e8e8e8"]
for x, t, c in zip(xs, inf, ic):
    box(ax, x, 0.15, 1.75, 1.0, t, c)
for a, b in zip(xs[:-1], xs[1:]):
    arrow(ax, a + 1.75, 0.65, b, 0.65)
arrow(ax, 8.95, 1.75, 0.98, 1.15)  # trained weights to inference
ax.text(5.0, 1.47, "trained weights; out-of-fold maps train the scorer and the refiner", fontsize=7, ha="center", style="italic",
        bbox=dict(fc="white", ec="none", pad=1.5))
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "report", "figures", "pipeline.pdf")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight"); print("saved", os.path.relpath(out))
