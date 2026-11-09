"""Re-run only the post-processing on probability maps saved by predict.py --save_probs (no GPU needed)."""
import argparse, os, glob
import numpy as np, pandas as pd
from common import encode_rle
from predict import extract_instances

ap = argparse.ArgumentParser()
ap.add_argument("--probs", required=True, help="directory of <stem>.npy uint8 (3, H, W) maps")
ap.add_argument("--thr", type=float, default=0.6)
ap.add_argument("--min_area", type=int, default=400)
ap.add_argument("--method", choices=["cc", "split", "watershed"], default="cc")
ap.add_argument("--aux_thr", type=float, default=0.5)
ap.add_argument("--out", required=True)
a = ap.parse_args()
rows = []; n_img = 0
for f in sorted(glob.glob(f"{a.probs}/*.npy")):
    stem = os.path.basename(f)[:-4]
    inst = extract_instances(np.load(f).astype(np.float32) / 255, a.method, a.thr, a.min_area, a.aux_thr)
    n_img += inst.max() > 0
    for k in range(1, inst.max() + 1):
        rows.append({"filament_id": f"{stem}_{k}", "segmentation_rle": encode_rle(inst == k)})
pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(a.out, index=False)
print(f"{len(rows)} filaments on {n_img} images -> {a.out}")
