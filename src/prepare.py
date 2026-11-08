"""Rasterize MAGFiLO COCO polygon annotations into instance maps and soft consensus masks.

Outputs (all 2048x2048):
  data/masks/inst/<image_id>.png   uint16 instance-id map for each annotator set (0 = background)
  data/masks/soft/<file_stem>.png  uint8 consensus mask: 255 * (fraction of annotators marking the pixel)
  data/masks/dist/<file_stem>.png  uint8 consensus of per-instance normalised distance-to-boundary maps
                                   (1 at the instance's medial axis, 0 at its boundary)
  data/masks/touch/<file_stem>.png uint8 consensus "contact line": pixels within 2 px of two different instances
  data/splits.csv                  file_stem, date, fold assignment (grouped by observation date)
"""
import json, os, sys, collections
import numpy as np, cv2, pandas as pd
from pycocotools import mask as mutils

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANN = f"{ROOT}/data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
H = W = 2048

def normalised_distance(inst):
    """Per-instance Euclidean distance to the instance boundary, divided by the instance's maximum."""
    dist = cv2.distanceTransform((inst > 0).astype(np.uint8), cv2.DIST_L2, 3)
    # separate touching instances: boundary between two labels must also have distance 0
    for k in range(1, inst.max() + 1):
        m = inst == k
        if not m.any():
            continue
        dk = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
        dist[m] = dk[m] / max(dk[m].max(), 1.0)
    dist[inst == 0] = 0
    return dist.astype(np.float32)


def main():
    d = json.load(open(ANN))
    os.makedirs(f"{ROOT}/data/masks/inst", exist_ok=True)
    os.makedirs(f"{ROOT}/data/masks/soft", exist_ok=True)
    os.makedirs(f"{ROOT}/data/masks/dist", exist_ok=True)
    os.makedirs(f"{ROOT}/data/masks/touch", exist_ok=True)
    anns_by_img = collections.defaultdict(list)
    for a in d["annotations"]:
        anns_by_img[a["image_id"]].append(a)
    by_file = collections.defaultdict(list)
    dist_by_file = collections.defaultdict(list)
    touch_by_file = collections.defaultdict(list)
    touch_pairs = 0; total_inst = 0
    for im in d["images"]:
        inst = np.zeros((H, W), np.uint16)
        for k, a in enumerate(anns_by_img[im["id"]], 1):
            rle = mutils.frPyObjects(a["segmentation"], H, W)
            m = mutils.decode(mutils.merge(rle)).astype(bool)
            inst[m] = k
        # contact lines: pixels within 2 px of two different instances; also adjacency statistics
        touch = np.zeros((H, W), bool)
        for k in range(1, inst.max() + 1):
            m = (inst == k).astype(np.uint8)
            dil = cv2.dilate(m, np.ones((5, 5), np.uint8)) > 0
            others = dil & (inst != 0) & (inst != k)
            if others.any():
                touch_pairs += 1
                touch |= cv2.dilate(others.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        touch_by_file[im["file_name"]].append(touch & (inst > 0))
        total_inst += inst.max()
        cv2.imwrite(f"{ROOT}/data/masks/inst/{im['id']}.png", inst)
        by_file[im["file_name"]].append(inst > 0)
        dist_by_file[im["file_name"]].append(normalised_distance(inst))
    print(f"instances touching another within 3px: {touch_pairs}/{total_inst} ({100*touch_pairs/total_inst:.1f}%)")
    rows = []
    for fn, masks in by_file.items():
        soft = np.mean(np.stack(masks), 0)
        stem = fn.rsplit(".", 1)[0]
        cv2.imwrite(f"{ROOT}/data/masks/soft/{stem}.png", (soft * 255).round().astype(np.uint8))
        dist = np.mean(np.stack(dist_by_file[fn]), 0)
        cv2.imwrite(f"{ROOT}/data/masks/dist/{stem}.png", (dist * 255).round().astype(np.uint8))
        touch = np.mean(np.stack(touch_by_file[fn]), 0)
        cv2.imwrite(f"{ROOT}/data/masks/touch/{stem}.png", (touch * 255).round().astype(np.uint8))
        rows.append({"stem": stem, "date": stem[:8], "n_annot": len(masks)})
    df = pd.DataFrame(rows).sort_values("stem")
    dates = sorted(df["date"].unique())
    rng = np.random.RandomState(0); rng.shuffle(dates)
    fold_of = {dt: i % 5 for i, dt in enumerate(dates)}
    df["fold"] = df["date"].map(fold_of)
    df.to_csv(f"{ROOT}/data/splits.csv", index=False)
    print(df["fold"].value_counts().sort_index().to_dict(), "files per fold")

if __name__ == "__main__":
    main()
