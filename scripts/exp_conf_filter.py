"""Experiment: reject connected components by their peak / mean filament probability (FP reduction)."""
import sys, os, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd, cv2
from common import ROOT, RAW, panoptic_quality

cache = sys.argv[1]
splits = pd.read_csv(f"{ROOT}/data/splits.csv"); va = splits[splits.fold == 0].stem.tolist()
images_df = pd.DataFrame(json.load(open(f"{RAW}/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"))["images"])
preds = {s: np.load(f"{cache}/{s}.npy")[0].astype(np.float32) / 255 for s in va}
gts = {s: [cv2.imread(f"{ROOT}/data/masks/inst/{i}.png", cv2.IMREAD_UNCHANGED).astype(np.int32)
           for i in images_df[images_df.file_name == s + ".jpeg"].id] for s in va}

def instances(prob, thr, min_area, min_peak, min_mean):
    n, lab, stats, _ = cv2.connectedComponentsWithStats((prob > thr).astype(np.uint8), connectivity=8)
    out = np.zeros_like(lab); k = 0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area: continue
        m = lab == i; p = prob[m]
        if p.max() < min_peak or p.mean() < min_mean: continue
        k += 1; out[m] = k
    return out

def score(thr, a, pk, mn):
    pqs = []; tp = fp = fn = 0
    for s in va:
        inst = instances(preds[s], thr, a, pk, mn)
        for g in gts[s]:
            r = panoptic_quality(g, inst); pqs.append(r[0]); tp += r[1]; fp += r[2]; fn += r[3]
    return np.mean(pqs), tp, fp, fn

print("baseline", score(0.5, 400, 0, 0))
for thr in [0.4, 0.5]:
    for a in [250, 400]:
        for pk in [0.0, 0.7, 0.8, 0.9, 0.95]:
            for mn in [0.0, 0.6, 0.7]:
                pq, tp, fp, fn = score(thr, a, pk, mn)
                print(f"thr {thr} A {a} peak>={pk} mean>={mn}: PQ {pq:.4f} TP {tp} FP {fp} FN {fn}", flush=True)
