"""Sweep post-processing parameters (probability threshold, min component area) on validation PQ.

Caches full-resolution probability maps for the validation fold so the sweep itself is cheap.
"""
import argparse, os, json
import numpy as np, pandas as pd, cv2, torch
from common import ROOT, RAW, load_image, panoptic_quality
from predict import load_models, predict_prob, extract_instances


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--cache", default=f"{ROOT}/runs/valprobs")
    ap.add_argument("--methods", nargs="+", default=["cc", "split", "watershed"])
    args = ap.parse_args()
    device = "cuda"
    splits = pd.read_csv(f"{ROOT}/data/splits.csv")
    images_df = pd.DataFrame(json.load(open(f"{RAW}/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"))["images"])
    va = splits[splits.fold == args.fold].stem.tolist()
    os.makedirs(args.cache, exist_ok=True)
    models = load_models(args.weights, device)
    probs, gts = {}, {}
    for stem in va:
        f = f"{args.cache}/{stem}.npy"
        if not os.path.exists(f):
            np.save(f, (predict_prob(models, load_image(f"{RAW}/train/train_images/{stem}.jpeg"), device, args.tta) * 255).astype(np.uint8))
        probs[stem] = np.load(f).astype(np.float32) / 255
        gts[stem] = [cv2.imread(f"{ROOT}/data/masks/inst/{iid}.png", cv2.IMREAD_UNCHANGED).astype(np.int32)
                     for iid in images_df[images_df.file_name == stem + ".jpeg"].id]
    results = []
    for method in args.methods:
        for aux_thr in ([0.3, 0.5, 0.7] if method != "cc" else [0.0]):
            for thr in [0.4, 0.5, 0.6, 0.7]:
                for min_area in [250, 400, 550]:
                    pqs = []
                    for stem in va:
                        inst = extract_instances(probs[stem], method, thr, min_area, aux_thr)
                        pqs += [panoptic_quality(g, inst)[0] for g in gts[stem]]
                    results.append((method, aux_thr, thr, min_area, float(np.mean(pqs))))
                    print(f"{method:9s} aux {aux_thr:.1f} thr {thr:.2f} min_area {min_area:4d}  PQ {np.mean(pqs):.4f}", flush=True)
    best = max(results, key=lambda r: r[-1]); print("best", best)
    json.dump({"results": results, "best": best}, open(f"{args.cache}/sweep.json", "w"))


if __name__ == "__main__":
    main()
