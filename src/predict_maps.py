"""Cache probability maps for the decision layer.

  oof : for every fold k, the fold-k model predicts the fold-k (held-out) training images
        -> <out>/<stem>.npy, uint8 (3, H, W) = 255 * [filament prob, distance, contact]
  test: the average of all given models on the test images -> <out>/<stem>.npy (same layout)
Flip TTA is always on. Existing files are skipped, so the job is resumable.
"""
import argparse, glob, os, re
import numpy as np, pandas as pd, torch
from common import ROOT, RAW, load_image
from predict import load_models, predict_prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["oof", "test"])
    ap.add_argument("--weights", nargs="+", required=True, help="oof: one checkpoint per fold, name must contain _f<k>")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = []
    if a.mode == "oof":
        splits = pd.read_csv(f"{ROOT}/data/splits.csv")
        for w in a.weights:
            k = int(re.search(r"_f(\d)", os.path.basename(w)).group(1))
            stems = splits[splits.fold == k].stem.tolist()
            jobs.append(([w], [(s, f"{RAW}/train/train_images/{s}.jpeg") for s in stems]))
    else:
        paths = sorted(glob.glob(f"{RAW}/test/test_images/*.jpeg"))
        jobs.append((a.weights, [(os.path.basename(p)[:-5], p) for p in paths]))
    for weights, items in jobs:
        todo = [(s, p) for s, p in items if not os.path.exists(f"{a.out}/{s}.npy")]
        print(f"{weights}: {len(items)} images, {len(todo)} to predict", flush=True)
        if not todo:
            continue
        models = load_models(weights, "cuda")
        for i, (s, p) in enumerate(todo):
            out = predict_prob(models, load_image(p), "cuda", tta=True)
            np.save(f"{a.out}/{s}.npy.tmp.npy", (out * 255).round().astype(np.uint8))
            os.replace(f"{a.out}/{s}.npy.tmp.npy", f"{a.out}/{s}.npy")
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(todo)}", flush=True)
        del models; torch.cuda.empty_cache()
    print("done", flush=True)


if __name__ == "__main__":
    main()
