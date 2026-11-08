"""Evaluate single models and their ensemble on a validation fold.

Caches each model's (3, H, W) prediction per validation image (uint8 .npy) so that ensembles and
post-processing variants are cheap to compare. Reports Dice, PQ (mean over annotation sets with
standard error and a 95% bootstrap interval over images), and TP/FP/FN counts.
"""
import argparse, os, json, itertools
import numpy as np, pandas as pd, cv2, torch
from common import ROOT, RAW, load_image, panoptic_quality
from predict import load_models, predict_prob, extract_instances


def cached_preds(model_path, stems, device, tta, cache_dir):
    """Predict (3, H, W) maps for all stems with one model, caching them as uint8 .npy files."""
    missing = [s for s in stems if not os.path.exists(f"{cache_dir}/{s}.npy")]
    if missing:
        models = load_models([model_path], device)
        for s in missing:
            out = predict_prob(models, load_image(f"{RAW}/train/train_images/{s}.jpeg"), device, tta)
            np.save(f"{cache_dir}/{s}.npy", (out * 255).round().astype(np.uint8))
        del models; torch.cuda.empty_cache()
    return {s: np.load(f"{cache_dir}/{s}.npy").astype(np.float32) / 255 for s in stems}


def score(preds, gts, softs, method, thr, min_area, aux, rng=None, n_boot=0):
    per_image_pq, per_set_pq, dices = [], [], []
    tp = fp = fn = 0
    for stem in preds:
        inst = extract_instances(preds[stem], method, thr, min_area, aux)
        pqs = []
        for g in gts[stem]:
            r = panoptic_quality(g, inst); pqs.append(r[0]); tp += r[1]; fp += r[2]; fn += r[3]
        per_set_pq += pqs; per_image_pq.append(np.mean(pqs))
        pb = preds[stem][0] > thr; gb = softs[stem] > 0.5
        dices.append(2 * (pb & gb).sum() / (pb.sum() + gb.sum() + 1e-6))
    out = {"pq": float(np.mean(per_set_pq)), "pq_se": float(np.std(per_set_pq, ddof=1) / np.sqrt(len(per_set_pq))),
           "dice": float(np.mean(dices)), "tp": tp, "fp": fp, "fn": fn, "n_sets": len(per_set_pq)}
    if n_boot:
        arr = np.array(per_image_pq); bs = [arr[rng.randint(0, len(arr), len(arr))].mean() for _ in range(n_boot)]
        out["pq_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True, help="checkpoints; the ensemble of all of them is also evaluated")
    ap.add_argument("--names", nargs="+", default=None)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--cache", default=f"{ROOT}/runs/evalcache")
    ap.add_argument("--out", default=f"{ROOT}/runs/evaluation.json")
    args = ap.parse_args()
    device = "cuda"
    names = args.names or [os.path.basename(os.path.dirname(w)) for w in args.weights]
    splits = pd.read_csv(f"{ROOT}/data/splits.csv"); va = splits[splits.fold == args.fold].stem.tolist()
    images_df = pd.DataFrame(json.load(open(f"{RAW}/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"))["images"])
    gts = {s: [cv2.imread(f"{ROOT}/data/masks/inst/{i}.png", cv2.IMREAD_UNCHANGED).astype(np.int32)
               for i in images_df[images_df.file_name == s + ".jpeg"].id] for s in va}
    softs = {s: cv2.imread(f"{ROOT}/data/masks/soft/{s}.png", cv2.IMREAD_UNCHANGED) / 255.0 for s in va}
    preds = {}
    for w, n in zip(args.weights, names):
        d = f"{args.cache}/{n}{'_tta' if args.tta else ''}"; os.makedirs(d, exist_ok=True)
        preds[n] = cached_preds(w, va, device, args.tta, d)
    if len(names) > 1:
        preds["ensemble"] = {s: np.mean([preds[n][s] for n in names], 0) for s in va}
    rng = np.random.RandomState(0)
    results = {}
    grid = [(m, t, a, x) for m in ["cc", "split"] for t in [0.4, 0.5, 0.6, 0.7] for a in [250, 400, 550] for x in ([0.0] if m == "cc" else [0.3, 0.5, 0.7])]
    for n, p in preds.items():
        best = None
        for m, t, a, x in grid:
            r = score(p, gts, softs, m, t, a, x)
            if best is None or r["pq"] > best["pq"]:
                best = dict(r, method=m, thr=t, min_area=a, aux=x)
        best = dict(best, **score(p, gts, softs, best["method"], best["thr"], best["min_area"], best["aux"], rng, 1000))
        # also report plain CC at the same thr/min_area for the ablation of the contact-line split
        cc = score(p, gts, softs, "cc", best["thr"], best["min_area"], 0.0)
        best["pq_cc_same_params"] = cc["pq"]
        results[n] = best
        print(f"{n:12s} {best['method']:5s} thr {best['thr']:.1f} A_min {best['min_area']:3d} aux {best['aux']:.1f} | "
              f"Dice {best['dice']:.4f} PQ {best['pq']:.4f} ± {best['pq_se']:.4f} CI {best['pq_ci95'][0]:.3f}-{best['pq_ci95'][1]:.3f} | "
              f"TP {best['tp']} FP {best['fp']} FN {best['fn']} | cc-only PQ {cc['pq']:.4f}", flush=True)
    json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
