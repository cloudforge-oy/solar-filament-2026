"""Predict filament instances on the test images and write a Kaggle submission CSV.

Pipeline: full-resolution U-Net probability map (optional flip TTA, optional model ensemble)
-> threshold -> connected components -> drop small components -> COCO RLE per instance.
Connected components are disjoint by construction, so no two rows of one image overlap.
"""
import argparse, os, glob
import numpy as np, pandas as pd, torch, cv2
from common import ROOT, RAW, load_image, build_model, masks_to_instances, masks_to_instances_watershed, masks_to_instances_split, encode_rle
from train import predict_full


def load_models(paths, device):
    models = []
    for p in paths:
        enc = os.path.basename(os.path.dirname(p)).split("_f")[0]
        m = build_model(enc, weights=None); m.load_state_dict(torch.load(p, map_location=device)); m.to(device).eval()
        models.append(m)
    return models


def predict_prob(models, img, device, tta):
    """Ensemble by averaging the (2, H, W) outputs of all models."""
    return np.mean([predict_full(m, img, device, tta) for m in models], 0)


def extract_instances(out, method, thr, min_area, aux_thr):
    """method: 'cc' (components), 'split' (components cut at predicted contact lines, aux_thr = contact
    threshold) or 'watershed' (distance-map watershed, aux_thr = seed threshold)."""
    if method == "watershed":
        return masks_to_instances_watershed(out[0], out[1], thr, min_area, aux_thr)
    if method == "split":
        return masks_to_instances_split(out[0], out[2], thr, min_area, aux_thr)
    return masks_to_instances(out[0], thr, min_area)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--min_area", type=int, default=150)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--method", choices=["cc", "split", "watershed"], default="split")
    ap.add_argument("--aux_thr", type=float, default=0.5)
    ap.add_argument("--out", default=f"{ROOT}/submission.csv")
    ap.add_argument("--save_probs", default=None, help="directory to store uint8 probability maps")
    args = ap.parse_args()
    device = "cuda"
    models = load_models(args.weights, device)
    paths = sorted(glob.glob(f"{RAW}/test/test_images/*.jpeg"))
    rows = []; n_img_with = 0
    for p in paths:
        stem = os.path.basename(p).rsplit(".", 1)[0]
        out = predict_prob(models, load_image(p), device, args.tta)
        if args.save_probs:
            os.makedirs(args.save_probs, exist_ok=True)
            np.save(f"{args.save_probs}/{stem}.npy", (out * 255).round().astype(np.uint8))
        inst = extract_instances(out, args.method, args.thr, args.min_area, args.aux_thr)
        n_img_with += inst.max() > 0
        for k in range(1, inst.max() + 1):
            rows.append({"filament_id": f"{stem}_{k}", "segmentation_rle": encode_rle(inst == k)})
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    df.to_csv(args.out, index=False)
    print(f"{len(df)} filaments on {n_img_with}/{len(paths)} images -> {args.out}")


if __name__ == "__main__":
    main()
