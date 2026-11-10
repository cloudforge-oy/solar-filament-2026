"""Train a U-Net for filament pixel segmentation on random full-resolution crops.

Targets (all consensus over annotators): soft filament mask, normalised distance-to-boundary map,
and the contact line between touching filaments, which is used at inference to split merged components.
Validation is run on full 2048x2048 images and reports Dice plus Panoptic Quality
(instances from connected components, averaged over all annotator sets of each image).
"""
import argparse, os, time, json, glob
import numpy as np, pandas as pd, cv2, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from common import ROOT, RAW, load_image, normalize, norm_stats, build_model, masks_to_instances, panoptic_quality


class CropDataset(Dataset):
    def __init__(self, stems, crop=512, per_image=4, fg_prob=0.7, norm="fixed"):
        self.stems, self.crop, self.per_image, self.fg_prob, self.norm = stems, crop, per_image, fg_prob, norm

    def __len__(self):
        return len(self.stems) * self.per_image

    def load(self, stem):
        img = load_image(f"{RAW}/train/train_images/{stem}.jpeg")
        soft = cv2.imread(f"{ROOT}/data/masks/soft/{stem}.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        dist = cv2.imread(f"{ROOT}/data/masks/dist/{stem}.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        touch = cv2.imread(f"{ROOT}/data/masks/touch/{stem}.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        ys, xs = np.nonzero(soft[::4, ::4] > 0.4)  # subsampled grid is enough to pick a crop centre
        return img, np.stack([soft, dist, touch]), ys * 4, xs * 4

    def __getitem__(self, i):
        img, soft, ys, xs = self.load(self.stems[i % len(self.stems)])
        stats = norm_stats(img, self.norm)                 # from the whole, un-augmented image
        c = self.crop
        if len(ys) and np.random.rand() < self.fg_prob:
            j = np.random.randint(len(ys))
            cy, cx = ys[j] + np.random.randint(-c // 3, c // 3), xs[j] + np.random.randint(-c // 3, c // 3)
        else:
            cy, cx = np.random.randint(c // 2, 2048 - c // 2, size=2)
        y0 = int(np.clip(cy - c // 2, 0, 2048 - c)); x0 = int(np.clip(cx - c // 2, 0, 2048 - c))
        x = img[y0:y0 + c, x0:x0 + c].copy(); y = soft[:, y0:y0 + c, x0:x0 + c].copy()
        # geometric augmentation: flips + 90 degree rotations
        if np.random.rand() < 0.5: x, y = x[:, ::-1], y[:, :, ::-1]
        if np.random.rand() < 0.5: x, y = x[::-1], y[:, ::-1]
        k = np.random.randint(4); x, y = np.rot90(x, k), np.rot90(y, k, axes=(1, 2))
        # photometric augmentation: gamma, contrast, brightness, mild noise
        x = np.clip(x, 0, 1) ** np.exp(np.random.uniform(-0.3, 0.3))
        x = x * np.random.uniform(0.85, 1.15) + np.random.uniform(-0.05, 0.05)
        x = x + np.random.normal(0, np.random.uniform(0, 0.02), x.shape).astype(np.float32)
        x = normalize(np.ascontiguousarray(x, dtype=np.float32), stats); y = np.ascontiguousarray(y, dtype=np.float32)
        return torch.from_numpy(x)[None], torch.from_numpy(y)


def soft_dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum((1, 2, 3)) + eps
    den = p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps
    return (1 - num / den).mean()


def tta_views(tta):
    """Test-time views as (rot90 k, flip dims): 1 view, the 4 flips (tta=True or 4) or the full dihedral group (8)."""
    n = 4 if tta is True else int(tta or 1)
    flips = [(), (-1,), (-2,), (-1, -2)]
    return [(k, f) for k in ((0, 1) if n >= 8 else (0,)) for f in (flips if n >= 4 else flips[:1])]


@torch.no_grad()
def predict_full(model, img, device, tta=False):
    """Returns float32 array (3, H, W): [0] filament probability, [1] normalised distance map, [2] contact line.
    The image is standardised the way the model was trained (model.norm, "fixed" if the attribute is missing)."""
    x = torch.from_numpy(normalize(img, norm_stats(img, getattr(model, "norm", "fixed"))))[None, None].to(device)
    views = tta_views(tta)
    p = 0
    with torch.autocast("cuda", dtype=torch.float16):
        for k, f in views:
            z = torch.rot90(x, k, (-2, -1)) if k else x
            q = torch.sigmoid(model(z.flip(f) if f else z)).float()
            q = q.flip(f) if f else q
            p = p + (torch.rot90(q, -k, (-2, -1)) if k else q)
    return (p / len(views))[0].cpu().numpy()


def evaluate(model, stems, device, images_df, thr=0.5, min_area=150, tta=False):
    model.eval()
    dices, pqs = [], []
    for stem in stems:
        img = load_image(f"{RAW}/train/train_images/{stem}.jpeg")
        prob = predict_full(model, img, device, tta)[0]
        soft = cv2.imread(f"{ROOT}/data/masks/soft/{stem}.png", cv2.IMREAD_UNCHANGED) / 255.0
        pred_bin = prob > thr; gt_bin = soft > 0.5
        dices.append(2 * (pred_bin & gt_bin).sum() / (pred_bin.sum() + gt_bin.sum() + 1e-6))
        inst = masks_to_instances(prob, thr, min_area)
        for iid in images_df[images_df.file_name == stem + ".jpeg"].id:
            gt = cv2.imread(f"{ROOT}/data/masks/inst/{iid}.png", cv2.IMREAD_UNCHANGED).astype(np.int32)
            pqs.append(panoptic_quality(gt, inst)[0])
    return float(np.mean(dices)), float(np.mean(pqs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--per_image", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--norm", choices=["fixed", "disk"], default="disk", help="input standardisation, see common.norm_stats")
    args = ap.parse_args()
    out = args.out or f"{ROOT}/runs/{args.encoder}_f{args.fold}"
    os.makedirs(out, exist_ok=True)
    json.dump(vars(args), open(f"{out}/config.json", "w"), indent=1)
    torch.manual_seed(0); np.random.seed(0)
    device = "cuda"
    splits = pd.read_csv(f"{ROOT}/data/splits.csv")
    images_df = pd.DataFrame(json.load(open(f"{RAW}/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"))["images"])
    tr = splits[splits.fold != args.fold].stem.tolist(); va = splits[splits.fold == args.fold].stem.tolist()
    print(f"train files {len(tr)}, val files {len(va)}")
    ds = CropDataset(tr, args.crop, args.per_image, norm=args.norm)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=args.workers, drop_last=True, persistent_workers=True)
    model = build_model(args.encoder).to(device).to(memory_format=torch.channels_last)
    model.norm = args.norm
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    scaler = torch.amp.GradScaler()
    bce = nn.BCEWithLogitsLoss()
    touch_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20.0, device=device))  # contact lines are rare
    best = -1; log = []
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0
        for x, y in dl:
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last); y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                pred = model(x)
                seg, dist, touch = pred[:, :1].float(), pred[:, 1:2].float(), pred[:, 2:].float()
                loss = bce(seg, y[:, :1]) + soft_dice_loss(seg, y[:, :1]) \
                    + 5.0 * F.mse_loss(torch.sigmoid(dist), y[:, 1:2]) \
                    + 0.5 * touch_bce(touch, y[:, 2:])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        msg = f"ep {ep+1}/{args.epochs} loss {tot/len(dl):.4f} time {time.time()-t0:.0f}s"
        if (ep + 1) % args.eval_every == 0 or ep + 1 == args.epochs:
            dice, pq = evaluate(model, va, device, images_df)
            msg += f" | val dice {dice:.4f} PQ {pq:.4f}"
            log.append({"epoch": ep + 1, "dice": dice, "pq": pq})
            if pq > best:
                best = pq; torch.save(model.state_dict(), f"{out}/best.pt")
        print(msg, flush=True)
        torch.save(model.state_dict(), f"{out}/last.pt")
    json.dump({"args": vars(args), "log": log, "best_pq": best}, open(f"{out}/log.json", "w"), indent=1)
    print("best PQ", best)


if __name__ == "__main__":
    main()
