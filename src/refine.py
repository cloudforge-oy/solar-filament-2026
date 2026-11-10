"""Stage 2: instance refinement.

The dense maps estimate p(x) = P(a random annotator marks pixel x). For a filament that only some annotators draw,
p is low everywhere inside it, so no threshold of p gives the outline an annotator would draw. The refiner is a
second U-Net that sees a native-resolution window around ONE selected instance and predicts
q(x) = P(x is inside the annotator's outline | the annotator drew this filament).

Input channels: image, p, the selected instance (seed), the other selected instances of the image.
Target: the mean of the matched ground-truth outlines (IoU > 0.5 with the seed) over the annotator records.
Like the scorer, the refiner is cross-fitted: the fold-k model never sees fold-k images.

Subcommands
  seeds    stage-1 selections and training seeds for all training images (from out-of-fold maps + scores)
  train    train the refiner of one fold
  eval     out-of-fold PQ before/after refinement, over a grid of pixel thresholds
  predict  refine a stage-1 submission CSV on the test images with the fold ensemble
"""
import argparse, glob, json, os, sys, time
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, RAW, load_image, normalize, norm_stats, encode_rle  # noqa: E402
from decision import process_image, select, pq_of, _iou_with_records, _load, _dump  # noqa: E402

PAD = 48            # context around the seed's bounding box at inference (pixels)
CROP = 256          # training window
MARGIN = 0.12       # training seeds: candidates scoring within this margin below tau, or selected
SIZE = {"size": [2048, 2048]}


def _enc(mask):
    return encode_rle(mask)


def _dec(counts):
    from pycocotools import mask as mutils
    return mutils.decode({**SIZE, "counts": counts.encode("ascii")}).astype(bool)


# ------------------------------------------------------------------------------------------------- seeds
def _seed_job(args):
    stem, maps_paths, img_path, gt_paths, u, tau, p_dir = args
    r = process_image(stem, maps_paths, img_path, gt_paths, keep_labels=True)
    nd = r["nodes"]; n = len(nd)
    fams = [np.load(q)[0] for q in maps_paths]
    p8 = fams[0] if len(fams) == 1 else np.round(np.mean(fams, 0)).astype(np.uint8)
    cv2.imwrite(f"{p_dir}/{stem}.png", p8)
    out = dict(stem=stem, n_gt=r["n_gt"], gt_paths=list(gt_paths), stats=norm_stats(load_image(img_path), "disk"),
               sel=[], train=[])
    if n == 0:
        cv2.imwrite(f"{os.path.dirname(p_dir)}/sel/{stem}.png", np.zeros((2048, 2048), np.uint8))
        return out
    assert len(u) == n, (stem, len(u), n)
    lev = nd.level_idx.to_numpy(); lid = nd.lab_id.to_numpy()
    gts = [cv2.imread(g, cv2.IMREAD_UNCHANGED).astype(np.int64) for g in gt_paths]
    best = np.zeros((n, len(gts)), np.float32); gid = np.zeros((n, len(gts)), np.int32)
    for k, lab in enumerate(r["labels"]):
        vk = np.nonzero(lev == k)[0]
        if not len(vk):
            continue
        for j, iou in enumerate(_iou_with_records(lab.astype(np.int64), int(lab.max()), gts)):
            if iou.shape[1] > 1:
                best[vk, j] = iou[lid[vk]].max(1); gid[vk, j] = iou[lid[vk]].argmax(1)
    assert np.allclose(best, r["ious"], atol=1e-5)
    mask = lambda v: r["labels"][lev[v]] == lid[v]  # noqa: E731
    chosen = select(nd.parent.to_numpy(), u - tau)
    selmap = np.zeros((2048, 2048), np.uint8)             # id map of the stage-1 selection, ids in score order
    for i, v in enumerate(sorted(chosen, key=lambda v: -u[v]), 1):
        out["sel"].append(dict(rle=_enc(mask(v)), u=float(u[v]), iou=best[v].tolist()))
        selmap[mask(v)] = min(i, 255)
    cv2.imwrite(f"{os.path.dirname(p_dir)}/sel/{stem}.png", selmap)
    cs = set(chosen)
    for v in range(n):
        m = [(j, int(gid[v, j])) for j in range(len(gts)) if best[v, j] > 0.5]
        if m and (v in cs or u[v] >= tau - MARGIN):
            out["train"].append(dict(rle=_enc(mask(v)), match=m, u=float(u[v]), sel=v in cs))
    return out


def cmd_seeds(a):
    with open(f"{RAW}/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json") as fh:
        images = pd.DataFrame(json.load(fh)["images"])
    ids = images.groupby(images.file_name.str[:-5]).id.apply(list).to_dict()
    splits = pd.read_csv(f"{ROOT}/data/splits.csv")
    scores = _load(a.scores); tau = a.tau if a.tau is not None else _load(a.scorer)["tau"]
    p_dir = os.path.join(a.out, "p"); os.makedirs(p_dir, exist_ok=True); os.makedirs(os.path.join(a.out, "sel"), exist_ok=True)
    jobs = [(s, [f"{d}/{s}.npy" for d in a.maps], f"{RAW}/train/train_images/{s}.jpeg",
             [f"{ROOT}/data/masks/inst/{i}.png" for i in ids[s]], scores[s], tau, p_dir) for s in splits.stem]
    t0, recs = time.time(), []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_seed_job, jobs, chunksize=2)):
            recs.append(r)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(jobs)} images, {time.time() - t0:.0f}s", flush=True)
    fold = splits.set_index("stem").fold
    for r in recs:
        r["fold"] = int(fold[r["stem"]])
    recs.sort(key=lambda r: r["stem"])
    _dump(dict(tau=float(tau), images=recs), os.path.join(a.out, "seeds.pkl"))
    parts = [_parts([s["iou"] for s in r["sel"]], r["n_gt"]) for r in recs]
    print(f"seeds: {len(recs)} images, {sum(len(r['sel']) for r in recs)} selected (tau {tau}), "
          f"{sum(len(r['train']) for r in recs)} training seeds; stage-1 pooled PQ {pq_of(parts)[0]:.4f} -> {a.out}/seeds.pkl")


def _parts(iou_rows, n_gt):
    """(sum matched IoU, TP, FP, FN) pooled over records; iou_rows[i][r] = best IoU of instance i with record r."""
    v = np.array(iou_rows, np.float32).reshape(len(iou_rows), len(n_gt)); m = v > 0.5
    return float(v[m].sum()), int(m.sum()), int((~m).sum()), int(sum(n_gt) - m.sum())


# ------------------------------------------------------------------------------------------------- model + inference
def build_refiner(encoder="resnet34", weights="imagenet", arch="Unet"):
    import segmentation_models_pytorch as smp
    return getattr(smp, arch)(encoder_name=encoder, encoder_weights=weights, in_channels=4, classes=1)


def load_refiners(paths, device="cuda"):
    import torch
    models = []
    for p in paths:
        ck = torch.load(p, map_location=device)
        m = build_refiner(ck["encoder"], weights=None, arch=ck.get("arch", "Unet")); m.load_state_dict(ck["state_dict"])
        models.append(m.float().to(device).eval())
    return models


def _window(mask):
    """Seed bounding box + PAD, grown to multiples of 32 and clipped to the image."""
    ys, xs = np.nonzero(mask)
    out = []
    for lo, hi in ((ys.min(), ys.max() + 1), (xs.min(), xs.max() + 1)):
        lo, hi = max(lo - PAD, 0), min(hi + PAD, 2048)
        size = min(-(-(hi - lo) // 32) * 32, 2048)
        lo = max(min(lo - (size - (hi - lo)) // 2, 2048 - size), 0)
        out += [lo, lo + size]
    return out


def refine_probs(models, img_n, p, seeds, device="cuda"):
    """q-maps for the seeds of one image (bool masks, pixel-disjoint). Returns [(y0, y1, x0, x1, q float16)]."""
    import torch
    ids = np.zeros((2048, 2048), np.int32)
    for i, m in enumerate(seeds, 1):
        ids[m] = i
    out = []
    for i, m in enumerate(seeds, 1):
        y0, y1, x0, x1 = _window(m)
        w = ids[y0:y1, x0:x1]
        x = np.stack([img_n[y0:y1, x0:x1], p[y0:y1, x0:x1], w == i, (w > 0) & (w != i)]).astype(np.float32)
        x = torch.from_numpy(x)[None].to(device)
        q = 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for f in ((), (-1,), (-2,), (-1, -2)):
                z = x.flip(f) if f else x
                for model in models:
                    o = torch.sigmoid(model(z)).float()
                    q = q + (o.flip(f) if f else o)
        out.append((y0, y1, x0, x1, (q / (4 * len(models)))[0, 0].cpu().numpy().astype(np.float16)))
    return out


def assemble(seeds, probs, t, lo=0.4, hi=3.0):
    """Refined, pixel-disjoint masks. Seeds come in descending score order and take pixels in that order. A refined
    mask keeps only its parts connected to the seed, never enters another seed, and falls back to the seed when
    its area leaves [lo, hi] x the seed area."""
    ids = np.zeros((2048, 2048), np.int32)
    for i, m in enumerate(seeds, 1):
        ids[m] = i
    taken = np.zeros((2048, 2048), bool); out = []
    for i, (m, (y0, y1, x0, x1, q)) in enumerate(zip(seeds, probs), 1):
        w = ids[y0:y1, x0:x1]; seed = w == i
        ref = (q > t) & ~((w > 0) & (w != i))
        n, lab = cv2.connectedComponents(ref.astype(np.uint8), connectivity=8)
        keep = np.unique(lab[seed & ref]); keep = keep[keep > 0]
        ref = np.isin(lab, keep) & ~taken[y0:y1, x0:x1]
        if not lo * seed.sum() <= ref.sum() <= hi * seed.sum():
            ref = seed & ~taken[y0:y1, x0:x1]
        full = np.zeros((2048, 2048), bool); full[y0:y1, x0:x1] = ref
        taken |= full; out.append(full)
    return out


# ------------------------------------------------------------------------------------------------- training
def cmd_train(a):
    import torch, torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    seeds = _load(os.path.join(a.seeds, "seeds.pkl"))["images"]
    p_dir = os.path.join(a.seeds, "p"); sel_dir = os.path.join(a.seeds, "sel")
    items = [(r, s) for r in seeds if r["fold"] != a.fold for s in r["train"]]
    print(f"fold {a.fold}: {len(items)} training seeds from {sum(r['fold'] != a.fold for r in seeds)} images", flush=True)

    class Seeds(Dataset):
        def __len__(self):
            return len(items)

        def __getitem__(self, i):
            r, s = items[i]
            cv2.setNumThreads(1)
            seed = _dec(s["rle"])
            bx, by, bw, bh = cv2.boundingRect(seed.view(np.uint8))
            ys, xs = np.nonzero(seed[by:by + bh, bx:bx + bw]); k = np.random.randint(len(ys)); c = CROP
            cy = by + ys[k] + np.random.randint(-c // 4, c // 4 + 1); cx = bx + xs[k] + np.random.randint(-c // 4, c // 4 + 1)
            y0 = int(np.clip(cy - c // 2, 0, 2048 - c)); x0 = int(np.clip(cx - c // 2, 0, 2048 - c))
            sl = (slice(y0, y0 + c), slice(x0, x0 + c))    # everything below works on the window only
            seed = seed[sl]
            img = load_image(f"{RAW}/train/train_images/{r['stem']}.jpeg")[sl]
            p = cv2.imread(f"{p_dir}/{r['stem']}.png", cv2.IMREAD_UNCHANGED)[sl].astype(np.float32) / 255
            w = cv2.imread(f"{sel_dir}/{r['stem']}.png", cv2.IMREAD_UNCHANGED)[sl]
            others = (w > 0) & ~np.isin(w, np.unique(w[seed]))   # selected instances that do not overlap the seed
            tgt = np.zeros(seed.shape, np.float32)
            for j, g in s["match"]:
                tgt += cv2.imread(r["gt_paths"][j], cv2.IMREAD_UNCHANGED)[sl] == g
            tgt /= len(s["match"])
            im = np.clip(img, 0, 1) ** np.exp(np.random.uniform(-0.2, 0.2))
            im = im * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.03, 0.03)
            x = np.stack([normalize(im, r["stats"]), p, seed, others]).astype(np.float32)
            y = tgt[None]
            if np.random.rand() < 0.5: x, y = x[:, :, ::-1], y[:, :, ::-1]
            if np.random.rand() < 0.5: x, y = x[:, ::-1], y[:, ::-1]
            k = np.random.randint(4); x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k, (1, 2))
            return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))

    torch.manual_seed(a.fold); np.random.seed(a.fold)
    # candidates of one filament are nested and redundant: an epoch is a random draw, stage-1 selections 3x as likely
    wts = torch.tensor([3.0 if s["sel"] else 1.0 for _, s in items])
    sampler = torch.utils.data.WeightedRandomSampler(wts, num_samples=a.per_epoch, replacement=True)
    dl = DataLoader(Seeds(), batch_size=a.bs, sampler=sampler, num_workers=a.workers, drop_last=True, persistent_workers=True)
    model = build_refiner(a.encoder, arch=a.arch).cuda().to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(dl), pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda"); bce = nn.BCEWithLogitsLoss()
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = 0
        for x, y in dl:
            x = x.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last); y = y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                o = model(x).float()
                pr = torch.sigmoid(o)
                dice = 1 - (2 * (pr * y).sum((1, 2, 3)) + 1) / (pr.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
                loss = bce(o, y) + dice.mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        print(f"ep {ep + 1}/{a.epochs} loss {tot / len(dl):.4f} time {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    sd = {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()}
    torch.save({"encoder": a.encoder, "arch": a.arch, "state_dict": sd}, a.out)
    print("saved", a.out)


# ------------------------------------------------------------------------------------------------- evaluation
def _iou_records(masks, gt_paths):
    """best IoU of every mask with every record -> list of rows."""
    if not masks:
        return []
    lab = np.zeros((2048, 2048), np.int64)
    for i, m in enumerate(masks, 1):
        lab[m] = i
    gts = [cv2.imread(g, cv2.IMREAD_UNCHANGED).astype(np.int64) for g in gt_paths]
    mats = _iou_with_records(lab, len(masks), gts)
    return np.stack([m[1:].max(1) if m.shape[1] > 1 else np.zeros(len(masks)) for m in mats], 1).tolist()


def cmd_eval(a):
    seeds = _load(os.path.join(a.seeds, "seeds.pkl"))["images"]
    p_dir = os.path.join(a.seeds, "p")
    ts = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    before, after = [], {t: [] for t in ts}
    import re
    by_fold = {}
    for w in a.weights:                                    # several refiners of one fold are averaged
        by_fold.setdefault(int(re.findall(r"_f(\d+)", w)[-1]), []).append(w)
    for k, ws in sorted(by_fold.items()):
        models = load_refiners(ws)
        recs = [r for r in seeds if r["fold"] == k]
        if a.limit:
            recs = recs[: a.limit]
        t0 = time.time()
        for r in recs:
            b = _parts([s["iou"] for s in r["sel"]], r["n_gt"]); before.append(b + (k,))
            if not r["sel"]:
                for t in ts:
                    after[t].append(b + (k,))
                continue
            img = load_image(f"{RAW}/train/train_images/{r['stem']}.jpeg")
            p = cv2.imread(f"{p_dir}/{r['stem']}.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
            masks = [_dec(s["rle"]) for s in r["sel"]]
            probs = refine_probs(models, normalize(img, r["stats"]), p, masks)
            for t in ts:
                after[t].append(_parts(_iou_records(assemble(masks, probs, t), r["gt_paths"]), r["n_gt"]) + (k,))
        print(f"fold {k}: {len(recs)} images in {time.time() - t0:.0f}s", flush=True)
    folds = sorted({x[4] for x in before})
    line = lambda parts: f"{pq_of(parts)[0]:.4f}  SQ {sum(x[0] for x in parts) / max(sum(x[1] for x in parts), 1):.3f}  TP {sum(x[1] for x in parts)}  | " + " ".join(f"{pq_of([x for x in parts if x[4] == k])[0]:.4f}" for k in folds)  # noqa: E731
    print(f"stage 1          pooled PQ {line(before)}")
    for t in ts:
        print(f"refined, t={t:.2f}  pooled PQ {line(after[t])}")
    if a.json:                                             # per-fold numbers for the report tables
        summ = lambda parts: {**{str(k): pq_of([x for x in parts if x[4] == k])[0] for k in folds}, "all": pq_of(parts)[0],  # noqa: E731
                              "tp": sum(x[1] for x in parts), "fp": sum(x[2] for x in parts), "fn": sum(x[3] for x in parts),
                              "sq": sum(x[0] for x in parts) / max(sum(x[1] for x in parts), 1)}
        with open(a.json, "w") as fh:
            json.dump({"stage1": summ(before), **{f"t{t:.2f}": summ(after[t]) for t in ts}}, fh, indent=1)


# ------------------------------------------------------------------------------------------------- test
def cmd_predict(a):
    models = load_refiners(a.weights)
    sub = pd.read_csv(a.stage1)
    sub["stem"] = sub.filament_id.str.rsplit("_", n=1).str[0]; sub["k"] = sub.filament_id.str.rsplit("_", n=1).str[1].astype(int)
    rows = []
    for stem, g in sub.sort_values(["stem", "k"]).groupby("stem", sort=True):
        img = load_image(f"{RAW}/test/test_images/{stem}.jpeg")
        fams = [np.load(f"{d}/{stem}.npy")[0] for d in a.maps]
        p = (fams[0] if len(fams) == 1 else np.round(np.mean(fams, 0))).astype(np.float32) / 255
        masks = [_dec(c) for c in g.segmentation_rle]                 # stage-1 rows are in descending score order
        out = assemble(masks, refine_probs(models, normalize(img, norm_stats(img, "disk")), p, masks), a.t)
        out = [m for m in out if m.any()]
        rows += [{"filament_id": f"{stem}_{k}", "segmentation_rle": _enc(m)} for k, m in enumerate(out, 1)]
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(a.out, index=False)
    print(f"{len(rows)} refined filaments (stage 1: {len(sub)}) -> {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seeds"); s.add_argument("--maps", required=True, nargs="+"); s.add_argument("--scores", required=True)
    s.add_argument("--scorer", default=None); s.add_argument("--tau", type=float, default=None)
    s.add_argument("--out", required=True); s.add_argument("--workers", type=int, default=6)
    t = sub.add_parser("train"); t.add_argument("--seeds", required=True); t.add_argument("--fold", type=int, required=True)
    t.add_argument("--out", required=True); t.add_argument("--encoder", default="resnet34")
    t.add_argument("--arch", default="Unet", help="segmentation_models_pytorch decoder class, e.g. Unet or UnetPlusPlus")
    t.add_argument("--epochs", type=int, default=10); t.add_argument("--bs", type=int, default=32)
    t.add_argument("--per_epoch", type=int, default=12000, help="training seeds drawn per epoch")
    t.add_argument("--lr", type=float, default=3e-4); t.add_argument("--workers", type=int, default=8)
    e = sub.add_parser("eval"); e.add_argument("--seeds", required=True); e.add_argument("--weights", nargs="+", required=True)
    e.add_argument("--limit", type=int, default=0); e.add_argument("--json", default=None, help="write per-fold PQ here")
    p = sub.add_parser("predict"); p.add_argument("--stage1", required=True); p.add_argument("--maps", required=True, nargs="+")
    p.add_argument("--weights", nargs="+", required=True); p.add_argument("--t", type=float, required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(seeds=cmd_seeds, train=cmd_train, eval=cmd_eval, predict=cmd_predict)[a.cmd](a)


if __name__ == "__main__":
    main()
