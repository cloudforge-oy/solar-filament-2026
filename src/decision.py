"""PQ-optimal instance selection on a component tree (the "decision layer").

Idea (credited to the public Kaggle notebook "Deciding under disagreement - a PQ-optimal detect-then-refine
cascade" by _.tushhaaarrrr, filseg v4): the host pools TP/FP/FN and matched IoUs over every annotator record, so

    PQ(P) = sum_i R_i sum_{c in P_i} u_i(c) / (0.5 sum_i R_i |P_i| + 0.5 N_G),
    u_i(c) = mean over the R_i records of IoU(c, g_r(c)) * 1{IoU > 0.5}.

By Dinkelbach's theorem the optimal submission maximises sum_c (u(c) - tau) with one global tau = PQ*/2.
Candidates are the connected components of the probability map thresholded at 16 levels (a nested forest);
every antichain of the forest is a valid pixel-disjoint submission, and the best antichain is found exactly by
a bottom-up dynamic programme. u(c) is unknown at test time, so a gradient-boosted "hurdle" model learns
u = P(an annotator record matches c) * E[IoU | match] from out-of-fold candidates.

Subcommands
  bank     build candidate trees + descriptors + per-record IoUs for all training images from out-of-fold maps
  fit      cross-fit the scorer by fold, report pooled PQ against baselines, pick tau, fit the final scorer
  predict  apply the final scorer + DP to cached test maps and write the submission CSV
"""
import argparse, json, os, pickle, sys, time
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, RAW, encode_rle  # noqa: E402

LEVELS = np.round(np.arange(0.05, 0.801, 0.05), 2)       # 16 threshold levels of the filament probability
MIN_NODE_AREA = 100                                       # candidates below this many pixels are never useful
CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
RING_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
DISK_CX, DISK_CY, DISK_R = 1023.5, 1024.0, 900.0          # GONG frames have a fixed disk geometry
STATIONS = ["Bh", "Ch", "Lh", "Mh", "Th", "Uh"]
PAD = 10
FEATURES = [
    "level", "log_area", "log_bbox_w", "log_bbox_h", "log_long", "log_short", "elong", "solidity", "fill",
    "p_mean", "p_max", "p_std", "p_q10", "p_q50", "p_q90", "frac_p50", "p_boundary",
    "dist_max", "dist_mean", "touch_max", "touch_mean",
    "rel_mean", "rel_q10", "rel_ring", "contrast_ring",
    "rho", "abs_lat", "n_children", "area_ratio_parent", "n_siblings", "parent_level_gap",
    "img_n_cc50", "img_fg50", "img_rel_std", "station",
]
CATEGORICAL = ["station"]


# ------------------------------------------------------------------------------------------------- tree building
def _levels(p):
    """Nested binary masks: threshold, 5x5 closing, hole filling (all increasing, so nesting is exact)."""
    for t in LEVELS:
        b = cv2.morphologyEx((p > t).astype(np.uint8), cv2.MORPH_CLOSE, CLOSE_KERNEL)
        yield t, ndimage.binary_fill_holes(b).astype(np.uint8)


def _iou_with_records(lab, n_lab, gts):
    """IoU matrix (n_lab, n_gt_r) for each record's instance map."""
    fg = lab > 0
    area_l = np.bincount(lab[fg], minlength=n_lab + 1).astype(np.int64)
    out = []
    for g in gts:
        ng = int(g.max())
        if ng == 0:
            out.append(np.zeros((n_lab + 1, 1))); continue
        area_g = np.bincount(g.ravel(), minlength=ng + 1).astype(np.int64)
        inter = np.bincount(lab[fg].astype(np.int64) * (ng + 1) + g[fg], minlength=(n_lab + 1) * (ng + 1))
        inter = inter.reshape(n_lab + 1, ng + 1)
        union = area_l[:, None] + area_g[None, :] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
        iou[0, :] = 0; iou[:, 0] = 0
        out.append(iou)
    return out


def process_image(stem, maps_path, img_path, gt_paths=(), keep_labels=False):
    """Build the candidate forest of one image.

    Returns dict(nodes=DataFrame of descriptors + tree links, ious=(n_nodes, R) best IoU per record,
    n_gt=[|G_r|], baseline=per-record parts of the plain rule (CC of p > 0.6, area >= 400), labels=[uint16 maps])."""
    cv2.setNumThreads(1)
    maps = np.load(maps_path)
    p = maps[0].astype(np.float32) / 255
    dist, touch = maps[1], maps[2]
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    bg = cv2.GaussianBlur(img, (0, 0), 40)
    rel = img / np.maximum(bg, 1.0)                         # relative intensity; filaments are < 1
    yy, xx = np.ogrid[:2048, :2048]
    disk = (xx - DISK_CX) ** 2 + (yy - DISK_CY) ** 2 < (0.98 * DISK_R) ** 2
    gts = [cv2.imread(g, cv2.IMREAD_UNCHANGED).astype(np.int64) for g in gt_paths]
    n_gt = [int(g.max()) for g in gts]
    station = float(STATIONS.index(stem[-2:])) if stem[-2:] in STATIONS else np.nan   # NaN = unknown category

    # baseline: the rule of our earlier submissions
    n, lab, st, _ = cv2.connectedComponentsWithStats((p > 0.6).astype(np.uint8), connectivity=8)
    small = np.nonzero(st[:, 4] < 400)[0]
    lab[np.isin(lab, small)] = 0
    keep_ids = np.unique(lab[lab > 0])
    baseline = []
    if gts:
        for iou in _iou_with_records(lab, n - 1, gts):
            best = iou[keep_ids].max(1) if len(keep_ids) and iou.shape[1] > 1 else np.zeros(len(keep_ids))
            m = best > 0.5
            baseline.append((float(best[m].sum()), int(m.sum()), int(len(keep_ids) - m.sum())))

    img_n_cc50 = img_fg50 = 0.0
    img_rel_std = float(rel[disk].std())
    rows, ious, labels = [], [], []
    prev_lab, prev_index = None, {}
    for k, (t, b) in enumerate(_levels(p)):
        n, lab, st, cen = cv2.connectedComponentsWithStats(b, connectivity=8)
        if keep_labels:
            labels.append(lab.astype(np.uint16))
        if abs(t - 0.5) < 1e-6:
            img_n_cc50 = float((st[1:, 4] >= MIN_NODE_AREA).sum()); img_fg50 = float(b[disk].mean())
        rec_iou = _iou_with_records(lab, n - 1, gts) if gts else []
        index = {}
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in st[i])
            if area < MIN_NODE_AREA:
                continue
            y0, y1, x0, x1 = max(y - PAD, 0), min(y + h + PAD, 2048), max(x - PAD, 0), min(x + w + PAD, 2048)
            m = lab[y0:y1, x0:x1] == i
            pc = p[y0:y1, x0:x1]; pv = pc[m]
            m8 = m.astype(np.uint8)
            edge = m & ~cv2.erode(m8, np.ones((3, 3), np.uint8)).astype(bool)
            ring = cv2.dilate(m8, RING_KERNEL).astype(bool) & ~m & (pc < 0.3)
            rc = rel[y0:y1, x0:x1]; rv = rc[m]
            cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pts = np.vstack(cnts)
            hull = cv2.contourArea(cv2.convexHull(pts))
            (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
            long_, short_ = max(rw, rh, 1.0), max(min(rw, rh), 1.0)
            cx, cy = cen[i]
            parent = -1
            if prev_lab is not None:
                yy0, xx0 = np.argwhere(m)[0]
                parent = prev_index.get(int(prev_lab[y0 + yy0, x0 + xx0]), -1)
            ring_mean = float(rc[ring].mean()) if ring.any() else float(rv.mean())
            rows.append(dict(
                level=float(t), level_idx=k, lab_id=i, parent=parent,
                log_area=np.log(area), log_bbox_w=np.log(w), log_bbox_h=np.log(h),
                log_long=np.log(long_), log_short=np.log(short_), elong=long_ / short_,
                solidity=area / max(hull, 1.0), fill=area / float(w * h),
                p_mean=float(pv.mean()), p_max=float(pv.max()), p_std=float(pv.std()),
                p_q10=float(np.quantile(pv, 0.1)), p_q50=float(np.quantile(pv, 0.5)), p_q90=float(np.quantile(pv, 0.9)),
                frac_p50=float((pv > 0.5).mean()), p_boundary=float(pc[edge].mean()) if edge.any() else float(pv.mean()),
                dist_max=float(dist[y0:y1, x0:x1][m].max()) / 255, dist_mean=float(dist[y0:y1, x0:x1][m].mean()) / 255,
                touch_max=float(touch[y0:y1, x0:x1][m].max()) / 255, touch_mean=float(touch[y0:y1, x0:x1][m].mean()) / 255,
                rel_mean=float(rv.mean()), rel_q10=float(np.quantile(rv, 0.1)), rel_ring=ring_mean,
                contrast_ring=float(rv.mean()) - ring_mean,
                rho=float(np.hypot(cx - DISK_CX, cy - DISK_CY) / DISK_R), abs_lat=float(abs(cy - DISK_CY) / DISK_R),
                area=area,
            ))
            index[i] = len(rows) - 1
            if gts:
                ious.append([float(r[i].max()) if r.shape[1] > 1 else 0.0 for r in rec_iou])
        prev_lab, prev_index = lab, index

    nodes = pd.DataFrame(rows)
    if len(nodes):
        par = nodes.parent.to_numpy()
        n_children = np.bincount(par[par >= 0], minlength=len(nodes))
        nodes["n_children"] = n_children
        nodes["n_siblings"] = [n_children[q] if q >= 0 else -1 for q in par]
        areas = nodes.area.to_numpy()
        nodes["area_ratio_parent"] = [areas[v] / areas[q] if q >= 0 else 1.0 for v, q in enumerate(par)]
        nodes["parent_level_gap"] = [0.0 if q < 0 else float(areas[q] != areas[v]) for v, q in enumerate(par)]
        nodes["img_n_cc50"], nodes["img_fg50"], nodes["img_rel_std"] = img_n_cc50, img_fg50, img_rel_std
        nodes["station"] = station
    return dict(stem=stem, nodes=nodes, ious=np.array(ious, np.float32).reshape(len(nodes), len(gts)),
                n_gt=n_gt, baseline=baseline, labels=labels)


# ------------------------------------------------------------------------------------------------- selection + metric
def select(parent, score):
    """Exact max of sum(score) over antichains of the forest. Parents must precede children in the arrays."""
    n = len(parent)
    children = [[] for _ in range(n)]
    for v in range(n):
        if parent[v] >= 0:
            children[parent[v]].append(v)
    best = np.zeros(n); take = np.zeros(n, np.int8)
    for v in range(n - 1, -1, -1):
        sc = sum(best[w] for w in children[v])
        opts = (0.0, float(score[v]), sc)
        j = int(np.argmax(opts)); best[v] = opts[j]; take[v] = j
    chosen, stack = [], [v for v in range(n) if parent[v] < 0]
    while stack:
        v = stack.pop()
        if take[v] == 1:
            chosen.append(v)
        elif take[v] == 2:
            stack.extend(children[v])
    return sorted(chosen)


def _choose(r, u, tau):
    return select(r["nodes"].parent.to_numpy(), u - tau) if len(r["nodes"]) else []


def pooled_parts(ious, n_gt, chosen):
    """(sum matched IoU, TP, FP, FN) pooled over the records of one image for a disjoint selection."""
    s = tp = fp = fn = 0
    for r, ng in enumerate(n_gt):
        v = ious[chosen, r] if len(chosen) else np.zeros(0)
        m = v > 0.5
        s += float(v[m].sum()); tp += int(m.sum()); fp += int(len(chosen) - m.sum()); fn += int(ng - m.sum())
    return s, tp, fp, fn


def pq_of(parts):
    s, tp, fp, fn = (sum(x[i] for x in parts) for i in range(4))
    return s / max(tp + 0.5 * fp + 0.5 * fn, 1e-9), tp, fp, fn


# ------------------------------------------------------------------------------------------------- subcommands
def _load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _dump(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)


def _bank_job(args):
    stem, maps_path, img_path, gt_paths = args
    r = process_image(stem, maps_path, img_path, gt_paths)
    r.pop("labels")
    return r


def cmd_bank(a):
    with open(f"{RAW}/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json") as fh:
        images = pd.DataFrame(json.load(fh)["images"])
    splits = pd.read_csv(f"{ROOT}/data/splits.csv")
    jobs = []
    for s in splits.stem:
        ids = images[images.file_name == s + ".jpeg"].id.tolist()
        jobs.append((s, f"{a.maps}/{s}.npy", f"{RAW}/train/train_images/{s}.jpeg",
                     [f"{ROOT}/data/masks/inst/{i}.png" for i in ids]))
    fold_of = splits.set_index("stem").fold
    jobs = [j for j in jobs if fold_of[j[0]] in a.folds]
    if a.limit:
        jobs = jobs[: a.limit]
    t0, bank = time.time(), []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_bank_job, jobs, chunksize=2)):
            bank.append(r)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(jobs)} images, {time.time() - t0:.0f}s", flush=True)
    fold = splits.set_index("stem").fold
    for r in bank:
        r["fold"] = int(fold[r["stem"]])
    _dump(bank, a.out)
    n = sum(len(r["nodes"]) for r in bank)
    print(f"bank: {len(bank)} images, {n} candidates ({n / len(bank):.0f}/image), {time.time() - t0:.0f}s -> {a.out}")


def _fit_scorer(X, Y, R_rows, seed=0):
    """Hurdle scorer: h = P(record matches) on (candidate, record) rows; m = E[IoU | match] on matched rows."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    cat = [FEATURES.index(c) for c in CATEGORICAL]
    kw = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
              categorical_features=cat, early_stopping=False, random_state=seed)
    Xr = X[R_rows]; Yr = Y
    h = HistGradientBoostingClassifier(**kw).fit(Xr, (Yr > 0.5).astype(np.int8))
    mm = Yr > 0.5
    m = HistGradientBoostingRegressor(**kw).fit(Xr[mm], Yr[mm])
    return dict(h=h, m=m)


def _score(model, X):
    return model["h"].predict_proba(X)[:, 1] * np.clip(model["m"].predict(X), 0.5, 1.0)


def _design(bank):
    X = np.concatenate([r["nodes"][FEATURES].to_numpy(np.float32) for r in bank if len(r["nodes"])])
    rows, Y, offs, off = [], [], [], 0
    for r in bank:
        n = len(r["nodes"]); offs.append(off)
        for j in range(r["ious"].shape[1]):
            rows.append(np.arange(off, off + n)); Y.append(r["ious"][:, j])
        off += n
    return X, np.concatenate(rows), np.concatenate(Y), offs


def cmd_fit(a):
    bank = _load(a.bank)                                   # all images, including those without candidates
    allbank = bank
    print(f"{len(bank)} images, {sum(len(r['nodes']) for r in bank)} candidates")
    # baselines
    base = pq_of([p for r in allbank for p in [(sum(b[0] for b in r["baseline"]), sum(b[1] for b in r["baseline"]),
                                                  sum(b[2] for b in r["baseline"]), sum(r["n_gt"]) - sum(b[1] for b in r["baseline"]))]])
    print(f"baseline  CC p>0.6, area>=400 (current submission rule): pooled PQ {base[0]:.4f}  TP {base[1]} FP {base[2]} FN {base[3]}")
    fixed = []
    for k, t in enumerate(LEVELS):
        for A in (250, 400, 550, 700):
            parts = []
            for r in allbank:
                nd = r["nodes"]
                ch = np.nonzero((nd.level_idx.to_numpy() == k) & (nd.area.to_numpy() >= A))[0] if len(nd) else []
                parts.append(pooled_parts(r["ious"], r["n_gt"], ch))
            fixed.append((pq_of(parts)[0], t, A))
    fb = max(fixed)
    print(f"best fixed rule on the tree (closing + hole fill): pooled PQ {fb[0]:.4f} at t={fb[1]}, A>={fb[2]}")
    oracle = pq_of([pooled_parts(r["ious"], r["n_gt"], _choose(r, (r["ious"] * (r["ious"] > 0.5)).mean(1), 0.2))
                    for r in bank])
    print(f"oracle (true u, tau 0.2): pooled PQ {oracle[0]:.4f}  -> ceiling of the candidate trees")

    # cross-fitted scores
    scores = {}
    for k in sorted({r["fold"] for r in bank}):
        tr = [r for r in bank if r["fold"] != k]; te = [r for r in bank if r["fold"] == k]
        X, rows, Y, _ = _design(tr)
        t0 = time.time(); model = _fit_scorer(X, Y, rows)
        for r in te:
            scores[r["stem"]] = _score(model, r["nodes"][FEATURES].to_numpy(np.float32)) if len(r["nodes"]) else np.zeros(0)
        print(f"  fold {k}: scorer fitted on {len(rows)} rows in {time.time() - t0:.0f}s", flush=True)
    _dump(scores, os.path.join(os.path.dirname(os.path.abspath(a.bank)), "oof_scores.pkl"))   # for report tables
    res = []
    for tau in np.round(np.arange(0.12, 0.42, 0.02), 2):
        parts, perfold = [], {}
        for r in bank:
            ch = _choose(r, scores[r["stem"]], tau)
            pp = pooled_parts(r["ious"], r["n_gt"], ch); parts.append(pp); perfold.setdefault(r["fold"], []).append(pp)
        q = pq_of(parts)
        res.append(dict(tau=float(tau), pq=q[0], tp=q[1], fp=q[2], fn=q[3], **{f"f{k}": pq_of(v)[0] for k, v in perfold.items()}))
    res = pd.DataFrame(res); print(res.round(4).to_string(index=False))
    best = res.loc[res.pq.idxmax()]
    print(f"decision layer: pooled OOF PQ {best.pq:.4f} at tau {best.tau} (baseline {base[0]:.4f}, best fixed {fb[0]:.4f})")
    # final scorer on all candidates
    X, rows, Y, _ = _design(bank)
    model = _fit_scorer(X, Y, rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    _dump(dict(model=model, tau=float(best.tau), features=FEATURES, levels=LEVELS.tolist(),
               min_node_area=MIN_NODE_AREA, oof=res.to_dict("records"), baseline=base[0], best_fixed=fb[:3]), a.out)
    print("saved", a.out)


_WORKER = {}


def _test_init(model_path, tau=None):
    _WORKER["d"] = _load(model_path)                       # once per worker process
    if tau is not None:
        _WORKER["d"]["tau"] = float(tau)


def _test_job(args):
    stem, maps_path, img_path = args
    d = _WORKER["d"]
    r = process_image(stem, maps_path, img_path, keep_labels=True)
    nd = r["nodes"]
    if not len(nd):
        return stem, []
    u = _score(d["model"], nd[FEATURES].to_numpy(np.float32))
    ch = select(nd.parent.to_numpy(), u - d["tau"])
    out = []
    for v in ch:
        mask = r["labels"][int(nd.level_idx.iloc[v])] == int(nd.lab_id.iloc[v])
        out.append((float(u[v]), encode_rle(mask)))
    out.sort(key=lambda x: -x[0])
    return stem, [rle for _, rle in out]


def cmd_predict(a):
    import glob
    paths = sorted(glob.glob(f"{RAW}/test/test_images/*.jpeg"))
    jobs = [(os.path.basename(p)[:-5], f"{a.maps}/{os.path.basename(p)[:-5]}.npy", p) for p in paths]
    rows = []
    with Pool(a.workers, initializer=_test_init, initargs=(a.model, a.tau)) as pool:
        for stem, rles in pool.imap(_test_job, jobs):
            rows += [{"filament_id": f"{stem}_{k}", "segmentation_rle": r} for k, r in enumerate(rles, 1)]
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(a.out, index=False)
    print(f"{len(rows)} filaments on {len({r['filament_id'].rsplit('_', 1)[0] for r in rows})} images -> {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bank"); b.add_argument("--maps", required=True); b.add_argument("--out", required=True)
    b.add_argument("--workers", type=int, default=6); b.add_argument("--limit", type=int, default=0)
    b.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    f = sub.add_parser("fit"); f.add_argument("--bank", required=True); f.add_argument("--out", required=True)
    p = sub.add_parser("predict"); p.add_argument("--maps", required=True); p.add_argument("--model", required=True)
    p.add_argument("--out", required=True); p.add_argument("--workers", type=int, default=4)
    p.add_argument("--tau", type=float, default=None, help="override the selection threshold stored with the scorer")
    a = ap.parse_args()
    dict(bank=cmd_bank, fit=cmd_fit, predict=cmd_predict)[a.cmd](a)


if __name__ == "__main__":
    main()
