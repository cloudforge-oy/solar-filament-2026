"""Shared utilities: image loading, model construction, instance extraction, PQ metric, RLE encoding."""
import os, numpy as np, cv2, torch
import segmentation_models_pytorch as smp
from pycocotools import mask as mutils

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = f"{ROOT}/data/raw/MAGFiLO_1.0_Kaggle_2026"
H = W = 2048
MEAN, STD = 0.30, 0.20  # approximate grayscale statistics of GONG H-alpha jpegs (disk + black corners)


def load_image(path):
    """Load a GONG H-alpha jpeg as float32 grayscale in [0,1]."""
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert im is not None, path
    return im.astype(np.float32) / 255.0


_yy, _xx = np.ogrid[:H, :W]
DISK = (_yy - 1024.0) ** 2 + (_xx - 1023.5) ** 2 <= 880.0 ** 2   # safely inside the solar disk (radius ~900 px)


def norm_stats(img, norm="fixed"):
    """(mean, std) used to standardise a full image: dataset constants ("fixed", the first model generation)
    or the image's own statistics inside the solar disk ("disk"), which removes station/exposure differences."""
    if norm == "disk":
        v = img[DISK]
        return float(v.mean()), max(float(v.std()), 1e-3)
    return MEAN, STD


def normalize(x, stats=(MEAN, STD)):
    return (x - stats[0]) / stats[1]


def build_model(encoder="resnet34", weights="imagenet", classes=3):
    """U-Net with three output channels: [0] filament probability, [1] normalised distance to boundary,
    [2] probability of lying on the contact line between two touching filaments."""
    return smp.Unet(encoder_name=encoder, encoder_weights=weights, in_channels=1, classes=classes)


def masks_to_instances(prob, thr=0.5, min_area=150, open_k=0):
    """Threshold a probability map and split into connected components. Returns int32 label map."""
    binary = (prob > thr).astype(np.uint8)
    if open_k > 0:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(lab)
    k = 0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            k += 1
            out[lab == i] = k
    return out


def masks_to_instances_watershed(prob, dist, thr=0.5, min_area=150, seed_thr=0.5, min_seed=10):
    """Split the foreground mask with a marker-based watershed on the predicted distance map.

    Markers are connected components of (dist > seed_thr) inside the mask; each marker grows over
    the mask following -dist, so two touching filaments with separate medial-axis peaks are split.
    Mask components that received no marker are kept as their own instances.
    """
    from skimage.segmentation import watershed
    binary = prob > thr
    seeds = (dist > seed_thr) & binary
    n, markers, stats, _ = cv2.connectedComponentsWithStats(seeds.astype(np.uint8), connectivity=8)
    small = np.isin(markers, np.nonzero(stats[:, cv2.CC_STAT_AREA] < min_seed)[0])
    markers[small] = 0
    lab = watershed(-dist, markers=markers, mask=binary) if markers.max() > 0 else np.zeros(prob.shape, np.int32)
    leftover = binary & (lab == 0)
    if leftover.any():
        n2, lab2 = cv2.connectedComponents(leftover.astype(np.uint8), connectivity=8)
        lab = lab.astype(np.int32); lab[leftover] = lab2[leftover] + lab.max()
    # relabel, dropping small instances
    out = np.zeros(prob.shape, np.int32); k = 0
    ids, counts = np.unique(lab, return_counts=True)
    for i, c in zip(ids, counts):
        if i == 0 or c < min_area:
            continue
        k += 1; out[lab == i] = k
    return out


def masks_to_instances_split(prob, touch, thr=0.5, min_area=150, touch_thr=0.5):
    """Connected components of the mask with predicted contact lines removed, then the removed pixels
    are given back to the nearest component (watershed on the mask with the components as markers)."""
    from skimage.segmentation import watershed
    binary = prob > thr
    cut = binary & (touch > touch_thr)
    n, markers = cv2.connectedComponents((binary & ~cut).astype(np.uint8), connectivity=8)
    if cut.any() and n > 1:
        lab = watershed(np.zeros(prob.shape, np.float32), markers=markers, mask=binary)
    else:
        lab = markers
    out = np.zeros(prob.shape, np.int32); k = 0
    for i, c in zip(*np.unique(lab, return_counts=True)):
        if i and c >= min_area:
            k += 1; out[lab == i] = k
    return out


def panoptic_quality(gt, pred):
    """PQ for one image. gt, pred: int label maps (0 = background). Returns (pq, tp, fp, fn, sum_iou)."""
    ng, npred = gt.max(), pred.max()
    if ng == 0 and npred == 0:
        return 1.0, 0, 0, 0, 0.0
    if ng == 0 or npred == 0:
        return 0.0, 0, int(npred), int(ng), 0.0
    inter = np.zeros((ng + 1, npred + 1), np.int64)
    np.add.at(inter, (gt.ravel(), pred.ravel()), 1)
    inter = inter[1:, 1:]
    ag = np.bincount(gt.ravel(), minlength=ng + 1)[1:][:, None]
    ap = np.bincount(pred.ravel(), minlength=npred + 1)[1:][None, :]
    iou = inter / (ag + ap - inter)
    match = iou > 0.5  # at most one match per row/col is guaranteed when IoU > 0.5
    tp = int(match.sum()); sum_iou = float(iou[match].sum())
    fp = int(npred - tp); fn = int(ng - tp)
    return sum_iou / (tp + 0.5 * fp + 0.5 * fn), tp, fp, fn, sum_iou


def encode_rle(binary):
    """COCO RLE counts string for a 2048x2048 uint8 mask."""
    return mutils.encode(np.asfortranarray(binary.astype(np.uint8)))["counts"].decode("ascii")
