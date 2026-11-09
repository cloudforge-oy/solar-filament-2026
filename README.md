# Solar Filament Segmentation Challenge 2026

Instance segmentation of solar filaments in GONG H-alpha images (MAGFiLO dataset) for the Kaggle
competition https://www.kaggle.com/competitions/filament-segmentation-2026.

The method and results are described in `report/main.tex` (organizers' template). The notebook
`notebooks/pipeline.ipynb` reproduces the submitted predictions from the stored weights in `weights/`.

## Method in one paragraph

A U-Net (ImageNet-pretrained EfficientNet-B3 encoder, one grayscale input channel) predicts three maps at native
2048x2048 resolution: the filament probability, a normalised distance-to-boundary map and the contact line between
touching filaments. Supervision uses *consensus* targets: the fraction of the 1-3 annotators of an image who marked
each pixel. Five models, one per date-grouped fold, are averaged with flip test-time augmentation. The averaged map is
turned into instances by a **decision layer**: thresholding at 16 levels gives a nested forest of candidate
components; a gradient-boosted hurdle model predicts each candidate's expected matched IoU against a random annotator
(u = P(match) x E[IoU | match]); an exact dynamic programme selects the pixel-disjoint set of candidates that
maximises sum(u - tau), which maximises the host's pooled Panoptic Quality (Dinkelbach). Each selected component is one
COCO RLE row of the submission.

| (out-of-fold, all 707 training images) | pooled PQ |
|---|---|
| plain rule: components of p > 0.6, area >= 400 | 0.405 |
| decision layer, tau = 0.26 | **0.431** (gain on every fold) |
| public leaderboard: plain rule / decision layer | 0.35 / **0.37** |

**Credit.** The decision layer (pooled-PQ reduction, component tree, hurdle scorer, tree DP, tau = PQ*/2) follows the
public Kaggle notebook *"Deciding under disagreement - a PQ-optimal detect-then-refine cascade for solar filaments
(filseg v4)"* by _.tushhaaarrrr
(https://www.kaggle.com/code/tushhaaarrrr/solar-filament-segmentation-2026-0-40). It was re-implemented here from the
description, on our own maps, descriptors and scorer; no code, weights or predictions were copied.

## Layout

| path | purpose |
|---|---|
| `src/prepare.py` | rasterise COCO polygons to instance maps, build consensus targets, make date-grouped folds |
| `src/common.py` | model definition, instance extraction (components / contact-line split / watershed), PQ metric, RLE |
| `src/train.py` | training loop (BCE + Dice + distance MSE + weighted contact BCE), full-frame validation |
| `src/tune.py` | grid search of post-processing parameters on validation PQ |
| `src/evaluate.py` | single-model and ensemble evaluation with standard errors and bootstrap intervals |
| `src/predict.py` | test-set inference and ensembling with the plain rule (also provides model loading) |
| `src/predict_maps.py` | caches out-of-fold maps (per fold model) and 5-fold mean test maps |
| `src/decision.py` | decision layer: candidate bank, cross-fitted scorer, exact tree selection, final submission |
| `src/rethreshold.py` | re-runs the plain rule on cached probability maps |
| `src/check_submission.py` | validates a CSV: RLE decodability, unique ids, no overlapping masks |
| `src/export_weights.py` | writes compact half-precision checkpoints into `weights/` |
| `scripts/train_folds.sh` | trains one encoder on several folds sequentially |
| `scripts/make_*_figures.py`, `scripts/make_*_table.py` | figures and tables for the report |
| `notebooks/pipeline.ipynb` | end-to-end illustration: data, model, validation, submission |
| `report/` | 4-page technical report (LaTeX, organizers' acmart template) |

## Reproduce the submission

```bash
pip install -r requirements.txt
# download and unzip the Kaggle data into data/raw/MAGFiLO_1.0_Kaggle_2026 (train/ and test/)
python src/predict_maps.py test --weights weights/effb3d_f*.pt --out runs/maps/test_effb3d     # ~5 s/image on GPU
python src/decision.py predict --maps runs/maps/test_effb3d --model weights/decision_scorer.pkl --tau 0.26 --out submission.csv
python src/check_submission.py submission.csv
```
`weights/params.json` records these settings. The submitted file is `report/submission_final.csv`.

## Train from scratch

```bash
python src/prepare.py
scripts/train_folds.sh efficientnet-b3 effb3d "0 1 2 3 4" --epochs 50 --crop 640 --bs 6 --eval_every 10
for k in 0 1 2 3 4; do python src/export_weights.py runs/effb3d_f$k weights/effb3d_f$k.pt; done
python src/predict_maps.py oof --weights weights/effb3d_f*.pt --out runs/maps/oof_effb3d          # out-of-fold maps
python src/decision.py bank --maps runs/maps/oof_effb3d --out runs/decision/bank_effb3d.pkl     # ~5 min, 6 CPU workers
python src/decision.py fit --bank runs/decision/bank_effb3d.pkl --out weights/decision_scorer.pkl  # cross-fit + tau curve
python scripts/make_decision_table.py 0.26 && python scripts/make_report_figures.py 0.26
```
`fit` prints the cross-fitted pooled PQ for tau in 0.12-0.40; the plateau is 0.26-0.32. tau = 0.26 was submitted
because the 5-fold mean test maps are slightly less confident than single-model out-of-fold maps, and 0.26 restores
the out-of-fold selection rate (6.3 vs 6.4 filaments per image). The ResNet34 comparison models (`resnet34d`) are
trained the same way with `--epochs 40 --eval_every 5` and evaluated with `src/evaluate.py` / `scripts/cv_evaluate.sh`.

## Data notes

707 training images, 1154 annotation sets (296 images annotated by 2-3 annotators), 8199 filaments,
median filament area 1228 px; only 0.9 % of filaments touch another filament within 3 px. Images are
2048x2048 8-bit grayscale JPEGs; file names encode time and observatory. Only the competition data is
used for training; the public MAGFiLO release overlaps the test set and is deliberately not used.
