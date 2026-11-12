# Solar Filament Segmentation Challenge 2026

Instance segmentation of solar filaments in GONG H-alpha images (MAGFiLO dataset) for the Kaggle
competition https://www.kaggle.com/competitions/filament-segmentation-2026.

The method and results are described in `report/main.tex` (organizers' template). The notebook
`notebooks/pipeline.ipynb` reproduces the submitted predictions from the stored weights in `weights/`.

## Method in one paragraph

U-Nets (ImageNet-pretrained EfficientNetV2-S and EfficientNet-B3 encoders, one grayscale input channel) predict three
maps at native 2048x2048 resolution: the filament probability, a normalised distance-to-boundary map and the contact
line between touching filaments. Supervision uses *consensus* targets: the fraction of the 1-3 annotators of an image
who marked each pixel. Ten models (two encoders x five date-grouped folds) are averaged over eight flip/rotation
views. The averaged map is turned into instances in two learned stages. **Stage 1, the decision layer:** thresholding
at 16 levels gives a nested forest of candidate components; a gradient-boosted hurdle model predicts each candidate's
expected matched IoU against a random annotator (u = P(match) x E[IoU | match]) from 67 descriptors computed from the image and the predicted maps only (nothing from the file name); an exact dynamic
programme selects the pixel-disjoint set of candidates that maximises sum(u - tau), which maximises the host's pooled
Panoptic Quality (Dinkelbach). **Stage 2, the refiner:** a UNet++ re-segments every selected instance from a
native-resolution window (image, probability, the instance, the other selected instances) and predicts the outline an
annotator would draw given that they drew this filament. Each refined instance is one COCO RLE row of the submission.

| pipeline (out-of-fold, all 707 training images) | pooled PQ | public LB |
|---|---|---|
| EfficientNet-B3, flip TTA, plain rule (components of p > 0.6, area >= 400) | 0.405 | 0.35 |
| + decision layer, 35 descriptors | 0.431 | 0.37 |
| + 8-view TTA, tree-context descriptors | 0.439 | 0.38 |
| + four averaged scorers | 0.441 | |
| + EfficientNetV2-S family | 0.442 | |
| - file-name descriptors (site, year), not allowed at inference | 0.440 | |
| + instance refiner (final) | **0.447** | **0.39** |

For scale: one annotator scored against another on the 296 multi-annotated training images gets a pooled PQ of 0.340.
Every stage is cross-fitted with the same folds, so the numbers are out-of-fold end to end.

**Credit.** The decision layer (pooled-PQ reduction, component tree, hurdle scorer, tree DP, tau = PQ*/2) and the idea
of an instance refiner follow the public Kaggle notebook *"Deciding under disagreement - a PQ-optimal detect-then-refine
cascade for solar filaments (filseg v4)"* by _.tushhaaarrrr
(https://www.kaggle.com/code/tushhaaarrrr/solar-filament-segmentation-2026-0-40). Both were re-implemented here from
the description, on our own maps, descriptors, scorer and refiner design; no code, weights or predictions were copied.

## Layout

| path | purpose |
|---|---|
| `src/prepare.py` | rasterise COCO polygons to instance maps, build consensus targets, make date-grouped folds |
| `src/common.py` | model definition, input standardisation, plain instance extraction, PQ metric, RLE |
| `src/train.py` | dense-model training loop, full-frame prediction with test-time augmentation |
| `src/predict_maps.py` | caches out-of-fold maps (per fold model) and fold-mean test maps |
| `src/decision.py` | stage 1: candidate bank, cross-fitted scorer, exact tree selection |
| `src/refine.py` | stage 2: training seeds, refiner training, out-of-fold evaluation, test refinement |
| `src/check_submission.py` | validates a CSV: RLE decodability, unique ids, no overlapping masks |
| `src/export_weights.py` | writes compact half-precision checkpoints into `weights/` |
| `src/predict.py`, `src/tune.py`, `src/evaluate.py`, `src/rethreshold.py` | the plain-rule baseline and its evaluation (model loading lives in `predict.py`) |
| `scripts/predict_test.sh` | the whole inference chain, weights -> submission CSV |
| `scripts/train_folds.sh` | trains one encoder on several folds sequentially |
| `scripts/make_*_figure*.py`, `scripts/make_decision_table.py` | figures and tables for the report |
| `notebooks/pipeline.ipynb` | end-to-end illustration: data, dense maps, both stages, submission |
| `report/` | technical report (LaTeX, organizers' acmart template) and the submitted CSV |

## Reproduce the submission

```bash
pip install -r requirements.txt
# download and unzip the Kaggle data into data/raw/MAGFiLO_1.0_Kaggle_2026 (train/ and test/)
scripts/predict_test.sh submission.csv        # GPU, about 50 minutes on a laptop RTX 5060
```
The script caches the dense maps of both encoder families (8 views x 5 fold models each), runs the decision layer with
`--match_rate`, refines the selected instances and validates the CSV. `weights/params.json` records the settings. The
submitted file is `report/submission_final.csv`. GPU arithmetic is not bit-reproducible across hardware: maps can
differ by one 8-bit level, which may move a few borderline instances.

`--match_rate`: the scorer is fitted on out-of-fold maps, where every image is predicted by one model per family; test
maps average five and are smoother, so the same tau keeps fewer instances. The option sets tau, without labels, to the
value that restores the cross-validated number of instances per image stored with the scorer.

## Train from scratch

```bash
python src/prepare.py
scripts/train_folds.sh efficientnet-b3 effb3d "0 1 2 3 4" --epochs 50 --crop 640 --bs 6 --eval_every 10 --norm fixed
scripts/train_folds.sh tu-tf_efficientnetv2_s effv2s "0 1 2 3 4" --epochs 75 --crop 512 --bs 16 --eval_every 25 --norm disk
for k in 0 1 2 3 4; do
  python src/export_weights.py runs/effb3d_f$k weights/effb3d_f$k.pt
  python src/export_weights.py runs/effv2s_f$k weights/effv2s_f$k.pt --ckpt last.pt
done
for f in effv2s effb3d; do python src/predict_maps.py oof --weights weights/${f}_f*.pt --out runs/maps/oof_$f --tta 8; done
# stage 1 (CPU): candidate bank ~6 min, cross-fit + tau curve ~10 min
python src/decision.py bank --maps runs/maps/oof_effv2s runs/maps/oof_effb3d --out runs/decision/bank_v2s_b3.pkl
python src/decision.py fit --bank runs/decision/bank_v2s_b3.pkl --out runs/decision/scorer_v2s_b3.pkl
cp runs/decision/scorer_v2s_b3.pkl weights/decision_scorer.pkl
# stage 2 (GPU): seeds from the out-of-fold selections, one refiner per fold, out-of-fold evaluation
python src/refine.py seeds --maps runs/maps/oof_effv2s runs/maps/oof_effb3d \
    --scores runs/decision/scorer_v2s_b3_oof_scores.pkl --scorer runs/decision/scorer_v2s_b3.pkl --out runs/refine/v2s_b3
for k in 0 1 2 3 4; do python src/refine.py train --seeds runs/refine/v2s_b3 --fold $k --arch UnetPlusPlus --out weights/refiner_f$k.pt; done
python src/refine.py eval --seeds runs/refine/v2s_b3 --weights weights/refiner_f*.pt --json runs/refine/v2s_b3/eval.json
python scripts/make_decision_table.py && python scripts/make_report_figures.py && python scripts/make_pipeline_figure.py
```
The map directories are always given in the same order (EfficientNetV2-S first): the scorer's per-family descriptors
depend on it. `fit` prints the cross-fitted pooled PQ for tau in 0.12-0.40 and stores the centre of the plateau.

## What was tried and left out

- Distance-map watershed and contact-line cuts for touching filaments: never better than connected components.
- More dense models: the second encoder family lifts the ceiling of the candidate trees (oracle selection 0.559 ->
  0.568) but the realised score by about 0.001.
- Candidates that group fragments at a low threshold while keeping the shape of a higher one: no oracle headroom.
- A CNN scoring each candidate from its image window as extra descriptors: +0.0004 +- 0.0015 when regularised; an
  over-trained version gains 0.003, but only on images within three days of a training image (it memorises how the same
  filament was labelled on a neighbouring day), so it is not part of the method.
- A plain U-Net as refiner and averaging two refiners: trained on an earlier selection, UNet++ scored 0.449, U-Net
  0.446 and their average 0.448; a retrained UNet++ scored 0.448, so these gaps are within run-to-run variation.
  The refiner in `weights/` was trained on the selections of the scorer that still had the two file-name descriptors;
  the refiner itself never sees file-name information, and it is evaluated out-of-fold on the final selections.

## Data notes

707 training images, 1154 annotation sets (296 images annotated by 2-3 annotators), 8199 filaments,
median filament area 1228 px; only 0.9 % of filaments touch another filament within 3 px. Images are
2048x2048 8-bit grayscale JPEGs; file names encode time and observatory. Polygons are rasterised with
pycocotools, which reproduces the `area` field of the annotation file exactly. Only the competition data is
used for training; the public MAGFiLO release overlaps the test set and is deliberately not used.
