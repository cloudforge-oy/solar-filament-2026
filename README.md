# Solar Filament Segmentation Challenge 2026

Instance segmentation of solar filaments in GONG H-alpha images (MAGFiLO dataset) for the Kaggle
competition https://www.kaggle.com/competitions/filament-segmentation-2026.

The method and results are described in `report/main.tex` (organizers' template). The notebook
`notebooks/pipeline.ipynb` reproduces the submitted predictions from the stored weights in `weights/`.

## Method in one paragraph

A U-Net (ImageNet-pretrained ResNet34 or EfficientNet-B3 encoder, one grayscale input channel) predicts
three maps at native 2048x2048 resolution: the filament probability, a normalised distance-to-boundary map
and the contact line between touching filaments. Supervision uses *consensus* targets: the fraction of the
1-3 annotators of an image who marked each pixel. Training uses random native-resolution crops with
flips, rotations and photometric jitter; validation folds are grouped by observation date. At inference the
models (and flip test-time augmentation) are averaged, the probability is thresholded, the mask is cut
along predicted contact lines and labelled into 8-connected components, small components are dropped, and
each component becomes one COCO RLE row of the submission. Threshold, minimum area and contact
threshold are chosen by grid search on validation Panoptic Quality computed against every annotator set.

## Layout

| path | purpose |
|---|---|
| `src/prepare.py` | rasterise COCO polygons to instance maps, build consensus targets, make date-grouped folds |
| `src/common.py` | model definition, instance extraction (components / contact-line split / watershed), PQ metric, RLE |
| `src/train.py` | training loop (BCE + Dice + distance MSE + weighted contact BCE), full-frame validation |
| `src/tune.py` | grid search of post-processing parameters on validation PQ |
| `src/evaluate.py` | single-model and ensemble evaluation with standard errors and bootstrap intervals |
| `src/predict.py` | test-set inference, ensembling, instance extraction, submission CSV |
| `src/check_submission.py` | validates a CSV: RLE decodability, unique ids, no overlapping masks |
| `src/export_weights.py` | writes compact half-precision checkpoints into `weights/` |
| `scripts/train_folds.sh` | trains one encoder on several folds sequentially |
| `scripts/make_*_figures.py` | figures for the report |
| `notebooks/pipeline.ipynb` | end-to-end illustration: data, model, validation, submission |
| `report/` | 4-page technical report (LaTeX, organizers' acmart template) |

## Reproduce the submission

```bash
pip install -r requirements.txt
# download and unzip the Kaggle data into data/raw/MAGFiLO_1.0_Kaggle_2026 (train/ and test/)
python src/prepare.py                       # only needed for training / validation
python src/predict.py --weights weights/*.pt --tta --method split --thr 0.6 --min_area 400 --aux_thr 0.5
python src/check_submission.py submission.csv
```
(The exact post-processing parameters of the final submission are recorded in `weights/README.md`.)

## Train from scratch

```bash
python src/prepare.py
scripts/train_folds.sh resnet34        resnet34d "0 1 2 3 4" --epochs 40 --eval_every 5
scripts/train_folds.sh efficientnet-b3 effb3d    "0 1 2 3 4" --epochs 50 --crop 640 --bs 6 --eval_every 10
python src/evaluate.py --weights runs/resnet34d_f0/best.pt runs/effb3d_f0/best.pt --tta
python src/export_weights.py runs/resnet34d_f0 weights/resnet34d_f0.pt
```
One ResNet34 fold takes about 45 minutes on an 8 GB laptop GPU; inference on the 180 test images with
TTA takes about 5 minutes per model.

## Data notes

707 training images, 1154 annotation sets (296 images annotated by 2-3 annotators), 8199 filaments,
median filament area 1228 px; only 0.9 % of filaments touch another filament within 3 px. Images are
2048x2048 8-bit grayscale JPEGs; file names encode time and observatory. Only the competition data is
used for training; the public MAGFiLO release overlaps the test set and is deliberately not used.
