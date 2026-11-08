# Solar Filament Segmentation Challenge 2026

Instance segmentation of solar filaments in GONG H-alpha images (MAGFiLO dataset), for the
Kaggle competition https://www.kaggle.com/competitions/filament-segmentation-2026.

## Method

1. **Targets.** Each training image may be annotated by 1 to 3 annotators. Polygons are
   rasterized to 2048x2048 instance maps (`src/prepare.py`). The pixel-wise training target is
   the *consensus* mask: the fraction of annotators that marked the pixel (a soft label).
2. **Model.** A U-Net with an ImageNet-pretrained ResNet34 encoder (segmentation-models-pytorch),
   single grayscale input channel, single output channel (filament probability).
3. **Training.** Random 512x512 crops at native resolution (70% centred near a filament),
   flips / 90-degree rotations, mild photometric jitter. Loss = BCE + soft Dice. AdamW with a
   one-cycle schedule, mixed precision. Validation fold is grouped by observation date.
4. **Inference.** Full 2048x2048 image in one forward pass, optional flip test-time augmentation.
5. **Instances.** Threshold the probability map and take 8-connected components, discarding
   components below a minimum area. Components are disjoint by construction, which the
   Panoptic Quality scorer requires. Threshold and minimum area are tuned on validation PQ
   (`src/tune.py`).
6. **Submission.** Each instance is encoded as a COCO RLE counts string (`src/predict.py`), and
   `src/check_submission.py` verifies decodability, unique ids and absence of overlaps.

## Reproduce

```bash
pip install -r requirements.txt
# unzip the Kaggle data into data/raw/MAGFiLO_1.0_Kaggle_2026
python src/prepare.py
python src/train.py --encoder resnet34 --fold 0 --epochs 40
python src/tune.py --weights runs/resnet34_f0/best.pt --tta
python src/predict.py --weights runs/resnet34_f0/best.pt --tta --thr 0.5 --min_area 150
python src/check_submission.py submission.csv
```
