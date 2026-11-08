# Trained weights

Half-precision state dicts of the five EfficientNet-B3 U-Nets used in the final submission, one per
cross-validation fold (`effb3d_f<k>.pt` was trained on all folds except `k`). Each file is a dict
`{"encoder": "efficientnet-b3", "state_dict": ...}` and is loaded by `src/predict.py:load_models`.
All five are averaged (together with flip test-time augmentation) at inference.

`params.json` records the post-processing used for the submitted CSV: connected components of the
averaged filament probability above `thr`, components smaller than `min_area` pixels dropped.

Training command (per fold, see `scripts/train_folds.sh`):
```
python src/train.py --encoder efficientnet-b3 --fold <k> --epochs 50 --crop 640 --bs 6 --eval_every 10
```
The ResNet34 ablation models described in the report are not needed to reproduce the submission and
are not stored here (each is 49 MB; available on request).
