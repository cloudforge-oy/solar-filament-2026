# Trained weights

Half-precision state dicts of the five EfficientNet-B3 U-Nets used in the final submission, one per
cross-validation fold (`effb3d_f<k>.pt` was trained on all folds except `k`). Each file is a dict
`{"encoder": "efficientnet-b3", "state_dict": ...}` and is loaded by `src/predict.py:load_models`.
All five are averaged (together with flip test-time augmentation) at inference.

`decision_scorer.pkl` holds the decision-layer scorer (two scikit-learn 1.9.1 HistGradientBoosting models, the
feature list and the out-of-fold tau curve), fitted on all 119,802 out-of-fold candidates. `params.json` records
the settings of the submitted CSV (`src/decision.py predict ... --tau 0.26`).

Training command (per fold, see `scripts/train_folds.sh`):
```
python src/train.py --encoder efficientnet-b3 --fold <k> --epochs 50 --crop 640 --bs 6 --eval_every 10
```
The ResNet34 ablation models described in the report are not needed to reproduce the submission and
are not stored here (each is 49 MB; available on request).
