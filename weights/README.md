# Trained weights

Everything `scripts/predict_test.sh` needs. All network files are half-precision state dicts.

| file | what |
|---|---|
| `effv2s_f<k>.pt` | U-Net, EfficientNetV2-S encoder (timm `tf_efficientnetv2_s`), trained on all folds except `k`; `{"encoder", "norm": "disk", "state_dict"}` |
| `effb3d_f<k>.pt` | U-Net, EfficientNet-B3 encoder, trained on all folds except `k`; `{"encoder", "state_dict"}` (inputs standardised with dataset constants) |
| `decision_scorer.pkl` | stage-1 scorer: four pairs of scikit-learn 1.9.1 HistGradientBoosting models, the 67 descriptor names (image and map statistics only), tau, the cross-validated instances per image and the out-of-fold tau curve |
| `refiner_f<k>.pt` | stage-2 refiner, UNet++ with a ResNet34 encoder and four input channels, trained on the out-of-fold selections of all folds except `k`; `{"encoder", "arch", "state_dict"}` |
| `params.json` | settings of the submitted CSV |

The dense models are loaded by `src/predict.py:load_models`, the refiners by `src/refine.py:load_refiners`. At
inference all five folds of every network are averaged.

Training commands are in the top-level README ("Train from scratch"). The ResNet34 U-Nets and the first-generation
scorer described in earlier versions of the report are not needed and not stored here.
