#!/bin/bash
# Full inference from the files in weights/: test images -> submission CSV.
# GPU required; about 50 minutes on a laptop RTX 5060 (8 GB), most of it the 8-view dense prediction of 10 models.
# usage: scripts/predict_test.sh [out.csv]
set -eo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}; W=weights; M=runs/maps; OUT=${1:-submission.csv}
mkdir -p runs

# 1. dense maps: mean of the five fold models of each encoder family, 8 dihedral views (existing files are skipped)
$PY src/predict_maps.py test --weights $W/effv2s_f0.pt $W/effv2s_f1.pt $W/effv2s_f2.pt $W/effv2s_f3.pt $W/effv2s_f4.pt --out $M/test_effv2s --tta 8
$PY src/predict_maps.py test --weights $W/effb3d_f0.pt $W/effb3d_f1.pt $W/effb3d_f2.pt $W/effb3d_f3.pt $W/effb3d_f4.pt --out $M/test_effb3d --tta 8
MAPS="$M/test_effv2s $M/test_effb3d"      # family order as in the candidate bank the scorer was fitted on

# 2. decision layer: component tree, scorer, exact selection; tau restores the cross-validated instances per image
$PY src/decision.py predict --maps $MAPS --model $W/decision_scorer.pkl --match_rate --out runs/stage1.csv

# 3. stage 2: every selected instance is re-segmented by the refiner ensemble
$PY src/refine.py predict --stage1 runs/stage1.csv --maps $MAPS --t 0.45 --out "$OUT" \
    --weights $W/refiner_f0.pt $W/refiner_f1.pt $W/refiner_f2.pt $W/refiner_f3.pt $W/refiner_f4.pt

$PY src/check_submission.py "$OUT"
