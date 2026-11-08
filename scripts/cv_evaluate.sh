#!/bin/bash
# Out-of-fold evaluation: for every fold, evaluate the models trained without that fold on it.
# usage: scripts/cv_evaluate.sh "<tags>" e.g. scripts/cv_evaluate.sh "resnet34d effb3d"
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
tags=$1
for f in 0 1 2 3 4; do
  w=""; for t in $tags; do w="$w runs/${t}_f${f}/best.pt"; done
  $PY src/evaluate.py --weights $w --fold $f --tta --out runs/evaluation_f${f}.json 2>&1 | grep -v Warning
done
