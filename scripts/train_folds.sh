#!/bin/bash
# Train one encoder on several folds sequentially.
# usage: scripts/train_folds.sh <encoder> <tag> "<folds>" [extra train.py args...]
# e.g.   scripts/train_folds.sh resnet34 resnet34d "1 2 3 4" --epochs 40 --eval_every 5
#        scripts/train_folds.sh efficientnet-b3 effb3d "1 2 3 4" --epochs 50 --crop 640 --bs 6 --eval_every 10
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
enc=$1; tag=$2; folds=$3; shift 3
for f in $folds; do
  $PY src/train.py --encoder "$enc" --fold "$f" --out "runs/${tag}_f${f}" "$@" > "runs/${tag}_f${f}.log" 2>&1
done
echo done > "runs/${tag}.done"
