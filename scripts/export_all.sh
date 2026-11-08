#!/bin/bash
# Export every trained fold to weights/ as half-precision checkpoints and record post-processing parameters.
# usage: scripts/export_all.sh "<tags>" <method> <thr> <min_area> <aux_thr>
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
tags=$1; method=${2:-cc}; thr=${3:-0.5}; min_area=${4:-400}; aux=${5:-0.5}
mkdir -p weights
for t in $tags; do for f in 0 1 2 3 4; do
  if [ -f "runs/${t}_f${f}/best.pt" ]; then
    $PY src/export_weights.py "runs/${t}_f${f}" "weights/${t}_f${f}.pt"   # export_weights.py takes the run directory
  fi
done; done
printf '{"method": "%s", "thr": %s, "min_area": %s, "aux_thr": %s, "tta": true}\n' "$method" "$thr" "$min_area" "$aux" > weights/params.json
cat weights/params.json
