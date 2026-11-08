#!/bin/bash
# Export every trained fold to weights/ as half-precision checkpoints and record post-processing parameters.
# usage: scripts/export_all.sh "<tags>" <method> <thr> <min_area> <aux_thr>
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
tags=$1; method=$2; thr=$3; min_area=$4; aux=$5
mkdir -p weights
for t in $tags; do for f in 0 1 2 3 4; do
  [ -f runs/${t}_f${f}/best.pt ] && $PY src/export_weights.py runs/${t}_f${f} weights/${t}_f${f}.pt
done; done
printf '{"method": "%s", "thr": %s, "min_area": %s, "aux_thr": %s, "tta": true}\n' "$method" "$thr" "$min_area" "$aux" > weights/params.json
cat weights/params.json
