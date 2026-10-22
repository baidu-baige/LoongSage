#!/usr/bin/env bash
# Launch RLCD² training on the given GPUs (the scorer servers must already be up, see launch_scorer.sh).
#   coda/custom/rlcd2/tools/launch_train.sh <gpu_ids comma-separated> <config-name> [hydra overrides...]
#   e.g. coda/custom/rlcd2/tools/launch_train.sh 0,1,2,3,4,5 rlcd2/rlcd2_4b_h20_1node_local total_steps=30
# Prints the trainer log path (LoongSage/log/trainer_<time>.log); the trainer runs in the background.
# Stop: pkill -f "[c]oda.controller.trainer"; ray stop --force
#
# The GPU list goes into CUDA_VISIBLE_DEVICES so the local Ray started by the trainer only sees these GPUs
# (the scorers keep theirs). It must be 0,1,...,n-1: Ray reports the physical id and SGLang reads base_gpu_id
# as an index into the visible GPUs; the two agree only for such a prefix.
set -euo pipefail
GPUS=${1:?gpu ids, e.g. 0,1,2,3,4,5}
CONFIG=${2:?config name, e.g. rlcd2/rlcd2_4b_h20_1node_local}
shift 2

n=$(awk -F, '{print NF}' <<< "$GPUS")
expected=$(seq -s, 0 $((n - 1)))
if [[ "$GPUS" != "$expected" ]]; then
  echo "gpu ids must be $expected (a prefix 0..n-1), not $GPUS" >&2
  exit 1
fi

ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)   # LoongSage
cd "$ROOT"
export PYTHONPATH=$ROOT${PYTHONPATH:+:$PYTHONPATH}
export CUDA_VISIBLE_DEVICES=$GPUS
bash examples/start.sh "$CONFIG" "$@"
