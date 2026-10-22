#!/usr/bin/env bash
# Launch frozen RLCD² scorer servers (SGLang, one per GPU) for the letter read-out.
#   coda/custom/rlcd2/tools/launch_scorer.sh <gpu_ids comma-separated> <base_port> <model_path>
#   e.g. coda/custom/rlcd2/tools/launch_scorer.sh 6,7 30106 /path/<model dir>
#        -> http://127.0.0.1:30106,http://127.0.0.1:30107
# Prints the comma-separated url list (data_source.reward.scorer_urls / build_rlcd2_data.py --scorer-url).
#
# --enable-fp32-lm-head: the read-out is a softmax over a few letter logits; bf16 logits are quantised
#   (about 0.06-0.125 at logit magnitude 10-20) and produce exact ties. FP32 logits halve the gap to a
#   transformers reference read-out.
# --disable-radix-cache: prefix caching on the hybrid GDN model was not validated for read-out parity.
# The servers listen on 127.0.0.1 only and have no authentication.
set -euo pipefail
GPUS=${1:?gpu ids, e.g. 6,7}
BASE_PORT=${2:?base port, e.g. 30106}
MODEL=${3:?model folder}
MEM=${SCORER_MEM_FRACTION:-0.8}
CTX=${SCORER_CONTEXT_LENGTH:-32768}
HERE=$(cd "$(dirname "$0")" && pwd)
LOG_DIR=${SCORER_LOG_DIR:-$HERE/../../../../log}   # LoongSage/log
mkdir -p "$LOG_DIR"

urls=()
i=0
for gpu in ${GPUS//,/ }; do
  port=$((BASE_PORT + i))
  CUDA_VISIBLE_DEVICES=$gpu nohup python3 -m sglang.launch_server --model-path "$MODEL" \
    --host 127.0.0.1 --port "$port" --tp 1 --mem-fraction-static "$MEM" --context-length "$CTX" \
    --disable-radix-cache --enable-fp32-lm-head --log-level warning \
    > "$LOG_DIR/scorer_gpu${gpu}_port${port}.log" 2>&1 &
  urls+=("http://127.0.0.1:$port")
  i=$((i + 1))
done

for u in "${urls[@]}"; do
  for _ in $(seq 1 120); do
    if curl -sf -m 5 "$u/health" > /dev/null; then break; fi
    sleep 5
  done
  curl -sf -m 5 "$u/health" > /dev/null || { echo "scorer $u did not come up" >&2; exit 1; }
done
(IFS=,; echo "${urls[*]}")
