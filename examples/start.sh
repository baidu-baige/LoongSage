#!/bin/bash
# Launch the trainer with a config from conf
# Usage: bash examples/start.sh <config-name> [hydra-overrides...]
# e.g:
#   bash examples/start.sh dsv4_flash_bf16/swe_h20_8node
#   bash examples/start.sh qwen3_coder_30b_a3b/opencode_h20_4node
#   bash examples/start.sh qwen3_30b_a3b/dapo_h20_1node hf_model_path=/root/Qwen3-30B-A3B

if [ $# -eq 0 ]; then
	echo "Usage: bash examples/start.sh <config-name> [hydra-overrides...]"
	echo "e.g.:  bash examples/start.sh qwen3_30b_a3b/dapo_h20_1node hf_model_path=/root/Qwen3-30B-A3B"
	exit 1
fi
CONFIG_NAME="$1"
shift   # remaining args are passed to Hydra as overrides

export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0

mkdir -p log
log_file=log/trainer_$(date +%Y%m%d_%H%M%S).log
echo $log_file

nohup python -m coda.controller.trainer --config-name $CONFIG_NAME "$@" > $log_file 2>&1 &
