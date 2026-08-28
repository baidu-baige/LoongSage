# Quick Start

This page walks the fastest path — environment → model and data → launch → verification —
running `conf/qwen3_4b/dapo_h800_1node` (Qwen3-4B + DAPO) on a single 8-GPU (H800) node to get
the first training step going. Other launch options (multi-node, …) and run modes live in the
[Run Guide](run-guide.md); every field is documented in the [Config Reference](config-reference.md).

## 1. Prepare the environment

Since LoongSage may include version-specific builds and temporary patches for SGLang/Megatron, we
strongly recommend using the latest Docker image we provide to avoid environment issues.

### Pull and start the Docker container

```bash
# pull the latest image
docker pull loongsage/loongsage:latest

docker run --rm --gpus all --ipc=host \
  -it loongsage/loongsage:latest /bin/bash
```

## 2. Prepare model and data

### Download the model and dataset

This preset uses Qwen3-4B. Download the weights straight from Hugging Face:

```bash
hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B
```

This preset is a DAPO math task and uses the DAPO-Math-17k training set:

```bash
hf download Haitao999/DAPO-Math-17k-unique --repo-type=dataset \
  --local-dir /root/DAPO-Math-17k-unique
```

## 3. Launch training

### Launch

There are two equivalent ways to launch:

**Method 1 — fill in the config file, then launch with `examples/start.sh` (recommended for the
first run).** Fill the two `???` directly in `conf/qwen3_4b/dapo_h800_1node.yaml`.

Fill in the required fields

```yaml
hf_model_path: /root/Qwen3-4B
data_source:
  dataset:
    prompt_data_path: /root/DAPO-Math-17k-unique/<file>.parquet
```

Run

```bash
bash examples/start.sh qwen3_4b/dapo_h800_1node
```

**Method 2 — leave the config untouched and override on the command line.** `examples/start.sh`
forwards everything after the config name straight to Hydra:

```bash
bash examples/start.sh qwen3_4b/dapo_h800_1node \
  hf_model_path=/root/Qwen3-4B \
  data_source.dataset.prompt_data_path=/root/DAPO-Math-17k-unique
```

## 4. Verify the results

`tracking.tracking_backend` defaults to `console`, so metrics go straight to the log. Watch these
keys:

| Metric | Meaning |
| --- | --- |
| `rollout/completed_count` | Trajectories collected this step; equals `num_prompts_per_step × num_trajectories_per_prompt` |
| `rollout/reward_mean` | Mean reward |
| `train/is_approx_k3_kl` | Approximate KL between the current policy and the sampling policy |
| `train/pg_loss`, `train/entropy` | Training loss, entropy |
| `timing/*` | Per-phase wall time |

## 5. Next steps

- Launch options (multi-node included), run modes, multiple data sources, evaluation → [Run Guide](run-guide.md)
- Field-by-field configuration → [Config Reference](config-reference.md)
- Plugging in your own agent / reward / sandbox → [Custom Agent](custom-agent.md),
  [Custom Reward](custom-reward.md), [Custom Sandbox](custom-sandbox.md)