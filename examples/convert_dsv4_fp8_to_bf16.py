"""Convert DeepSeek-V4 FP8/MXFP4 HuggingFace checkpoint to BF16.
Usage:
    python tools/convert_dsv4_fp8_to_bf16.py \
        --input-fp8-hf-path /path/to/DeepSeek-V4-Flash \
        --output-bf16-hf-path /path/to/DeepSeek-V4-Flash-BF16
"""

import json
import os
import shutil
from argparse import ArgumentParser
from glob import glob

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

# fp4_e2m1 lookup table: 4 bits -> float value
# Layout: bit3=sign, bit2-1=exponent, bit0=mantissa
_FP4_E2M1_TABLE = torch.zeros(16, dtype=torch.float32)
for _i in range(16):
    _sign = -1.0 if (_i >> 3) & 1 else 1.0
    _exp = (_i >> 1) & 0x3
    _man = _i & 0x1
    if _exp == 0:
        _val = 0.5 * _man  # subnormal
    else:
        _val = (1.0 + 0.5 * _man) * (2.0 ** (_exp - 1))
    _FP4_E2M1_TABLE[_i] = _sign * _val


def _is_mxfp4_packed(x: torch.Tensor, s: torch.Tensor) -> bool:
    """Detect MXFP4 packing.

    The column ratio alone is ambiguous between MXFP4 (block [1, 32] over unpacked
    columns) and MXFP8 (block [128, 128]), so we disambiguate on the row block size:
    MXFP4 experts scale per row (block_m == 1), MXFP8 attention weights use 128.
    """
    if x.dtype != torch.int8:
        return False
    M, _ = x.shape
    return M // s.shape[0] == 1


def weight_dequant_fp8(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Dequantize MXFP8 (e4m3fn) weight to BF16."""
    M, N = x.shape
    if x.dtype == torch.int8:
        x = x.view(torch.float8_e4m3fn)

    block_size_m = M // s.shape[0]
    block_size_n = N // s.shape[1]

    scale_float = s.to(torch.float32)
    scale_expanded = scale_float.repeat_interleave(block_size_m, dim=0).repeat_interleave(block_size_n, dim=1)
    scale_expanded = scale_expanded[:M, :N]

    return (x.to(torch.float32) * scale_expanded).to(torch.bfloat16)


def weight_dequant_fp4(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Dequantize MXFP4 (e2m1, 2 values packed per int8 byte) weight to BF16.

    Each int8 byte stores 2 fp4 values: low nibble first, high nibble second.
    Block size is [1, 32] (32 fp4 elements share one e8m0 scale).
    """
    M, N_packed = x.shape
    N_actual = N_packed * 2

    raw = x.to(torch.uint8)
    low = (raw & 0x0F).long()
    high = ((raw >> 4) & 0x0F).long()

    unpacked = torch.stack([low, high], dim=-1).reshape(M, N_actual)

    values = _FP4_E2M1_TABLE[unpacked]

    scale_float = s.to(torch.float32)
    block_n = N_actual // s.shape[1]
    scale_expanded = scale_float.repeat_interleave(block_n, dim=1)[:, :N_actual]

    return (values * scale_expanded).to(torch.bfloat16)


def main(fp8_path, bf16_path):
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(bf16_path, exist_ok=True)

    # Copy config.json with quantization_config removed.
    # Also force num_nextn_predict_layers=0: after conversion the MTP layer is
    # kept only for weight compatibility, not executed. Leaving it at 1 makes
    # Megatron actually run MTP forward, where router_replay hits a NoneType
    # top_indices (no replay is stored for MTP MoE during forward-only calls).
    config_src = os.path.join(fp8_path, "config.json")
    config_dst = os.path.join(bf16_path, "config.json")
    with open(config_src) as f:
        config = json.load(f)
    config.pop("quantization_config", None)
    config["torch_dtype"] = "bfloat16"
    config["num_nextn_predict_layers"] = 0
    with open(config_dst, "w") as f:
        json.dump(config, f, indent=2)

    # Copy tokenizer, chat_template, and other auxiliary files
    for pattern in ("tokenizer*", "chat_template*", "*.py", "generation_config.json"):
        for src_file in glob(os.path.join(fp8_path, pattern)):
            basename = os.path.basename(src_file)
            if basename == "config.json":
                continue
            shutil.copy2(src_file, os.path.join(bf16_path, basename))

    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    with open(model_index_file) as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]

    # Native DSV4 names scales <param>.scale; sgl-project FP8 uses <param>_scale_inv.
    has_dot_scale = any(k.endswith(".scale") for k in weight_map)
    has_scale_inv = any(k.endswith("_scale_inv") for k in weight_map)
    if has_dot_scale:
        print("Detected native DSV4 scale format: <param>.scale")
    elif has_scale_inv:
        print("Detected sgl-project scale format: <param>_scale_inv")
    else:
        print("Warning: No scale tensors detected, will copy weights as-is")

    # MTP e_proj/h_proj must remain in FP8 format (weight + scale preserved)
    # megatron-bridge expects these as quantized weights
    fp8_keep_prefixes = ("mtp.0.e_proj", "mtp.0.h_proj")

    # Tied MTP aliases of the top-level embed/head. Native DSV4 emits them as
    # duplicates; the reference conversion omits them, and Megatron/bridge
    # does not expect them when num_nextn_predict_layers=0.
    tied_mtp_skip = ("mtp.0.emb.tok_emb.weight", "mtp.0.head.weight")

    # Cache for loaded safetensor files (for cross-file scale lookup)
    loaded_files = {}
    converted_count = 0

    def get_tensor(tensor_name):
        """Get tensor from the correct file (handles cross-file scale references)."""
        file_name = weight_map[tensor_name]
        if file_name not in loaded_files:
            file_path = os.path.join(fp8_path, file_name)
            loaded_files[file_name] = load_file(file_path, device="cpu")
        return loaded_files[file_name][tensor_name]

    def find_scale(weight_name):
        """Find the scale tensor for a given weight, trying both naming conventions."""
        if has_dot_scale:
            # layers.0.attn.wkv.weight -> layers.0.attn.wkv.scale
            if weight_name.endswith(".weight"):
                scale_name = weight_name.rsplit(".weight", 1)[0] + ".scale"
            else:
                scale_name = weight_name + ".scale"
            if scale_name in weight_map:
                return get_tensor(scale_name)
        if has_scale_inv:
            # layers.0.attn.wkv.weight -> layers.0.attn.wkv.weight_scale_inv
            scale_name = f"{weight_name}_scale_inv"
            if scale_name in weight_map:
                return get_tensor(scale_name)
        return None

    safetensor_files = sorted(glob(os.path.join(fp8_path, "*.safetensors")))
    for safetensor_file in tqdm(safetensor_files, desc="Converting FP8 -> BF16"):
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cpu")
        loaded_files[file_name] = current_state_dict

        new_state_dict = {}
        for weight_name, weight in current_state_dict.items():
            if weight_name in tied_mtp_skip:
                continue

            if any(weight_name.startswith(p) for p in fp8_keep_prefixes):
                # Scale is cast to float32; the weight stays fp8_e4m3fn.
                if weight_name.endswith(".scale"):
                    new_state_dict[weight_name] = weight.to(torch.float32)
                else:
                    new_state_dict[weight_name] = weight
                continue

            if weight_name.endswith(".scale") or weight_name.endswith("_scale_inv"):
                continue

            # 1-byte dtypes: fp8_e4m3fn or int8 (MXFP4 packed)
            if weight.element_size() == 1 and weight.dim() == 2:
                scale = find_scale(weight_name)
                if scale is not None:
                    if _is_mxfp4_packed(weight, scale):
                        new_state_dict[weight_name] = weight_dequant_fp4(weight, scale)
                    else:
                        new_state_dict[weight_name] = weight_dequant_fp8(weight, scale)
                    converted_count += 1
                else:
                    print(f"Warning: No scale found for {weight_name}, keeping as-is")
                    new_state_dict[weight_name] = weight
            else:
                new_state_dict[weight_name] = weight

        new_safetensor_file = os.path.join(bf16_path, file_name)
        save_file(new_state_dict, new_safetensor_file)

        # Memory management: keep only the 2 most recently loaded files
        if len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]

    # Update model index: drop scale entries except the FP8-preserved MTP ones,
    # and drop tied MTP aliases. Native keys are kept unchanged.
    new_weight_map = {}
    for weight_name, file_name in weight_map.items():
        if weight_name in tied_mtp_skip:
            continue
        if weight_name.endswith(".scale") or weight_name.endswith("_scale_inv"):
            if any(weight_name.startswith(p) for p in fp8_keep_prefixes):
                new_weight_map[weight_name] = file_name
            continue
        new_weight_map[weight_name] = file_name

    new_model_index_file = os.path.join(bf16_path, "model.safetensors.index.json")
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": new_weight_map}, f, indent=2)

    print(f"Done. Converted {converted_count} quantized tensors to BF16.")
    print(f"Output: {bf16_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Convert DeepSeek-V4 FP8 checkpoint to BF16 (CPU-only)")
    parser.add_argument("--input-fp8-hf-path", type=str, required=True, help="Path to FP8 HF checkpoint")
    parser.add_argument("--output-bf16-hf-path", type=str, required=True, help="Path for BF16 HF output")
    args = parser.parse_args()
    main(args.input_fp8_hf_path, args.output_bf16_hf_path)

