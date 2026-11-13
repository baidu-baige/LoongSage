"""LoRA helpers shared by the trainer, the Megatron worker, the SGLang engine and the router."""

# Same mapping as miles' convert_target_modules_to_hf.
_MCORE_TO_HF_LORA_MODULES = {
    "linear_qkv": ["q_proj", "k_proj", "v_proj"],
    "linear_proj": ["o_proj"],
    "linear_fc1": ["gate_proj", "up_proj"],
    "linear_fc2": ["down_proj"],
}


def lora_delta_sync_enabled(config) -> bool:
    """True for LoRA merge=false: SGLang keeps the base and serves the adapter."""
    lora = (config.get("megatron") or {}).get("lora") or {}
    return bool(lora.get("enable", False) and not lora.get("merge", True))


def hf_lora_target_modules(target_modules) -> list[str]:
    """Map mcore LoRA targets to the HF module names SGLang wraps.

    Wildcards map by their last segment. Unknown names raise.
    """
    hf_modules: list[str] = []
    for module in target_modules:
        leaf = module.rsplit(".", 1)[-1]
        if leaf not in _MCORE_TO_HF_LORA_MODULES:
            raise ValueError(
                f"megatron.lora.target_modules entry {module!r} has no SGLang LoRA "
                f"mapping (supported: {sorted(_MCORE_TO_HF_LORA_MODULES)}); "
                "use megatron.lora.merge=true for other targets."
            )
        for hf_name in _MCORE_TO_HF_LORA_MODULES[leaf]:
            if hf_name not in hf_modules:
                hf_modules.append(hf_name)
    return hf_modules
