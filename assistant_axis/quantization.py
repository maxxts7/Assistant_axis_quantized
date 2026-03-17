"""
Central quantization utilities for assistant-axis.

This module is the single source of truth for quantization support.
All other modules import from here — never construct quantization configs directly.

Supported methods:
    "gptq"     — Pre-quantized GPTQ models (auto-detected from Hub)
    "awq"      — Pre-quantized AWQ models (auto-detected from Hub)
    "bnb-4bit" — On-the-fly 4-bit quantization via BitsAndBytes (NF4)
    "bnb-8bit" — On-the-fly 8-bit quantization via BitsAndBytes

Example:
    from assistant_axis.quantization import resolve_for_hf, resolve_for_vllm, get_compute_dtype

    # Loading a model with HuggingFace
    hf_kwargs = resolve_for_hf("awq")
    model = AutoModelForCausalLM.from_pretrained(name, **hf_kwargs)

    # Loading a model with vLLM
    vllm_kwargs = resolve_for_vllm("awq")
    llm = LLM(model=name, **vllm_kwargs)

    # Getting the compute dtype from a loaded model
    dtype = get_compute_dtype(model)  # always returns a float dtype
"""

import torch
import torch.nn as nn
from typing import Optional


# Maps quantization string to vLLM's quantization kwarg
_VLLM_METHODS = {
    "gptq": "gptq",
    "awq": "awq",
    "awq_marlin": "awq_marlin",
    "bnb-4bit": "bitsandbytes",
    "bnb-8bit": "bitsandbytes",
}

# Valid quantization method strings
VALID_METHODS = frozenset(_VLLM_METHODS.keys())


def get_compute_dtype(model: nn.Module) -> torch.dtype:
    """
    Extract the floating-point compute dtype from any model (quantized or not).

    For quantized models, this returns the dtype used for activations during
    forward passes — NOT the integer dtype of stored weights.

    Args:
        model: A loaded HuggingFace model (may be quantized)

    Returns:
        A floating-point torch.dtype (never int4/int8/uint8)
    """
    # Check for quantization config (populated by HuggingFace for pre-quantized models)
    if hasattr(model, "config") and hasattr(model.config, "quantization_config"):
        qconfig = model.config.quantization_config
        # BitsAndBytes stores compute dtype explicitly
        if hasattr(qconfig, "bnb_4bit_compute_dtype"):
            return qconfig.bnb_4bit_compute_dtype
        # Some configs expose compute_dtype directly
        if hasattr(qconfig, "compute_dtype"):
            dt = qconfig.compute_dtype
            if isinstance(dt, torch.dtype):
                return dt
        # GPTQ/AWQ default to float16 for compute
        return torch.float16

    # Not quantized — find first floating-point parameter
    for p in model.parameters():
        if p.dtype.is_floating_point:
            return p.dtype

    return torch.bfloat16


def resolve_for_hf(quantization: Optional[str]) -> dict:
    """
    Convert a quantization string into kwargs for AutoModelForCausalLM.from_pretrained().

    For pre-quantized models (GPTQ/AWQ), returns empty dict — HuggingFace
    auto-detects from the model's config.json on the Hub.

    For on-the-fly quantization (BitsAndBytes), returns a quantization_config.

    Args:
        quantization: One of "gptq", "awq", "bnb-4bit", "bnb-8bit", or None

    Returns:
        Dict of kwargs to pass to from_pretrained()
    """
    if not quantization:
        return {}

    if quantization in ("gptq", "awq", "awq_marlin"):
        # Auto-detected from model config on Hub — no explicit config needed
        return {}

    if quantization == "bnb-4bit":
        from transformers import BitsAndBytesConfig
        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        }

    if quantization == "bnb-8bit":
        from transformers import BitsAndBytesConfig
        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_8bit=True,
            )
        }

    raise ValueError(
        f"Unknown quantization method: {quantization!r}. "
        f"Valid options: {sorted(VALID_METHODS)}"
    )


def resolve_for_vllm(quantization: Optional[str]) -> dict:
    """
    Convert a quantization string into kwargs for vllm.LLM().

    Args:
        quantization: One of "gptq", "awq", "bnb-4bit", "bnb-8bit", or None

    Returns:
        Dict of kwargs to pass to LLM()
    """
    if not quantization:
        return {}

    if quantization not in _VLLM_METHODS:
        raise ValueError(
            f"Unknown quantization method: {quantization!r}. "
            f"Valid options: {sorted(VALID_METHODS)}"
        )

    return {"quantization": _VLLM_METHODS[quantization]}
