#!/usr/bin/env bash
# ============================================================================
# Central config for all pipeline scripts.
# Source this file — don't run it directly.
#
# Usage in other scripts:
#   source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
# ============================================================================

# ── Model Configuration ─────────────────────────────────────────────────────
# Change these to switch models. Every script picks them up automatically.

MODEL="${MODEL:-Qwen/Qwen3-32B-AWQ}"
QUANTIZATION="${QUANTIZATION:-awq_marlin}" # "", "gptq", "awq", "awq_marlin", "bnb-4bit", "bnb-8bit"
DTYPE="${DTYPE:-half}"                    # "auto", "half", "bfloat16", "float16"
TP_SIZE="${TP_SIZE:-}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"

# ── Paths ────────────────────────────────────────────────────────────────────
# Base workspace — all outputs live under here
WORKSPACE="${WORKSPACE:-/workspace}"

# Derive a short tag for output directories (e.g. "qwen3-32b-gptq" or "qwen3-32b")
if [ -n "$QUANTIZATION" ]; then
    MODEL_TAG="${MODEL_TAG:-$(echo "$MODEL" | sed 's|.*/||; s|[^a-zA-Z0-9._-]|-|g')-${QUANTIZATION}}"
else
    MODEL_TAG="${MODEL_TAG:-$(echo "$MODEL" | sed 's|.*/||; s|[^a-zA-Z0-9._-]|-|g')}"
fi

RESPONSES_DIR="${RESPONSES_DIR:-${WORKSPACE}/${MODEL_TAG}/responses}"
ACTIVATIONS_DIR="${ACTIVATIONS_DIR:-${WORKSPACE}/${MODEL_TAG}/activations}"
SCORES_DIR="${SCORES_DIR:-${WORKSPACE}/${MODEL_TAG}/scores}"
VECTORS_DIR="${VECTORS_DIR:-${WORKSPACE}/${MODEL_TAG}/vectors}"
AXIS_OUTPUT="${AXIS_OUTPUT:-${WORKSPACE}/${MODEL_TAG}/axis.pt}"

# ── Pipeline Defaults ────────────────────────────────────────────────────────
QUESTION_COUNT="${QUESTION_COUNT:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
LAYERS="${LAYERS:-all}"
ROLES="${ROLES:-default pirate therapist demon altruist}"
MIN_COUNT="${MIN_COUNT:-5}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1-mini}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-50}"

# ── Project Paths (auto-detected) ───────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Common Setup Function ───────────────────────────────────────────────────
setup_environment() {
    # Ensure uv is on PATH (needed after fresh install)
    [ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"

    # Install uv if not present
    if ! command -v uv &> /dev/null; then
        echo "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source "$HOME/.local/bin/env"
    fi

    # HuggingFace token (RunPod or env)
    if [ -n "${RUNPOD_SECRET_hf_token:-}" ]; then
        export HUGGING_FACE_HUB_TOKEN="$RUNPOD_SECRET_hf_token"
        echo "HF token loaded from RUNPOD_SECRET_hf_token"
    elif [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
        echo "WARNING: No HuggingFace token found. Gated models will fail."
    fi

    export HF_HOME="${HF_HOME:-${WORKSPACE}/huggingface_cache}"

    # Install project dependencies
    cd "$PROJECT_DIR"
    uv sync
}

# ── GPU Check ────────────────────────────────────────────────────────────────
check_gpu() {
    if ! uv run python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "ERROR: No CUDA GPU detected."
        exit 1
    fi
    GPU_COUNT=$(uv run python3 -c "import torch; print(torch.cuda.device_count())")
    echo "GPUs detected: $GPU_COUNT"
}

# ── Build common CLI args ────────────────────────────────────────────────────
build_tp_arg() {
    if [ -n "$TP_SIZE" ]; then
        echo "--tensor_parallel_size $TP_SIZE"
    fi
}

build_quant_args() {
    # Quantization args for vLLM scripts (generation) — includes --dtype
    local args=""
    if [ -n "$QUANTIZATION" ]; then
        args="--quantization $QUANTIZATION"
    fi
    if [ "$DTYPE" != "auto" ]; then
        args="$args --dtype $DTYPE"
    fi
    echo "$args"
}

build_quant_arg() {
    # Quantization arg for HuggingFace scripts (activations) — no --dtype
    # awq_marlin is vLLM-only; HuggingFace sees it as plain awq
    if [ -n "$QUANTIZATION" ]; then
        local q="$QUANTIZATION"
        [ "$q" = "awq_marlin" ] && q="awq"
        echo "--quantization $q"
    fi
}

# ── Print current config ────────────────────────────────────────────────────
print_config() {
    echo "═══════════════════════════════════════════════════════"
    echo "  Model:         $MODEL"
    echo "  Quantization:  ${QUANTIZATION:-none}"
    echo "  Dtype:         $DTYPE"
    echo "  Model tag:     $MODEL_TAG"
    echo "  TP size:       $TP_SIZE"
    echo "  Workspace:     $WORKSPACE"
    echo "═══════════════════════════════════════════════════════"
}
