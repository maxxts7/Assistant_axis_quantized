#!/usr/bin/env bash
# Step 2: Extract activations (small test run)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

setup_environment
check_gpu
print_config

# Verify responses exist
if [ ! -d "$RESPONSES_DIR" ]; then
    echo "ERROR: Responses directory not found: $RESPONSES_DIR"
    echo "Run test_generate.sh first."
    exit 1
fi

RESPONSE_COUNT=$(find "$RESPONSES_DIR" -name '*.jsonl' | wc -l)
if [ "$RESPONSE_COUNT" -eq 0 ]; then
    echo "ERROR: No .jsonl files found in $RESPONSES_DIR"
    exit 1
fi
echo "Found $RESPONSE_COUNT response file(s)"

echo "Batch size: $BATCH_SIZE"
echo "Layers:     $LAYERS"
echo "Output:     $ACTIVATIONS_DIR"
echo ""

uv run "$PROJECT_DIR/pipeline/2_activations.py" \
    --model "$MODEL" \
    --responses_dir "$RESPONSES_DIR" \
    --output_dir "$ACTIVATIONS_DIR" \
    --batch_size "$BATCH_SIZE" \
    --layers "$LAYERS" \
    --roles $ROLES \
    $(build_tp_arg) \
    $(build_quant_args)

echo ""
echo "Done! Extracted activation files:"
ls -lh "$ACTIVATIONS_DIR/"*.pt 2>/dev/null || echo "(no .pt files found)"
