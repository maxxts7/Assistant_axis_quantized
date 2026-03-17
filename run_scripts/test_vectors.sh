#!/usr/bin/env bash
# Step 4: Compute per-role vectors (small test run)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

setup_environment
print_config

# Verify inputs exist
if [ ! -d "$ACTIVATIONS_DIR" ]; then
    echo "ERROR: Activations directory not found: $ACTIVATIONS_DIR"
    echo "Run test_activations.sh first."
    exit 1
fi

ACT_COUNT=$(find "$ACTIVATIONS_DIR" -name '*.pt' | wc -l)
if [ "$ACT_COUNT" -eq 0 ]; then
    echo "ERROR: No .pt files found in $ACTIVATIONS_DIR"
    exit 1
fi
echo "Found $ACT_COUNT activation file(s)"

if [ ! -d "$SCORES_DIR" ]; then
    echo "ERROR: Scores directory not found: $SCORES_DIR"
    echo "Run test_judge.sh first."
    exit 1
fi

SCORE_COUNT=$(find "$SCORES_DIR" -name '*.json' | wc -l)
if [ "$SCORE_COUNT" -eq 0 ]; then
    echo "ERROR: No .json files found in $SCORES_DIR"
    exit 1
fi
echo "Found $SCORE_COUNT score file(s)"

echo "Min count: $MIN_COUNT"
echo "Output:    $VECTORS_DIR"
echo ""

uv run "$PROJECT_DIR/pipeline/4_vectors.py" \
    --activations_dir "$ACTIVATIONS_DIR" \
    --scores_dir "$SCORES_DIR" \
    --output_dir "$VECTORS_DIR" \
    --min_count "$MIN_COUNT"

echo ""
echo "Done! Role vector files:"
ls -lh "$VECTORS_DIR/"*.pt 2>/dev/null || echo "(no .pt files found)"
