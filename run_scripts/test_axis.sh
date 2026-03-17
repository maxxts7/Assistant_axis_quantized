#!/usr/bin/env bash
# Step 5: Compute assistant axis (small test run)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

setup_environment
print_config

# Verify vectors exist
if [ ! -d "$VECTORS_DIR" ]; then
    echo "ERROR: Vectors directory not found: $VECTORS_DIR"
    echo "Run test_vectors.sh first."
    exit 1
fi

VEC_COUNT=$(find "$VECTORS_DIR" -name '*.pt' | wc -l)
if [ "$VEC_COUNT" -eq 0 ]; then
    echo "ERROR: No .pt files found in $VECTORS_DIR"
    exit 1
fi
echo "Found $VEC_COUNT vector file(s)"

echo "Output: $AXIS_OUTPUT"
echo ""

uv run "$PROJECT_DIR/pipeline/5_axis.py" \
    --vectors_dir "$VECTORS_DIR" \
    --output "$AXIS_OUTPUT"

echo ""
echo "Done! Axis file:"
ls -lh "$AXIS_OUTPUT" 2>/dev/null || echo "(axis file not found)"
