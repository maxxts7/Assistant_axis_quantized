#!/usr/bin/env bash
# Step 3: Score responses with LLM judge (small test run)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

setup_environment
print_config

# Check API key
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set."
    echo "Usage: OPENAI_API_KEY=sk-... ./run_scripts/test_judge.sh"
    exit 1
fi

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

echo "Judge model: $JUDGE_MODEL"
echo "Batch size:  $JUDGE_BATCH_SIZE"
echo "Output:      $SCORES_DIR"
echo ""

uv run "$PROJECT_DIR/pipeline/3_judge.py" \
    --responses_dir "$RESPONSES_DIR" \
    --roles_dir "$PROJECT_DIR/data/roles/instructions" \
    --output_dir "$SCORES_DIR" \
    --judge_model "$JUDGE_MODEL" \
    --batch_size "$JUDGE_BATCH_SIZE" \
    --roles $ROLES

echo ""
echo "Done! Score files:"
ls -lh "$SCORES_DIR/"*.json 2>/dev/null || echo "(no .json files found)"
