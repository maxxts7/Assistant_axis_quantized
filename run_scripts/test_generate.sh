#!/usr/bin/env bash
# Step 1: Generate responses (small test run)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

setup_environment
check_gpu
print_config

echo "Questions: $QUESTION_COUNT per role (x5 prompts x $(echo $ROLES | wc -w) roles)"
echo "Output:    $RESPONSES_DIR"
echo ""

uv run "$PROJECT_DIR/pipeline/1_generate.py" \
    --model "$MODEL" \
    --roles_dir "$PROJECT_DIR/data/roles/instructions" \
    --questions_file "$PROJECT_DIR/data/extraction_questions.jsonl" \
    --output_dir "$RESPONSES_DIR" \
    --question_count "$QUESTION_COUNT" \
    --gpu_memory_utilization "$GPU_MEM_UTIL" \
    --roles $ROLES \
    $(build_tp_arg) \
    $(build_quant_args)

echo ""
echo "Done! Generated files:"
ls -lh "$RESPONSES_DIR/"*.jsonl 2>/dev/null
