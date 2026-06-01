#!/usr/bin/env bash
# 一键运行全量 thinkOFF 评测
# 用法: bash run_all.sh
set -e

BATCH=${1:-1}
echo "Running batch$BATCH thinkOFF for all 8 models..."

# 逐个运行（serial，避免显存不足）
for MODEL in minicpm-v-4.6 qwen3.5-0.8B qwen3.5-2B qwen3.5-4B \
             qwen3-vl-2B qwen3-vl-4B gemma-4-e2b gemma-4-e4b; do
    echo ""
    echo "=========================================="
    echo "  $MODEL (batch=$BATCH)"
    echo "=========================================="
    python3 eval.py --model "$MODEL" --batch "$BATCH"
done

echo ""
echo "=== All done. Generating Excel... ==="
python3 gen_excel.py --batch "$BATCH"

echo "=== Done! ==="
