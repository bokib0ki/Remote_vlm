#!/bin/bash
# 跑完 lever_test2 5 个 json × 2 个模型
# - Qwen2.5-VL-3B thinkOFF (max_new=64)
# - Qwen3-VL-4B-Thinking thinkON (max_new=4096)
# 串行执行（不可并发，会弄乱 raw_outputs/）
# 用法: bash run_lever_test2.sh

set -u
cd /home/admin1/projects/remote_vlm_eval

PY=/home/admin1/miniconda3/envs/vlm-eval/bin/python
SEL_DIR=annotation_data/sampled_eval/lever_test2
LEVERS=(lever_k=20_vqa.json lever_k=50_vqa.json lever_k=100_vqa.json lever_k=200_vqa.json lever_k=400_vqa.json)

LOG_DIR=/tmp/lever_test2_eval_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"
echo "日志目录: $LOG_DIR"
echo "PID: $$"

ts() { date "+%Y-%m-%d %H:%M:%S"; }

# ─── Phase 1: Qwen2.5-VL-3B thinkOFF ───
echo ""
echo "[$(ts)] ============ Phase 1: Qwen2.5-VL-3B thinkOFF ============"
for f in "${LEVERS[@]}"; do
  echo "[$(ts)] ▶ $f"
  $PY eval_select.py --select "$SEL_DIR/$f" --model qwen2.5-vl-3B --max_new 64 \
    2>&1 | tee "$LOG_DIR/3B_$f.log"
  echo "[$(ts)] ✓ $f done"
done
echo "[$(ts)] Phase 1 DONE"

# ─── Phase 2: Qwen3-VL-4B-Thinking thinkON ───
echo ""
echo "[$(ts)] ============ Phase 2: Qwen3-VL-4B-Thinking thinkON ============"
for f in "${LEVERS[@]}"; do
  echo "[$(ts)] ▶ $f"
  $PY eval_select.py --select "$SEL_DIR/$f" --model qwen3-vl-4B-thinking --thinking --max_new 4096 \
    2>&1 | tee "$LOG_DIR/Thinking_$f.log"
  echo "[$(ts)] ✓ $f done"
done
echo "[$(ts)] Phase 2 DONE"

echo ""
echo "[$(ts)] ██████████ ALL DONE ██████████"
date > "$LOG_DIR/DONE.flag"
