#!/bin/bash
# 批量跑 L3 NLI 评测（增量式，跳过已跑过的）
# 用法: bash run_l3_nli_batch.sh

set -e
cd /home/admin1/projects/remote_vlm_eval
export HF_HOME=/home/admin1/models/bert_cache

# 方式 1: --batch-dir 批量模式（推荐，VQA_L3_bert.py 内置扫子目录）
python VQA_L3_bert.py --batch-dir /home/admin1/models/raw_outputs/lever_test2 2>&1 | tee /tmp/l3_nli_batch.log

# 方式 2: 手动 loop（更细粒度控制）
# for d in /home/admin1/models/raw_outputs/lever_test2/lever_k=*_vqa/*/; do
#   if [ -f "$d/vqa_l3_raw.json" ] && [ ! -f "$d/vqa_l3_nli.json" ]; then
#     echo "→ $d"
#     python VQA_L3_bert.py "$d/vqa_l3_raw.json" 2>&1 | tail -3
#   fi
# done
