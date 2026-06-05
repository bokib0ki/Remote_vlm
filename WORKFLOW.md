# VLM 评测完整工作流

本工作流覆盖：模型推理 → L3 NLI 评测 → Excel 报告生成 → 验证。

工作目录：`/home/admin1/projects/remote_vlm_eval/`
原始输出目录：`/home/admin1/models/raw_outputs/lever_test2/`
Python 环境：`/home/admin1/miniconda3/envs/vlm-eval/bin/python`

---

## 0. 一次性准备（首次跑评测前）

### 0.1 创建 raw_outputs 目录结构

脚本默认从 `lever_test2/lever_k=*_vqa/{model}_{mode}_*/raw_outputs.json` 读数据。
目录命名必须严格匹配（脚本里 parse_dir_name 用正则提取 model/mode/k）。

```bash
mkdir -p /home/admin1/models/raw_outputs/lever_test2
```

### 0.2 下载 L3 NLI 模型（一次性）

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/admin1/models/bert_cache
python -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
m = 'microsoft/deberta-xlarge-mnli'
AutoTokenizer.from_pretrained(m)
AutoModelForSequenceClassification.from_pretrained(m)
"
```

模型会下载到 `/home/admin1/models/bert_cache/models--microsoft--deberta-xlarge-mnli/`。

---

## 1. 模型推理（生成 raw_outputs.json）

每个 (model, mode, k) 组合一个 run。inference 脚本要输出 `raw_outputs.json`，
每条记录至少包含：`question, gt, pred, correct (0/1), judge_level (L1/L2/L3), tokens, input_tokens, img_tokens, speed, image_id, type`。

跑完一个 run 的示例路径：
```
/home/admin1/models/raw_outputs/lever_test2/lever_k=20_vqa/
  qwen3.5-4B_thinkOFF_lever_test2_lever_k=20_vqa/
    raw_outputs.json     ← 必须有
    vqa_l3_nli.json      ← L3 NLI 评测后才有（步骤 2）
```

`judge_level` 字段是关键：
- **L1**：客观题（颜色、形状、数量、yes/no 等），用规则判断 correct
- **L2**：短语题（位置、方向、场景类型等），用规则 + 词袋判断 correct
- **L3**：开放题（reasoning、image 等），correct 字段填 0，等步骤 2 用 NLI 评测

可以参考 `VQA_V3.py`（旧 API 评测版本，已废弃）了解如何用 Qwen 评判。
新版（inference.py）在 `inference.py` 里（如果有 max_thinking_tokens 参数更佳）。

---

## 2. L3 语义评测（NLI 头版本）

针对每个有 L3 题的 run，跑一次 NLI 评测。**NLI 头比 BERTScore 准 16pp**
（BERTScore 在 DeBERTa embedding 空间里把反义词的 cosine 算得很高，
例如 smaller/larger = 0.991 recall，导致 100% 反义误判）。

VQA_L3_bert.py 读 `vqa_l3_raw.json`（不是 raw_outputs.json），输出 `vqa_l3_nli.json`。

```bash
cd /home/admin1/projects/remote_vlm_eval
export HF_HOME=/home/admin1/models/bert_cache

# 方式 1（推荐）: --batch-dir 批量模式，脚本内置扫子目录 + 跳过已跑过的
python VQA_L3_bert.py --batch-dir /home/admin1/models/raw_outputs/lever_test2

# 方式 2: 手动 loop 单文件
# for d in /home/admin1/models/raw_outputs/lever_test2/lever_k=*_vqa/*/; do
#   if [ -f "$d/vqa_l3_raw.json" ] && [ ! -f "$d/vqa_l3_nli.json" ]; then
#     echo "→ $(basename $d)"
#     python VQA_L3_bert.py "$d/vqa_l3_raw.json" 2>&1 | tail -3
#   fi
# done
```

注意：
- `--batch-dir` 模式自动扫 `**/vqa_l3_raw.json`，已存在 `vqa_l3_nli.json` 的 run 会跳过
- 加 `--force` 可覆盖重跑
- GPU 占用约 3.5GB（fp32）。如果 GPU 被占，临时用 `--device cpu`（慢 10-20 倍）
- 默认 batch_size=32，可通过 `--batch-size` 调

跑完每个 run 会在同目录生成 `vqa_l3_nli.json`（每条 L3 题的 entailment/neutral/contradiction 概率 + correct_l3）。

### NLI 阈值

- 公式：`is_correct = (entailment_prob >= 0.5) AND (entailment_prob > contradiction_prob)`
- 0.5 是 MNLI 标准阈值
- DeBERTa-xlarge-mnli 标签顺序（**非标准 MNLI 顺序**）：`{0: CONTRADICTION, 1: NEUTRAL, 2: ENTAILMENT}`

预期 L3 acc：约 28-30%（参考 28 个 run 跑出来的 28.82%）。

---

## 3. 生成 Excel 报告

```bash
cd /home/admin1/projects/remote_vlm_eval
/home/admin1/miniconda3/envs/vlm-eval/bin/python gen_vqa_type_report_v3.py
```

输出：`excel_result/VQA_type_report_v3.xlsx`，3 个 sheet：
- **per_lever**（51 行 × 13 列）：每 run 一行
  - 模型/模式/k/总N / 4 perf 列 / **(L1-L3) 整体 acc + 排名** / **L3 N** / **L1+L2 严格版 acc + 排名**
  - **(L1-L3)**：有 L3 评测时含 L3 答对；无 L3 评测时退化为 (L1_c+L2_c)/N_total
  - **L1+L2 严格版** = (L1_c + L2_c) / (L1 + L2 + L3_pending)，L3 算入分母按错算
- **type_acc**（44 行 × 28 列）：12 type 准确率 + 12 type per-rank + TOTAL + TOTAL 排名
- **raw_concat**（57675 条）：全部原始记录扁平

### 常用 CLI 参数

```bash
# 指定不同 raw_root（如 batch1/batch3_sel/bench4_sel 旧数据）
python gen_vqa_type_report_v3.py --raw-root /home/admin1/models/raw_outputs/batch1

# 只看 thinkOFF
python gen_vqa_type_report_v3.py --include-mode thinkOFF

# 输出不同文件名
python gen_vqa_type_report_v3.py --name my_report
```

---

## 4. 验证报告（必跑！）

脚本可能因为 raw 数据缺失、字段错位等原因生成"看起来对实际错"的报告。
**至少跑下面 4 个验证**：

### 4.1 验证 L3 评测覆盖率

```bash
# 应该每个 (model, mode, k) 都覆盖
echo "=== 哪些 run 跑了 L3 NLI ==="
for d in /home/admin1/models/raw_outputs/lever_test2/lever_k=*_vqa/*/; do
  if [ -f "$d/raw_outputs.json" ]; then
    if [ -f "$d/vqa_l3_nli.json" ]; then
      echo "  ✓ $(basename $d)"
    else
      echo "  ✗ $(basename $d)  ← 没跑 NLI"
    fi
  fi
done
```

### 4.2 验证 Excel 行数 vs raw 文件数

```bash
python -c "
from openpyxl import load_workbook
import json
wb = load_workbook('/home/admin1/projects/remote_vlm_eval/excel_result/VQA_type_report_v3.xlsx', read_only=True)
ws_raw = wb['raw_concat']
n_excel = sum(1 for _ in ws_raw.iter_rows(min_row=2)) - 1
n_dir = 0
import os, glob
for f in glob.glob('/home/admin1/models/raw_outputs/lever_test2/lever_k=*_vqa/*/raw_outputs.json'):
    n_dir += sum(1 for _ in open(f))
print(f'raw_concat 行数: {n_excel}')
print(f'raw_outputs.json 总记录: {n_dir}')
print(f'差异: {n_excel - n_dir}  ← 应为 0')
"
```

### 4.3 验证 per_lever 三大口径

```bash
python -c "
from openpyxl import load_workbook
wb = load_workbook('/home/admin1/projects/remote_vlm_eval/excel_result/VQA_type_report_v3.xlsx', read_only=True)
ws = wb['per_lever']
print(f'{\"模型\":<22} {\"模式\":<10} {\"k\":>3} {\"N\":>5} {\"(L1-L3)\":>9} {\"L1+L2\":>9} {\"关系\"}')
for r in range(3, ws.max_row+1):
    m, md, k, n, _, _, _, _, l1l3, _, l3n, l1l2, _ = [ws.cell(r, c).value for c in range(1, 14)]
    if m is None: continue
    if k in (20, 40, 70):
        rel = '<' if (l1l3 or '0') > (l1l2 or '0') else ('=' if l1l3 == l1l2 else '?')
        print(f'{str(m)[:22]:<22} {str(md)[:10]:<10} {k!s:>3} {n!s:>5} {str(l1l3):>9} {str(l1l2):>9}  {rel}')
"
```

**预期**：
- k=20（有 L3）：`(L1-L3) > L1+L2`，差值是 L3 答对贡献（5-15pp）
- k=40/70（无 L3）：`(L1-L3) = L1+L2`（两列同公式）
- L3 N 列在无 L3 run 上为 `—`

### 4.4 验证 NLI 阈值合理性（每 1-2 周抽样）

```bash
python -c "
import json, glob
from collections import Counter
ent = []
for f in glob.glob('/home/admin1/models/raw_outputs/lever_test2/**/vqa_l3_nli.json', recursive=True):
    recs = json.load(open(f))
    for r in recs:
        if r.get('e_p') is not None:
            ent.append((r['e_p'], r['c_p'], r.get('correct_l3', 0)))
import statistics
print(f'NLI 样本: {len(ent)}')
print(f'entailment 均值: {statistics.mean(e for e,_,_ in ent):.3f}')
print(f'entailment 中位: {statistics.median(e for e,_,_ in ent):.3f}')
above = sum(1 for e,_,_ in ent if e >= 0.5)
print(f'≥0.5 的: {above}/{len(ent)} = {above/len(ent)*100:.1f}%')
"
```

**预期**：entailment 中位 ~0.2，均值 ~0.3，≥0.5 的约 30-40%。

---

## 5. 已知坑（踩过的）

1. **NLI 头 vs BERTScore**：必须用 NLI 头。BERTScore 反义 100% 误判。
2. **fp32 而非 fp16**：DeBERTa-v2-xlarge fp16 下有 matmul dtype 冲突。
3. **本地模型路径**：传 `models--microsoft--deberta-xlarge-mnli/snapshots/<hash>/` 路径，不要传 model id。
4. **judge_level 字段**：raw_outputs.json 里每条 L3 题的 `correct` 字段必须填 0（默认），等 NLI 覆盖。
5. **目录命名**：必须匹配 `parse_dir_name` 正则，否则 discover_runs 跳过。
6. **eval.py 的 max_new bug**：MME-RS/VQA 评测脚本里 MME=10, VQA=64 hardcoded，会覆盖用户的 --max_new。如果用那个脚本跑 thinkON 必须修。
7. **thinkON 循环死锁**：Qwen3-VL-4B-Thinking 在 4000+ tokens 区间 60% 循环。max_new=4096 会强制截断但也会损失答案。
8. **Qwen3.5-4B 数据散落**：raw_concat 只扫 lever_test2，batch1/batch3_sel/bench4_sel 数据需手动统计。
9. **NLI 与 eval 抢 GPU**：eval_select.py 占满 21GB 时 NLI 跑不起来。可 `--device cpu`（慢 10-20 倍）或 `kill -STOP <pid>` 暂停 eval 让路。NLI 增量式跑，已有的 `vqa_l3_nli.json` 自动跳过。

---

## 6. 关键文件

| 文件 | 用途 |
|---|---|
| `inference.py` | 模型推理（VQA 评测） |
| `VQA_L3_bert.py` | L3 NLI 评测（DeBERTa-xlarge-mnli） |
| `gen_vqa_type_report_v3.py` | Excel 报告生成器 |
| `gen_vrs_vqa_data_analysis.py` | VRS-VQA 数据集统计（37k 条） |
| `excel_result/VQA_type_report_v3.xlsx` | 最新评测报告 |
| `excel_result/VRS-VQA-DataAnalysis.xlsx` | VRS-VQA 数据集统计 |
| `WORKFLOW.md` | 本文档 |

skill: `mlops/vrs-vqa-l3-nli`（NLI 评测详细文档）
