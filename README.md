# VLM 遥感评测框架 — remote_vlm_eval

生成时间：2026-06-01

基于 transformers 的零样本遥感 VLM 评测框架。支持 4 个 benchmark（VRSBench / MME-RS / LEVIR-CC / XLRS）× 多个模型，含 thinkON/thinkOFF 双模式对比。

**核心特性**：
- 所有评测严格使用 `temperature=0.0`（greedy 解码）
- 评测方式完全按照 `bench_cofig/` 中的官方 benchmark 配置
- 支持 VQA 三级评测（L1 substring → L2 yes/no/数字 → L3 LLM 语义）
- 支持 selection JSON 题目清单评测（按需采样，不跑全量）

# 一、数据集说明

## 1\.1 VRSBench Caption

- 来源：xiang709/VRSBench \(HuggingFace\)

- 图片量：9,351 张（验证集 Images\_val/）

- 标注量：877 条（VRSBench\_EVAL\_Cap\.json）

- 采样：50 条 × 2 batch

- 任务：遥感图像描述（Scene\-level Caption）

- 指标：BLEU\-1\~4, ROUGE\-L, CIDER

- Prompt：\&\#34;Describe this remote sensing image in detail\.\&\#34;

## 1\.2 VRSBench VQA

- 图片量：9,351 张（同上）

- 标注量：1,958 条（VRSBench\_EVAL\_vqa\.json）

- 任务：遥感 VQA（12 种类型）

- 指标：L1 substring / L3 LLM 语义判定

- Prompt：GeoChat 官方 prompt

## 1\.3 MME\-RealWorld RS

- 来源：yifanzhang114/MME\-RealWorld\-Base64

- 图片量：1,300 张

- 标注量：1,265 条 RS 子集

- 任务：多选 VQA

- 指标：准确率

## 1\.4 LEVIR\-CC

- 来源：Chen\-Yang\-Liu/LEVIR\-CC\-Dataset

- 图片量：2,135 对（A=before, B=after）

- 标注量：2,135 条，每样本 2\-3 参考

- 任务：变化描述（双图输入）

- 指标：BLEU\-1\~4, ROUGE\-L, CIDER（多参考）

## 1\.5 XLRS\-Bench

- 来源：initiacms/XLRS\-Bench\-lite

- 图片量：42 张（超高分辨率 \~2500x2500）

- 标注量：42 条（全量）

- 任务：超高分辨率多选 VQA

- 指标：准确率

# 二、目录结构

```
remote_vlm_eval/                    项目根目录（/home/admin1/projects/remote_vlm_eval）
├── config.py                      全局配置（路径、模型列表）
├── inference.py                   模型加载与推理（单/双图，thinkON/OFF）
├── metrics.py                      BLEU/ROUGE/CIDER 评分接口（pycocoevalcap）
├── datasets.py                     5个数据集的加载函数
├── eval_select.py                  【推荐】按 selection JSON 题目清单评测
│                                  严格遵循 bench_cofig/ 官方配置
├── bench_config_loader.py          bench_cofig 配置加载器
├── vqa_judge.py                    VRSBench VQA 三级评测器（L1/L2/L3）
├── gen_excel_sel.py                selection 评测的 Excel 报告生成
├── create_feishu_doc*.py           飞书文档生成（旧版，可忽略）
├── analyze_selection_tasks.py       任务分析工具
├── requirements.txt
├── README.md
└── Agent_trash/                    临时文件目录（可忽略）

bench_cofig/                        官方 Benchmark 配置（核心）
├── VRSBench/
│   ├── vrs_vqa/                   VRS-VQA 配置
│   │   └── config.py             TEMPERATURE=0, PROMPT, EVAL_METHOD, MAX_NEW
│   └── vrs_caption/               VRS-Caption 配置
│       └── config.py
├── MME_RS/                        MME-RS 配置
│   └── config.py
├── LEVIR_CC/                      LEVIR-CC 配置
│   └── config.py
├── XLRS/                          XLRS 配置
│   └── config.py
└── bench_src/                     各 benchmark 官方源码（只读参考）
    └── VRSBench_src/
        ├── eval_fianl/            VRSBench 官方评测脚本
        │   ├── eval_vqa_gpt.ipynb  VQA 三级评测源码（L1/L2/GPT）
        │   └── compute_metrics.ipynb Caption 指标计算源码
        └── rs_instruction.txt      数据生成 prompt

annotation_data/                    评测标注数据
├── full/                          完整标注
│   ├── VRSBench_EVAL_Cap.json    VRS-Caption（877条）
│   ├── VRSBench_EVAL_vqa.json     VRS-VQA（1958条）
│   ├── mme_rs_annotations.json    MME-RS（1265条）
│   ├── LevirCCcaptions.json       LEVIR-CC（2135条）
│   └── xlrs_samples_42.json       XLRS（42条）
└── sampled_eval/                   采样数据（按 selection JSON 跑子集）
    └── lever_test/
        └── lever_k=50_sel.json    LEVIR-CC 50条采样

sota_api/                           SOTA API 评测（OpenAI/Gemini 百炼）
├── qwen_3.6_plus.py
└── config_sota.py
```

**重要**：
- `bench_cofig/` 中的每个 `config.py` 定义了官方评测方式
- `eval_select.py` 必须从 `bench_cofig/` 读取配置，不允许回退
- 所有评测默认使用 `temperature=0.0`（严格 greedy 解码）
- 图片数据在 `models/` 目录（项目根目录外）

# 三、评测流程

## 环境准备

```bash
pip install -r requirements.txt
```

## 推理评测

单模型，按照抽样所得的题目清单进行评测（使用 bench_cofig 官方配置）：

```bash
python3 eval_select.py --select annotation_data/sampled_eval/lever_test/lever_k=50_sel.json --model qwen3.5-4B
```

单模型，评测 bench 中的某一数据集的任务：

```bash
python3 eval_select.py --select annotation_data/sampled_eval/lever_test/lever_k=50_sel.json --model qwen3.5-4B --bench vrs_vqa
```

多模型批量评测：

```bash
python3 eval_select.py --select annotation_data/sampled_eval/lever_test/lever_k=50_sel.json --model qwen3.5-0.8B,qwen3.5-2B
```

所有模型批量评测（包含 thinkON/thinkOFF 双模式）：

```bash
python3 eval_select.py --select annotation_data/sampled_eval/lever_test/lever_k=50_sel.json --all
```

## 生成 Excel 报告

selection 评测结果导出 Excel：

```bash
python3 gen_excel_sel.py --select annotation_data/sampled_eval/lever_test/lever_k=50_sel.json
```

输出指标：指标对比表（含SOTA）+ 原始输出 + 分析总结



# 四、源码详解

## config\.py — 全局配置

```Plain Text
"""
全局配置 — 路径、模型列表、prompt 模板。
运行前请根据实际环境修改 ROOT。
"""
from pathlib import Path

# ─── 根目录 ───────────────────────────────────────────────
ROOT = Path('/home/admin1/models')

# ─── 模型列表 ─────────────────────────────────────────────
ALL_MODELS = [
    'minicpm-v-4.6',  # MiniCPM-V 4.6B (thinkON 支持)
    'qwen3.5-0.8B',   # Qwen3.5-VL 0.8B (thinkON 支持)
    'qwen3.5-2B',     # Qwen3.5-VL 2B (thinkON 支持)
    'qwen3.5-4B',     # Qwen3.5-VL 4B (thinkON 支持)
    'qwen3-vl-2B',    # Qwen3-VL 2B
    'qwen3-vl-4B',    # Qwen3-VL 4B
    'gemma-4-e2b',    # Gemma-4 2B
    'gemma-4-e4b',    # Gemma-4 4B
]
DUAL_MODELS = ['minicpm-v-4.6', 'qwen3.5-0.8B', 'qwen3.5-2B', 'qwen3.5-4B']
# ↑ 支持 thinkON/thinkOFF 双模式的模型

# ─── 图片目录 ─────────────────────────────────────────────
VRS_IMG_DIR   = ROOT / 'vrsbench_images' / 'Images_val'        # 9351 张
LEVIR_DIR     = ROOT / 'levircc_data' / 'extracted' / 'images' / 'test'  # 2135 对 (A/B)
MME_RS_DIR    = ROOT / 'mme_images' / 'remote_sensing_full' / 'remote_sensing'  # 1300 张
TEST_IMGS     = sorted((ROOT / 'test_images').glob('*.*'))      # 备用占位图

# ─── 采样文件目录 ──────────────────────────────────────────
# batch1 = 前50样本, batch2 = 后50样本, full = 全部
SAMP = ROOT / 'sampled_eval' / 'batch1'   # 切换时改这里

# ─── 输出目录 ─────────────────────────────────────────────
RAW_DIR_ROOT = ROOT / 'raw_outputs'       # 推理原始输出
RES_DIR      = ROOT / 'results'           # JSON 指标
OUT_DIR      = ROOT / '..' / 'projects'   # Excel 报告

# ─── 推理参数 ─────────────────────────────────────────────
MAX_NEW_TOKENS = 128       # caption/change 任务
MAX_NEW_VQA    = 64        # VQA 短答
DO_SAMPLE      = False     # greedy 解码

# ─── Prompt 模板（官方/论文标准） ────────────────────────

CAPTION_PROMPT = "Describe this remote sensing image in detail."
# VRSBench Caption 官方 prompt

LEVIR_PROMPT = (
    "Describe the changes between these two remote sensing images "
    "taken at different times."
)
# LEVIR-CC 官方 prompt（双图输入）

VRS_VQA_PROMPT_TPL = "{question}\nAnswer the question using a single word or phrase."
# GeoChat / VRSBench 官方 VQA prompt

MME_PROMPT_TPL = (
    "{question}\nThe choices are listed below:\n{choices}\n"
    "Select the best answer to the above multiple-choice question based on the image. "
    "Respond with only the letter (A, B, C, D, or E) of the correct option.\n"
    "The best answer is:"
)
# MME-RealWorld 官方 prompt

XLRS_PROMPT_TPL = (
    "{question}\nThe choices are listed below:\n{choices}\n"
    "Select the best answer for the multiple-choice question based on the image. "
    "Only respond with the letter corresponding to the correct answer (A, B, C, D).\n"
    "The answer is:"
)
# XLRS-Bench 官方 prompt

# ─── 指标定义（Excel 列顺序） ─────────────────────────────
METRICS = [
    'mme_rs', 'xlrs',
    'vrs_cap_bleu4', 'vrs_cap_rouge_l', 'vrs_cap_cider',
    'levir_bleu4', 'levir_rouge_l', 'levir_cider',
    'vrs_vqa',
]
MLABELS = [
    'MME-RS', 'XLRS',
    'VRS-Cap B4', 'VRS-Cap R-L', 'VRS-Cap CIDER',
    'LEVIR B4', 'LEVIR R-L', 'LEVIR CIDER',
    'VRS-VQA',
]
MINFO = {
    'mme_rs':           'MME-RealWorld RS 多选VQA准确率',
    'xlrs':             'XLRS-Bench 超高分辨率多选VQA',
    'vrs_cap_bleu4':    'VRSBench Caption (BLEU-4)',
    'vrs_cap_rouge_l':  'VRSBench Caption (ROUGE-L)',
    'vrs_cap_cider':    'VRSBench Caption (CIDER)',
    'levir_bleu4':      'LEVIR-CC 变化描述 (BLEU-4)',
    'levir_rouge_l':    'LEVIR-CC 变化描述 (ROUGE-L)',
    'levir_cider':      'LEVIR-CC 变化描述 (CIDER)',
    'vrs_vqa':          'VRSBench VQA (L1 substring 准确率)',
}

```

## inference\.py — 模型加载与推理

```Plain Text
"""
模型加载和推理。
支持 thinkOFF / thinkON 双模式，支持单图/双图输入。
"""
import re
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from config import DO_SAMPLE


def load_model(model_name: str, model_root: str):
    """加载 VLM 模型和 processor。"""
    path = f"{model_root}/{model_name}"
    model = AutoModelForImageTextToText.from_pretrained(
        path, torch_dtype=torch.bfloat16,
        device_map='auto', trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        path, trust_remote_code=True,
    )
    return model, processor


def infer(
    model, processor,
    img: Image.Image,
    prompt: str,
    max_new_tokens: int = 128,
    extra_imgs: list | None = None,
    enable_thinking: bool = False,
    do_sample: bool = DO_SAMPLE,
) -> tuple[str, int]:
    """
    单次推理。

    Args:
        img: 主图片（PIL RGB）
        prompt: 文本 prompt
        max_new_tokens: 最大生成 token 数
        extra_imgs: 额外图片（如 LEVIR-CC 的 after 图）
        enable_thinking: 是否开启 thinkON 模式

    Returns:
        (输出文本, 实际生成 token 数)
    """
    content = [{"type": "image", "image": img}]
    if extra_imgs:
        for ei in extra_imgs:
            content.append({"type": "image", "image": ei})
    content.append({"type": "text", "text": prompt})

    conv = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        conv, add_generation_prompt=True,
        tokenize=False, enable_thinking=enable_thinking,
    )

    if extra_imgs:
        all_imgs = [img] + extra_imgs
        inputs = processor(images=all_imgs, text=text,
                           return_tensors="pt").to('cuda:0')
    else:
        inputs = processor(images=img, text=text,
                           return_tensors="pt").to('cuda:0')

    input_len = inputs['input_ids'].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
    gen = out[:, input_len:]
    out_text = processor.batch_decode(
        gen, skip_special_tokens=True)[0].strip()
    out_tokens = gen.shape[1]
    return out_text, out_tokens


def strip_thinking(text: str) -> str:
    """清除 thinkON 模式产生的 thinking 块。"""
    text = re.sub(r'<think>.*?</think>\s*\n?\s*', '', text, flags=re.DOTALL)
    text = re.sub(r'<thinking>.*?</thinking>\s*\n?\s*', '', text, flags=re.DOTALL)
    return text.strip()


def extract_letter(text: str, choices: str = 'ABCDE') -> str:
    """
    从模型输出中提取选项字母。
    支持多种常见格式：
      - "The best answer is: A"
      - "(B)"
      - "C."
      - "D" (行末独立字母)
    """
    s = text.strip().upper()
    # 精确匹配各种前缀模式
    for pat in [
        r'(?:ANSWER|OPTION)\s+(?:IS|:)\s*[(]?([' + choices + r'])[)]?',
        r'CORRECT\s+(?:ANSWER|OPTION)\s+IS\s+[(]?([' + choices + r'])[)]?',
        r'THE\s+BEST\s+ANSWER\s+IS\s+[(]?([' + choices + r'])[)]?',
        r'IS\s+CORRECT\s*[.:]?\s*[(]?([' + choices + r'])[)]?',
    ]:
        m = re.findall(pat, s)
        if m:
            return m[-1]
    # 括号内字母
    parenthesized = re.findall(r'\(' + choices + r'\)', s)
    if parenthesized:
        return parenthesized[-1].strip('()')
    # 行首/空白后的独立字母
    letters = re.findall(
        r'(?:^|[\s(])([' + choices + r'])(?:\)|\]|\}|\.|,|\s|$)',
        s,
    )
    if letters:
        return letters[-1]
    return ""


def safe_img(path, fallback_dir=None):
    """加载图片，不存在时用占位图代替。"""
    p = Path(path)
    if p.exists():
        return Image.open(p).convert('RGB')
    if fallback_dir:
        fallbacks = sorted(Path(fallback_dir).glob('*.*'))
        if fallbacks:
            return Image.open(fallbacks[0]).convert('RGB')
    return Image.new('RGB', (512, 512), 'gray')

```

## metrics\.py — BLEU/ROUGE/CIDER 计算

```Plain Text
"""
评分指标 — BLEU, ROUGE-L, CIDER。
封装 pycocoevalcap，提供统一接口。
"""
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


def compute_caption_scores(refs_dict: dict, hyps_dict: dict) -> dict:
    """
    计算 caption 类任务的 BLEU-1~4, ROUGE-L, CIDER。

    Args:
        refs_dict: {sample_id: [ref1, ref2, ...]}
        hyps_dict: {sample_id: [hypothesis]}

    Returns:
        {'bleu1': ..., 'bleu4': ..., 'rouge_l': ..., 'cider': ...}
    """
    bs, _ = Bleu(4).compute_score(refs_dict, hyps_dict)
    rouge_s, _ = Rouge().compute_score(refs_dict, hyps_dict)
    cider_s, _ = Cider().compute_score(refs_dict, hyps_dict)
    return {
        **{f'bleu{i+1}': round(bs[i], 4) for i in range(4)},
        'rouge_l': round(rouge_s, 4),
        'cider': round(cider_s, 4),
    }

```

## datasets\.py — 数据集加载

```Plain Text
"""
数据集加载 — 读取采样文件/完整标注，统一返回记录列表。

每条记录包含评测所需字段：image_path, question, ground_truth, choices 等。
"""
import json
from pathlib import Path

from config import (
    ROOT, VRS_IMG_DIR, LEVIR_DIR, MME_RS_DIR,
    SAMP,
)


def load_vrs_caption(sample_dir=None, full=False):
    """
    加载 VRSBench Caption 数据集。

    完整标注：VRSBench_EVAL_Cap.json (4.6MB, ~877 条)
    采样文件：sampled_eval/batch1/vrs_cap_sampled.json (50 条)

    每个记录字段: image_id, ground_truth
    """
    if sample_dir or (not full):
        path = (sample_dir or SAMP) / 'vrs_cap_sampled.json'
        if not path.exists():
            path = ROOT / 'sampled_eval' / 'batch1' / 'vrs_cap_sampled.json'
    else:
        path = ROOT / 'VRSBench_EVAL_Cap.json'
    with open(path) as f:
        anns = json.load(f)
    return anns


def load_vrs_vqa(sample_dir=None, full=False):
    """
    加载 VRSBench VQA 数据集。

    完整标注：VRSBench_EVAL_vqa.json (9.0MB, ~1958 条)
    采样文件：vrs_vqa_sampled.json (50 条)

    每个记录字段: image_id, question, ground_truth/answer, type/question_type
    """
    if sample_dir or (not full):
        path = (sample_dir or SAMP) / 'vrs_vqa_sampled.json'
        if not path.exists():
            path = ROOT / 'sampled_eval' / 'batch1' / 'vrs_vqa_sampled.json'
    else:
        path = ROOT / 'VRSBench_EVAL_vqa.json'
    with open(path) as f:
        anns = json.load(f)
    return anns


def load_mme_rs(sample_dir=None, full=False):
    """
    加载 MME-RealWorld Remote Sensing 子集。

    完整标注：mme_rs_annotations.json (2.1MB, ~1265 条 RS 问题)
    采样文件：mme_rs_sampled.json (50 条)

    注意：MME-RS 的 "Image" 字段是相对路径，需拼接 MME_RS_DIR 取文件名。
    """
    if sample_dir or (not full):
        path = (sample_dir or SAMP) / 'mme_rs_sampled.json'
        if not path.exists():
            path = ROOT / 'sampled_eval' / 'batch1' / 'mme_rs_sampled.json'
    else:
        path = ROOT / 'mme_rs_annotations.json'
    with open(path) as f:
        anns = json.load(f)
    return anns


def load_xlrs(sample_dir=None, full=False):
    """
    加载 XLRS-Bench 数据集。

    完整标注：xlrs_arrow/xlrs_samples_42.json (39KB, 42 条)
    采样文件：xlrs_full.json (42 条，全量)

    每个记录字段: question, multi-choice options, answer, local_image, path
    注意：图片是超高分辨率（~2500x2500）卫星图。
    """
    if sample_dir or (not full):
        path = (sample_dir or SAMP) / 'xlrs_full.json'
        if not path.exists():
            path = ROOT / 'sampled_eval' / 'batch1' / 'xlrs_full.json'
    else:
        path = ROOT / 'xlrs_arrow' / 'xlrs_samples_42.json'
    with open(path) as f:
        anns = json.load(f)
    return anns


def load_levir_cc(sample_dir=None, full=False):
    """
    加载 LEVIR-CC 变化描述数据集。

    完整标注：levircc_data/extracted/LevirCCcaptions.json (~2135 条 test)
    采样文件：levir_cc_sampled.json (50 条, 含 "images" 外壳)

    每个记录字段: filename, sentences[{raw}], filepath='test'
    双图路径：LEVIR_DIR/A/{filename} (before) + LEVIR_DIR/B/{filename} (after)
    """
    if sample_dir or (not full):
        path = (sample_dir or SAMP) / 'levir_cc_sampled.json'
        if not path.exists():
            path = ROOT / 'sampled_eval' / 'batch1' / 'levir_cc_sampled.json'
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            anns = [a for a in data.get('images', data) if a.get('filepath') == 'test']
            if not anns:
                anns = data.get('images', data)
            return anns
    # fallback: 从完整标注加载
    from config import LEVIR_ANN_PATH
    if LEVIR_ANN_PATH and LEVIR_ANN_PATH.exists():
        from datasets import load_levir as _ll
        return _ll(full=True)
    return []


# ─── 图片路径解析 ──────────────────────────────────────────

def get_vrs_image(image_id: str) -> str:
    """返回 VRSBench 图片的绝对路径。"""
    return str(VRS_IMG_DIR / image_id)


def get_levir_image(filename: str, which: str = 'A') -> str:
    """返回 LEVIR-CC 图片路径。which = 'A'(before) 或 'B'(after)。"""
    return str(LEVIR_DIR / which / filename)


def get_mme_image(img_path_str: str) -> str:
    """返回 MME-RS 图片的绝对路径（文件名匹配）。"""
    from pathlib import Path
    return str(MME_RS_DIR / Path(img_path_str).name)

```

## eval\.py — 主评测入口

```Plain Text
#!
"""
主评测入口 — 对指定模型 × 模式运行全部 benchmark。
支持单次（单模型）和批量（所有模型）两种模式。

用法:
  # 单模型评测
  python3 eval.py --model qwen3.5-0.8B
  python3 eval.py --model qwen3.5-0.8B --thinking --max_new 4096 --skip_caption

  # 批量评测所有模型（thinkOFF）
  python3 eval.py --all

  # 指定 batch
  python3 eval.py --model minicpm-v-4.6 --batch 2
"""
import os, sys, json, time, argparse
from pathlib import Path

import torch
from PIL import Image

from config import (
    ROOT, ALL_MODELS, DUAL_MODELS,
    VRS_IMG_DIR, LEVIR_DIR, MME_RS_DIR, TEST_IMGS,
    SAMP, RAW_DIR_ROOT, RES_DIR, OUT_DIR,
    CAPTION_PROMPT, LEVIR_PROMPT,
    VRS_VQA_PROMPT_TPL, MME_PROMPT_TPL, XLRS_PROMPT_TPL,
)
from inference import load_model, infer, strip_thinking, extract_letter, safe_img
from metrics import compute_caption_scores

print = lambda *a, **kw: __builtins__.print(*a, **kw, flush=True)


# ─── 参数 ──────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='模型名称')
parser.add_argument('--all', action='store_true', help='批量评测所有模型')
parser.add_argument('--thinking', action='store_true', help='启用 thinkON 模式')
parser.add_argument('--batch', type=int, default=1, help='batch 编号')
parser.add_argument('--max_new', type=int, default=128, help='最大生成 tokens')
parser.add_argument('--skip_caption', action='store_true', help='跳过 caption 类任务')
args = parser.parse_args()

BATCH = args.batch
SAMPLE_DIR = SAMP.parent / f'batch{BATCH}' if SAMP.parent else Path(f'/home/admin1/models/sampled_eval/batch{BATCH}')
MAX_NEW = args.max_new
SKIP_CAP = args.skip_caption
MODE_TAG = '_thinkON' if args.thinking else '_thinkOFF'


def get_test_img(i=0):
    """安全获取测试图片。"""
    if TEST_IMGS:
        return safe_img(str(TEST_IMGS[i % len(TEST_IMGS)]),
                         fallback_dir=str(TEST_IMGS[0].parent))
    return Image.new('RGB', (512, 512), 'gray')


def run_eval(model_name: str, enable_thinking: bool = False):
    """对单个模型运行全部 benchmark。"""
    config_tag = f"{model_name}{MODE_TAG}_batch{BATCH}"
    print(f"\n{'='*60}")
    print(f"  {config_tag}")
    print(f"{'='*60}")

    # 加载模型
    model, processor = load_model(model_name, str(ROOT))
    total_vram = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM: {total_vram:.1f}GB")

    def _run(img, prompt, extra_imgs=None, max_new=MAX_NEW):
        pred, nt = infer(
            model, processor, img, prompt,
            max_new_tokens=max_new,
            extra_imgs=extra_imgs,
            enable_thinking=enable_thinking,
        )
        clean = strip_thinking(pred) if enable_thinking else pred
        return clean, pred, nt

    row = {
        'config': config_tag, 'model': model_name,
        'thinking': enable_thinking, 'batch': BATCH,
        'max_new': MAX_NEW, 'benchmarks': {},
    }
    raw_records = []

    # ── 1. VRS-Caption ──
    if not SKIP_CAP:
        print(f"\n  VRS-Caption...")
        t0 = time.time()
        cap_path = SAMPLE_DIR / 'vrs_cap_sampled.json'
        if cap_path.exists():
            with open(cap_path) as f:
                anns = json.load(f)
            refs, hyps = {}, {}
            token_lens = []
            for i, ann in enumerate(anns):
                img = safe_img(str(VRS_IMG_DIR / ann.get('image_id', '')),
                               fallback_dir=str(TEST_IMGS[0].parent) if TEST_IMGS else None)
                gt = ann.get('ground_truth', '').strip()
                if not gt:
                    continue
                clean, raw, nt = _run(img, CAPTION_PROMPT)
                token_lens.append(nt)
                refs[str(i)] = [gt]
                hyps[str(i)] = [clean]
                raw_records.append({
                    'benchmark': 'VRS-Caption',
                    '_idx': ann.get('_original_index', i),
                    'image_id': ann.get('image_id', ''),
                    'gt': gt, 'pred': clean, 'tokens': nt,
                })
            scores = compute_caption_scores(refs, hyps)
            stats = {'min': min(token_lens), 'max': max(token_lens),
                     'avg': round(sum(token_lens)/len(token_lens), 1)} if token_lens else {}
            row['benchmarks']['vrs_caption'] = {**scores, 'token_stats': stats}
            print(f"    done ({time.time()-t0:.0f}s) bleu4={scores['bleu4']}")

    # ── 2. VRS-VQA ──
    print(f"\n  VRS-VQA...")
    t0 = time.time()
    vqa_path = SAMPLE_DIR / 'vrs_vqa_sampled.json'
    if vqa_path.exists():
        with open(vqa_path) as f:
            anns = json.load(f)
        correct, total = 0, 0
        vqa_records = []
        for i, ann in enumerate(anns):
            img = safe_img(str(VRS_IMG_DIR / ann.get('image_id', '')),
                           fallback_dir=str(TEST_IMGS[0].parent) if TEST_IMGS else None)
            q = ann.get('question', '') or ann.get('text', '')
            gt = str(ann.get('ground_truth', '') or ann.get('answer', '')).strip()
            if not gt:
                continue
            prompt = VRS_VQA_PROMPT_TPL.format(question=q)
            clean, raw, nt = _run(img, prompt, max_new=64)
            ok = 1 if gt.lower() in clean.lower() else 0
            correct += ok
            total += 1
            rec = {
                'benchmark': 'VRS-VQA', '_idx': ann.get('_original_index', i),
                'image_id': ann.get('image_id', ''),
                'gt': gt, 'pred': clean, 'tokens': nt,
                'question': q, 'correct': ok,
            }
            vqa_records.append(rec)
            raw_records.append(rec)
        row['benchmarks']['vrs_vqa'] = {
            'acc_l1': round(correct / max(total, 1), 4),
            'total': total, 'correct_l1': correct,
        }
        # 保存 VQA 数据用于后续三级判定
        raw_dir = RAW_DIR_ROOT / f'batch{BATCH}' / config_tag
        raw_dir.mkdir(parents=True, exist_ok=True)
        with open(raw_dir / 'vqa_raw.json', 'w') as f:
            json.dump(vqa_records, f, indent=2)
        print(f"    done ({time.time()-t0:.0f}s) acc_l1={correct}/{total}")

    # ── 3. MME-RS ──
    print(f"\n  MME-RS...")
    t0 = time.time()
    mme_path = SAMPLE_DIR / 'mme_rs_sampled.json'
    if mme_path.exists():
        with open(mme_path) as f:
            anns = json.load(f)
        correct, total = 0, 0
        token_lens = []
        for i, ann in enumerate(anns):
            img_path_str = ann.get('Image', '')
            if MME_RS_DIR.exists() and img_path_str:
                fp = MME_RS_DIR / Path(img_path_str).name
                if fp.exists():
                    img = safe_img(str(fp))
                else:
                    img = get_test_img(i)
            else:
                img = get_test_img(i)
            q = ann.get('Text', '') or ann.get('question', '')
            choices = ann.get('Answer choices', [])
            prompt = MME_PROMPT_TPL.format(question=q, choices='\n'.join(choices))
            gt = str(ann.get('Ground truth', '') or ann.get('answer', '')).strip().upper()
            clean, raw, nt = _run(img, prompt, max_new=10)
            token_lens.append(nt)
            pred = extract_letter(clean)
            ok = 1 if pred == gt else 0
            correct += ok
            total += 1
            raw_records.append({
                'benchmark': 'MME-RS',
                '_idx': ann.get('Question_id', ann.get('_original_index', i)),
                'image_id': img_path_str,
                'gt': gt, 'pred': pred, 'tokens': nt, 'question': q,
            })
        row['benchmarks']['mme_rs'] = {
            'acc': round(correct / max(total, 1), 4),
            'correct': correct, 'total': total,
            'token_stats': {'min': min(token_lens), 'max': max(token_lens),
                            'avg': round(sum(token_lens)/len(token_lens), 1)} if token_lens else {},
        }
        print(f"    done ({time.time()-t0:.0f}s) acc={correct}/{total}")

    # ── 4. LEVIR-CC ──
    if not SKIP_CAP:
        print(f"\n  LEVIR-CC...")
        t0 = time.time()
        levir_path = SAMPLE_DIR / 'levir_cc_sampled.json'
        if levir_path.exists():
            with open(levir_path) as f:
                data = json.load(f)
            anns = [a for a in data.get('images', data)]
            refs, hyps = {}, {}
            token_lens = []
            for i, ann in enumerate(anns):
                fname = ann['filename']
                before = safe_img(str(LEVIR_DIR / 'A' / fname),
                                  fallback_dir=str(TEST_IMGS[0].parent) if TEST_IMGS else None)
                after = safe_img(str(LEVIR_DIR / 'B' / fname),
                                 fallback_dir=str(TEST_IMGS[0].parent) if TEST_IMGS else None)
                refs_list = [s['raw'].strip() for s in ann.get('sentences', [])]
                clean, raw, nt = _run(before, LEVIR_PROMPT, extra_imgs=[after])
                token_lens.append(nt)
                refs[str(i)] = refs_list
                hyps[str(i)] = [clean]
                raw_records.append({
                    'benchmark': 'LEVIR-CC',
                    '_idx': ann.get('_original_index', ann.get('imgid', i)),
                    'image_id': fname,
                    'gt': ' || '.join(refs_list),
                    'pred': clean, 'tokens': nt,
                })
            scores = compute_caption_scores(refs, hyps)
            stats = {'min': min(token_lens), 'max': max(token_lens),
                     'avg': round(sum(token_lens)/len(token_lens), 1)} if token_lens else {}
            row['benchmarks']['levir_cc'] = {**scores, 'token_stats': stats}
            print(f"    done ({time.time()-t0:.0f}s) bleu4={scores['bleu4']}")

    # ── 5. XLRS ──
    print(f"\n  XLRS...")
    t0 = time.time()
    xlrs_path = SAMPLE_DIR / 'xlrs_full.json'
    if xlrs_path.exists():
        with open(xlrs_path) as f:
            anns = json.load(f)
        correct, total = 0, 0
        token_lens = []
        for i, ann in enumerate(anns):
            local_img = ann.get('local_image', '')
            if local_img and Path(local_img).exists():
                img = Image.open(local_img).convert('RGB')
            else:
                img = get_test_img(i)
            q = ann.get('question', '')
            choices = ann.get('multi-choice options', [])
            prompt = XLRS_PROMPT_TPL.format(question=q, choices='\n'.join(choices))
            gt = str(ann.get('answer', '')).strip().upper()
            gt_set = set(gt.replace(',', ' ').split()) if gt else set()
            clean, raw, nt = _run(img, prompt, max_new=10)
            token_lens.append(nt)
            letter = extract_letter(clean, choices='ABCD')
            pred_set = {letter} if letter else set()
            ok = 1 if gt_set and pred_set == gt_set else 0
            correct += ok
            total += 1
            raw_records.append({
                'benchmark': 'XLRS',
                '_idx': ann.get('_original_index', ann.get('idx', i)),
                'image_id': ann.get('path', ''),
                'gt': gt, 'pred': list(pred_set), 'tokens': nt,
                'question': q[:60],
            })
        row['benchmarks']['xlrs'] = {
            'acc': round(correct / max(total, 1), 4),
            'correct': correct, 'total': total,
            'token_stats': {'min': min(token_lens), 'max': max(token_lens),
                            'avg': round(sum(token_lens)/len(token_lens), 1)} if token_lens else {},
        }
        print(f"    done ({time.time()-t0:.0f}s) acc={correct}/{total}")

    # ── 保存结果 ──
    res_file = RES_DIR / f'{config_tag}_metrics.json'
    RES_DIR.mkdir(parents=True, exist_ok=True)
    with open(res_file, 'w') as f:
        json.dump(row, f, indent=2)
    print(f"  Saved metrics: {res_file}")

    raw_dir = RAW_DIR_ROOT / f'batch{BATCH}' / config_tag
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / 'raw_outputs.json'
    with open(raw_file, 'w') as f:
        json.dump(raw_records, f, indent=2)
    print(f"  Saved raw: {raw_file} ({len(raw_records)} records)")

    # 清理 GPU
    del model, processor
    torch.cuda.empty_cache()
    print(f"  [{config_tag}] DONE")
    return row


# ─── 主入口 ──────────────────────────────────────────────────
if __name__ == '__main__':
    os.environ.update({'TF_CPP_MIN_LOG_LEVEL': '3', 'CUDA_VISIBLE_DEVICES': '0'})
    models_to_run = []

    if args.all:
        models_to_run = [(m, False) for m in ALL_MODELS]  # thinkOFF
        models_to_run += [(m, True) for m in DUAL_MODELS]  # thinkON
    elif args.model:
        models_to_run = [(args.model, args.thinking)]
    else:
        parser.print_help()
        sys.exit(1)

    for model_name, enable_thinking in models_to_run:
        run_eval(model_name, enable_thinking)

```

## vqa\_judge\.py — VQA 三级判定器

```Plain Text
"""
VRSBench VQA 三级判定器 — 本地 LLM 语义等价性判断。

Level 1: GT 是否在预测字符串中 (substring match)
Level 2: 对 yes/no/数字 做精确匹配
Level 3: LLM judge 判断语义等价（当 L1/L2 不适用或不匹配时）
"""
import json, re, time, sys
from pathlib import Path

from config import RES_DIR, RAW_DIR_ROOT


def level1_substring(gt: str, pred: str) -> bool:
    """Level 1: ground_truth in predicted (不区分大小写)。"""
    return gt.lower() in pred.lower()


def level2_exact(gt: str, pred: str) -> bool | None:
    """
    Level 2: yes/no/数字 精确匹配。
    Returns: True/False (匹配/不匹配), None (不适用)
    """
    yes_no = ['yes', 'no']
    numbers = [str(i) for i in range(100)]
    try:
        import inflect
        engine = inflect.engine()
        for num in range(100):
            word = engine.number_to_words(num)
            numbers.append(word)
            numbers.append(word.replace('-', ' '))
    except ImportError:
        pass
    if gt.lower() in yes_no + numbers:
        return gt.lower() == pred.lower()
    return None  # 不适用


def load_judge_llm(gguf_path: str):
    """加载 Qwen3.6-35B-A3B Q4_K_M GGUF 判分模型。"""
    from llama_cpp import Llama
    print(f"Loading judge LLM: {gguf_path}...", flush=True)
    t0 = time.time()
    llm = Llama(
        model_path=gguf_path,
        n_gpu_layers=-1,   # 全量 GPU
        n_ctx=2048,
        verbose=False,
    )
    print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)
    return llm


def level3_judge(llm, question: str, gt: str, pred: str) -> int:
    """
    Level 3: LLM 判断语义等价。
    Returns: 1 (match), 0 (not match)
    """
    prompt = (
        f"Question: {question}\n"
        f"Ground Truth Answer: {gt}\n"
        f"Predicted Answer: {pred}\n"
        f"Does the predicted answer match the ground truth? "
        f"Answer 1 for match and 0 for not match. "
        f"Use semantic meaning not exact match. "
        f"Synonyms are also treated as a match, e.g., "
        f"football and soccer, playground and ground track field, "
        f"building and rooftop, pond and swimming pool. "
        f"Shorten the predicted answer from a full sentence to remove "
        f"unnecessary words before comparison. "
        f"Do not explain the reason.\n"
    )
    out = llm(prompt, max_tokens=2, temperature=0.0, echo=False)
    answer = out['choices'][0]['text'].strip()
    return 1 if '1' in answer else 0


def run_vqa_judge(
    model_name: str,
    mode: str = 'thinkOFF',
    batch: int = 1,
    gguf_path: str | None = None,
):
    """
    对指定模型的 VRS-VQA 预测运行三级判定。

    Args:
        model_name: 模型名称
        mode: 'thinkOFF' 或 'thinkON'
        batch: batch 编号

    Returns:
        三级判定的准确率统计
    """
    config_tag = f"{model_name}_{'thinkON' if mode == 'thinkON' else 'thinkOFF'}_batch{batch}"
    vqa_file = RAW_DIR_ROOT / f'batch{batch}' / config_tag / 'vqa_raw.json'

    if not vqa_file.exists():
        print(f"[WARN] VQA predictions not found: {vqa_file}")
        return None

    with open(vqa_file) as f:
        records = json.load(f)

    # 确定数据类型，按 type 分组
    from collections import defaultdict
    by_type = defaultdict(list)
    for rec in records:
        qtype = rec.get('type', 'unknown').lower()
        by_type[qtype].append(rec)

    # L1 判定（substring）
    l1_results = {}
    for rec in records:
        qid = rec.get('question_id', '')
        gt = rec.get('ground_truth', '') or rec.get('gt', '')
        pred = rec.get('predicted', '') or rec.get('pred', '')
        l1_results[qid] = 1 if level1_substring(gt, pred) else 0

    # L2 判定（yes/no/数字）
    l2_results = {}
    for rec in records:
        qid = rec.get('question_id', '')
        gt = rec.get('ground_truth', '') or rec.get('gt', '')
        pred = rec.get('predicted', '') or rec.get('pred', '')
        l2 = level2_exact(gt, pred)
        if l2 is not None:
            l2_results[qid] = 1 if l2 else 0

    # L3 判定（LLM judge）
    l3_results = {}
    if gguf_path and len(records) > len(l2_results):
        llm = load_judge_llm(gguf_path) if isinstance(gguf_path, str) else gguf_path
        for rec in records:
            qid = rec.get('question_id', '')
            if qid in l2_results:
                continue  # L2 已覆盖的跳过
            gt = rec.get('ground_truth', '') or rec.get('gt', '')
            pred = rec.get('predicted', '') or rec.get('pred', '')
            q = rec.get('question', '')
            l3_results[qid] = level3_judge(llm, q, gt, pred)

    # 合并结果（L2 > L1 > L3 优先级）
    total, correct = 0, 0
    type_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    for rec in records:
        qid = rec.get('question_id', '')
        qtype = rec.get('type', 'unknown').lower()
        # 优先 L2 结果，其次 L1，最后 L3
        if qid in l2_results:
            ok = l2_results[qid]
        elif qid in l1_results:
            ok = l1_results[qid]
        elif qid in l3_results:
            ok = l3_results[qid]
        else:
            ok = 0
        correct += ok
        total += 1
        type_stats[qtype]['correct'] += ok
        type_stats[qtype]['total'] += 1

    # 输出
    acc = round(correct / max(total, 1), 4)
    print(f"\n=== {config_tag} VQA 三级判定结果 ===")
    print(f"Overall: {correct}/{total} = {acc:.2%}")
    print(f"  L1 covered: {len(l1_results)}, L2 covered: {len(l2_results)}, L3 covered: {len(l3_results)}")
    for qtype, st in sorted(type_stats.items()):
        print(f"  {qtype}: {st['correct']}/{st['total']} = {st['correct']/max(st['total'],1):.2%}")

    return {
        'acc_l1': round(sum(l1_results.values()) / max(len(l1_results), 1), 4),
        'acc_l3': round(correct / max(total, 1), 4),
        'correct': correct,
        'total': total,
        'by_type': dict(type_stats),
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--mode', default='thinkOFF')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--gguf', default='/home/admin1/models/qwen3.6-35b-q4_k_m.gguf')
    args = parser.parse_args()
    run_vqa_judge(args.model, args.mode, args.batch, args.gguf)

```

## gen\_excel\.py — Excel 报告生成

```Plain Text
"""
生成 Excel 报告 — 从 results/*_metrics.json 读取分数，汇总为表格。

用法:
  python3 gen_excel.py                          # batch1
  python3 gen_excel.py --batch 2

输出:
  projects/batch{1,2}_thinkOFF.xlsx
  projects/batch{1,2}_thinkON.xlsx
"""
import json, os, argparse
from pathlib import Path
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from config import (
    ALL_MODELS, DUAL_MODELS, METRICS, MLABELS, MINFO,
    RES_DIR, RAW_DIR_ROOT, OUT_DIR,
)

BATCH = 1  # 默认

# ─── 样式 ──────────────────────────────────────────────────
HDR_FONT = Font(bold=True, color='FFFFFF')
HDR_FILL = PatternFill('solid', fgColor='1F4E79')
BEST_FILL = PatternFill('solid', fgColor='C6EFCE')
SOTA_FILL = PatternFill('solid', fgColor='FFF2CC')
BORDER = Border(
    left=Side('thin'), right=Side('thin'),
    top=Side('thin'), bottom=Side('thin'),
)


def _fmt(v):
    if v is None:
        return ''
    return round(v, 4) if isinstance(v, float) else v


def gs(benchmarks, key, sub='acc'):
    """从 benchmarks 字典中安全取值。"""
    mapping = {
        'mme_rs': ('mme_rs', 'acc'),
        'xlrs': ('xlrs', 'acc'),
        'vrs_cap_bleu4': ('vrs_caption', 'bleu4'),
        'vrs_cap_rouge_l': ('vrs_caption', 'rouge_l'),
        'vrs_cap_cider': ('vrs_caption', 'cider'),
        'levir_bleu4': ('levir_cc', 'bleu4'),
        'levir_rouge_l': ('levir_cc', 'rouge_l'),
        'levir_cider': ('levir_cc', 'cider'),
        'vrs_vqa': ('vrs_vqa', 'acc_l1'),
    }
    bk, sk = mapping.get(key, (key, sub))
    v = benchmarks.get(bk, {}).get(sk)
    return v


def load_metrics(tag):
    fp = RES_DIR / f'{tag}_metrics.json'
    if fp.exists():
        return json.load(open(fp))
    return None


def collect_models(models, mode_tag):
    """收集模型指标。"""
    rows = []
    for m in models:
        d = load_metrics(f'{m}_{mode_tag}_batch{BATCH}')
        if d and d.get('benchmarks'):
            scores = {x: gs(d['benchmarks'], x) for x in METRICS}
            rows.append((m, mode_tag.replace('_', ''), scores))
        else:
            rows.append((m, f'{mode_tag.replace("_", "")}(TODO)', {}))
    return rows


def add_sota_rows(ws, start_row):
    """在指标表底部添加 SOTA 对比行。"""
    sota_data = [
        ('LLaVA-1.5 (7B)', 'SOTA', None, 22.8, 0.0147, None, None, None, None, 76.4),
        ('Mini-Gemini (7B)', 'SOTA', None, None, 0.0143, None, None, None, None, 77.8),
        ('GeoChat (7B)', 'SOTA', None, 22.0, 0.0138, None, None, None, None, 76.0),
        ('Qwen2-VL (7B)', 'SOTA', 44.8, 39.0, None, None, None, None, None, None),
        ('CogVLM2 (8B)', 'SOTA', None, 39.8, None, None, None, None, None, None),
        ('GPT-4o', 'SOTA', '~38', 32.2, 0.0086, None, 0.1805, None, None, 65.6),
        ('Qwen-VL-Plus', 'SOTA', None, None, None, None, 0.2295, None, None, None),
    ]
    for ri, row in enumerate(sota_data, start_row):
        for ci, v in enumerate(row):
            cell = ws.cell(ri, ci + 1, v if v is not None else '')
            cell.font = Font(italic=True, color='666666')
            cell.fill = SOTA_FILL
            cell.border = BORDER


def make_excel(fname, met_rows, raw_models):
    """生成 Excel 报告。"""
    wb = openpyxl.Workbook()

    # ── Sheet 1: 指标表 ──
    ws = wb.active
    ws.title = '指标表'
    headers = ['模型', '模式'] + MLABELS
    for ci, h in enumerate(headers, 1):
        cell = ws.cell(1, ci, h)
        cell.fill = HDR_FILL
        cell.font = HDR_FONT
        cell.border = BORDER

    for ri, (model, mode, scores) in enumerate(met_rows, 2):
        ws.cell(ri, 1, model).border = BORDER
        ws.cell(ri, 2, mode).border = BORDER
        for ci, m in enumerate(METRICS, 3):
            cell = ws.cell(ri, ci, _fmt(scores.get(m)))
            cell.border = BORDER

    # SOTA 对比行
    sep_row = len(met_rows) + 2
    ws.cell(sep_row, 1, '═══ SOTA 参考 ═══').font = Font(bold=True, color='999999')
    add_sota_rows(ws, sep_row + 1)

    for ci in range(1, ws.max_column + 1):
        ws.column_dimensions[get_column_letter(ci)].width = 18

    # ── Sheet 2: 原始输出表 ──
    ws2 = wb.create_sheet('原始输出表')
    raw_headers = ['模型', '模式', 'Benchmark', 'idx', 'GT', 'Prediction', 'Token数']
    for ci, h in enumerate(raw_headers, 1):
        cell = ws2.cell(1, ci, h)
        cell.fill = HDR_FILL
        cell.font = HDR_FONT
        cell.border = BORDER

    r = 2
    for model, mode in raw_models:
        tag_mode = mode.replace('(TODO)', '').lower()
        tag = f'{model}_{"thinkON" if "thinkON" in tag_mode else "thinkOFF"}_batch{BATCH}'
        rp = RAW_DIR_ROOT / f'batch{BATCH}' / tag / 'raw_outputs.json'
        if rp.exists():
            for rec in json.load(open(rp)):
                vals = [
                    model, mode,
                    rec.get('benchmark', ''),
                    rec.get('_idx', ''),
                    str(rec.get('gt', ''))[:120],
                    str(rec.get('pred', ''))[:120],
                    rec.get('tokens', ''),
                ]
                for ci, v in enumerate(vals, 1):
                    cell = ws2.cell(r, ci, v)
                    cell.border = BORDER
                r += 1

    # ── Sheet 3: 分析总结 ──
    ws3 = wb.create_sheet('分析总结')
    sum_headers = ['指标', '说明', '最佳模型/模式', '最佳得分']
    for ci, h in enumerate(sum_headers, 1):
        cell = ws3.cell(1, ci, h)
        cell.fill = HDR_FILL
        cell.font = HDR_FONT
        cell.border = BORDER

    for ri, m in enumerate(METRICS, 2):
        desc = MINFO.get(m, m)
        best_val, best_mod = None, ''
        for model, mode, scores in met_rows:
            val = scores.get(m)
            if val is not None and (best_val is None or val > best_val):
                best_val = val
                best_mod = f'{model} ({mode})'
        ws3.cell(ri, 1, desc).border = BORDER
        ws3.cell(ri, 2, m).border = BORDER
        cm = ws3.cell(ri, 3, best_mod)
        cm.border = BORDER
        cm.fill = BEST_FILL
        ws3.cell(ri, 4, _fmt(best_val)).border = BORDER

    wb.save(OUT_DIR / fname)
    print(f"  Saved: {OUT_DIR / fname}")


# ─── 主入口 ──────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1)
    args = parser.parse_args()
    BATCH = args.batch

    # 收集指标
    print(f"Batch {BATCH}:")
    print("  thinkOFF...")
    met_off = collect_models(ALL_MODELS, 'thinkOFF')
    print("  thinkON...")
    met_on = collect_models(DUAL_MODELS, 'thinkON')

    # 原始输出用的模型模式列表（所有有数据的配置）
    raw_off = [(m, 'thinkOFF') for m, ms, _ in met_off if ms != 'thinkOFF(TODO)']
    raw_on = [(m, 'thinkON') for m, ms, _ in met_on if ms != 'thinkON(TODO)']

    make_excel(f'batch{BATCH}_thinkOFF.xlsx', met_off, raw_off)
    make_excel(f'batch{BATCH}_thinkON.xlsx', met_on, raw_on)
    print("Done!")

```

# 五、输出指标说明

MME\-RS — MME\-RealWorld RS 准确率 \(0\-1\)：多选 VQA

XLRS — XLRS\-Bench 准确率 \(0\-1\)：超高分辨率多选

VRS\-Cap B4 — VRSBench Caption BLEU\-4 \(0\-1\)

VRS\-Cap R\-L — VRSBench Caption ROUGE\-L \(0\-1\)

VRS\-Cap CIDER — VRSBench Caption CIDER \(0\-∞\)

LEVIR B4 — LEVIR\-CC BLEU\-4 \(0\-1\)：变化描述 n\-gram

LEVIR R\-L — LEVIR\-CC ROUGE\-L \(0\-1\)

LEVIR CIDER — LEVIR\-CC CIDER \(0\-∞\)

VRS\-VQA — VRSBench VQA L1 准确率 \(0\-1\)



⚠️ 注意：Caption 指标（BLEU/CIDER）对输出风格极其敏感。模型输出长段落时即使语义正确，BLEU\-4 和 CIDER 也可能接近 0。

# 六、模型列表

所有模型通过 AutoModelForImageTextToText 加载。

- minicpm\-v\-4\.6 — 4\.6B — thinkON ✅

- qwen3\.5\-0\.8B — 0\.8B — thinkON ✅

- qwen3\.5\-2B — 2B — thinkON ✅

- qwen3\.5\-4B — 4B — thinkON ✅

- qwen3\-vl\-2B — 2B — thinkON ❌

- qwen3\-vl\-4B — 4B — thinkON ❌

- gemma\-4\-e2b — 2B — thinkON ❌

- gemma\-4\-e4b — 4B — thinkON ❌

