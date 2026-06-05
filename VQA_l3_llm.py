#!/usr/bin/env python3
"""
VQA L3 评测 — 本地大模型 (Qwen3.5-4B 等)
========================================

设计:
  - 输入: vqa_l3_raw.json (由 apply_model_adapter.py 重生成)
  - LLM: 本地部署的 DUAL 模型（默认 qwen3.5-4B），走 inference.load_model
  - 评测流程: 严格按 VRSBench 官方源码 (eval_vqa_gpt.ipynb)
    4 步短路:
      1) gt.lower() ⊆ pred.lower()              → '1'
      2) gt ∈ {'yes','no','0'..'99'}            → 严格相等判 1/0
      3) 记录有 'correct' ∈ {'0','1'} 缓存值     → 直接用
      4) 兜底: 调本地 LLM（温度 = 0）
  - Prompt: 严格按 VRS 官方 (eval_utils.py / eval_vqa_gpt.ipynb cell 1)
    原句:
      Question: {question}
      Ground Truth Answer: {ground_truth}
      Predicted Answer: {predicted}
      Does the predicted answer match the ground truth? Answer 1 for match
      and 0 for not match. Use semantic meaning not exact match. Synonyms
      are also treated as a match, e.g., football and soccer, playground
      and ground track field, building and rooftop, pond and swimming pool.
      Do not explain the reason.
  - 温度: do_sample=False（温度 0）
  - 输出:
      1) vqa_l3_llm.json          - 逐条记录 + 响应 + 判定
      2) vqa_l3_llm_metrics.json  - {summary: {total_l3, correct_l3, acc_l3, short_circuits}}

用法:
  python VQA_l3_llm.py --batch-dir /home/admin1/models/raw_outputs/lever_test2
  python VQA_l3_llm.py <single_vqa_l3_raw.json>
  python VQA_l3_llm.py --model qwen3.5-4B --batch-dir /home/admin1/models/raw_outputs/lever_test2
  python VQA_l3_llm.py --model qwen3-vl-4B --batch-dir /home/admin1/models/raw_outputs/lever_test2
  python VQA_l3_llm.py --model gemma-4-e4b --batch-dir /home/admin1/models/raw_outputs/lever_test2
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

# ─── 默认配置 ─────────────────────────────────────────────
DEFAULT_MODEL = 'qwen3.5-4B'           # 走 inference.load_model 的本地 DUAL 模型
MODEL_ROOT = '/home/admin1/models'     # 跟 config.ROOT 同
DEFAULT_MAX_NEW = 8                    # L3 判定只需 1 token，预留余量
DEFAULT_BATCH_SIZE = 16                # batch 推理：实测 16 比 8 再快 2x，20 题 26ms/条
TEMPERATURE = 0.0                      # 张哥要求：温度强制为 0
DO_SAMPLE = False                      # 温度 0 对应 greedy（do_sample=False）


# ─── VRS 官方 L3 评测 prompt（严格按 eval_vqa_gpt.ipynb cell 1） ──
# 原文 (GPT-4o-mini):
#   Question: {question}
#   Ground Truth Answer: {ground_truth}
#   Predicted Answer: {predicted}
#   Does the predicted answer match the ground truth?
#   Answer 1 for match and 0 for not match.
#   Use semantic meaning not exact match. Synonyms are also treated as a match,
#   e.g., football and soccer, playground and ground track field,
#   building and rooftop, pond and swimming pool. Do not explain the reason.
VRS_L3_PROMPT_TPL = (
    "Question: {question}\n"
    "Ground Truth Answer: {ground_truth}\n"
    "Predicted Answer: {predicted}\n"
    "Does the predicted answer match the ground truth? "
    "Answer 1 for match and 0 for not match. "
    "Use semantic meaning not exact match. "
    "Synonyms are also treated as a match, "
    "e.g., football and soccer, playground and ground track field, "
    "building and rooftop, pond and swimming pool. "
    "Do not explain the reason."
)


# ─── VRS 源码 4 步短路常量 ─────────────────────────────────
# 短路 2: GT ∈ {'yes','no','0','1',...,'99'}
SHORT_GT_SET = {'yes', 'no'} | {str(i) for i in range(100)}


# ─── 解析 LLM 响应（提取 0/1） ─────────────────────────────
def parse_judgment(response_text: str) -> int:
    """
    从本地 LLM 响应中提取 0/1。优先级：
      P1: 响应开头首个 '0'/'1' 字符
      P2: 全文首个独立 '0'/'1' 单词
      P3: 否定词（not match / no match / incorrect / wrong / doesn't match / does not match）
      P4: 肯定词（match / yes / true / correct）
      P5: 单独 'no'/'false' → 0；'yes'/'true' → 1
      兜底: 0
    """
    if not response_text:
        return 0
    s = response_text.strip()

    # P1: 首个字符
    if s and s[0] in ('0', '1'):
        return int(s[0])

    # P2: 首个独立 0/1
    m = re.search(r'\b([01])\b', s)
    if m:
        return int(m.group(1))

    lower = s.lower()

    # P3: 否定词（覆盖各种写法，优先级 > 肯定词）
    if ('not match' in lower or 'not_match' in lower
            or "doesn't match" in lower or 'doesnt match' in lower
            or 'does not match' in lower or 'do not match' in lower
            or 'no match' in lower or 'not a match' in lower
            or 'not correct' in lower or 'not true' in lower
            or 'not a correct' in lower
            or 'incorrect' in lower or 'wrong' in lower):
        return 0

    # P4: 肯定词
    if 'match' in lower or 'yes' in lower or 'true' in lower or 'correct' in lower:
        return 1

    # P5: 兜底
    if re.search(r'\b(no|false)\b', lower):
        return 0
    if re.search(r'\b(yes|true)\b', lower):
        return 1

    return 0


# ─── 本地 LLM 调用（纯文本，无图） ──────────────────────────
def _apply_template_text(processor, prompt: str) -> str:
    """纯文本模板（无图）"""
    conv = [{'role': 'user',
             'content': [{'type': 'text', 'text': prompt}]}]
    try:
        return processor.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=False,
            enable_thinking=False,  # thinkOFF 模式
        )
    except TypeError:
        return processor.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=False,
        )


def call_llm_text(model, processor, prompt: str, max_new_tokens: int) -> dict:
    """
    纯文本对话（无图），跟 VRS 官方 4o-mini 调用方式一致（不带图）。
    温度强制 = 0（do_sample=False）。

    返回 {'ok': bool, 'response_text': str, 'error': str}
    """
    import torch

    try:
        text = _apply_template_text(processor, prompt)
    except Exception as e:
        return {'ok': False, 'response_text': '', 'error': f'apply_chat_template failed: {e}'}

    try:
        inputs = processor(text=[text], return_tensors='pt',
                           add_special_tokens=False).to('cuda:0')
    except Exception as e:
        return {'ok': False, 'response_text': '', 'error': f'processor(text) failed: {e}'}

    try:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=DO_SAMPLE,        # 温度 0
                temperature=TEMPERATURE,    # 显式再写一次，防止 do_sample 配错
                return_dict_in_generate=True,
            )
    except TypeError:
        # 某些模型 generate 不接受 temperature 单独传
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=DO_SAMPLE,
                return_dict_in_generate=True,
            )

    gen = out.sequences[0, inputs['input_ids'].shape[1]:]
    response_text = processor.decode(gen, skip_special_tokens=True)
    return {'ok': True, 'response_text': response_text.strip(), 'error': ''}


def call_llm_text_batch(model, processor, prompts: list[str], max_new_tokens: int) -> list[dict]:
    """
    批量 LLM 调用（无图，左 padding），温度 = 0。
    实测 batch=16 时 29ms/条，比单条 537ms 快 18x。
    16139 条 L3 题预计 8 分钟跑完。

    返回 [{'ok': bool, 'response_text': str, 'error': str}, ...]（按 prompts 顺序）
    """
    import torch

    if not prompts:
        return []

    n = len(prompts)
    results = [{'ok': False, 'response_text': '', 'error': 'init'}] * n

    # 1. 构造所有模板文本
    texts = []
    for i, p in enumerate(prompts):
        try:
            texts.append(_apply_template_text(processor, p))
        except Exception as e:
            results[i] = {'ok': False, 'response_text': '', 'error': f'template failed: {e}'}
            texts.append(None)

    # 2. 过滤掉模板失败的，处理顺序
    valid_indices = [i for i, t in enumerate(texts) if t is not None]
    valid_texts = [texts[i] for i in valid_indices]

    if not valid_texts:
        return results

    # 3. 左 padding（生成时必须，否则 pad token 干扰首 token）
    original_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = 'left'
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    try:
        inputs = processor(
            text=valid_texts,
            return_tensors='pt',
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=512,        # L3 prompt 都很短
        ).to('cuda:0')

        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=DO_SAMPLE,
                    temperature=TEMPERATURE,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                )
        except TypeError:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=DO_SAMPLE,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                )

        # 4. 解析每个生成
        prompt_len = inputs['input_ids'].shape[1]
        gen_tokens = out.sequences[:, prompt_len:]
        for j, idx in enumerate(valid_indices):
            try:
                resp_text = processor.decode(gen_tokens[j], skip_special_tokens=True)
                results[idx] = {'ok': True, 'response_text': resp_text.strip(), 'error': ''}
            except Exception as e:
                results[idx] = {'ok': False, 'response_text': '', 'error': f'decode failed: {e}'}
    except Exception as e:
        # batch 整体失败，所有 valid 都标记
        for idx in valid_indices:
            results[idx] = {'ok': False, 'response_text': '', 'error': f'batch generate failed: {e}'}
    finally:
        processor.tokenizer.padding_side = original_padding_side

    return results


# ─── VRS 4 步短路（严格按 eval_vqa_gpt.ipynb 源码） ─────────
def vrs_4step_judge(question: str, gt: str, pred: str,
                    record: dict,
                    model, processor, max_new_tokens: int) -> dict:
    """
    返回 {'correct_l3': int(0/1), 'response_text': str, 'short_circuit': str, 'ok': bool, 'error': str}

    短路记录 'short_circuit':
      - 'l1_substr'      (短路 1: GT ⊆ pred)
      - 'l2_short_gt'    (短路 2: GT ∈ yes/no/数字，严格相等判)
      - 'cached_correct' (短路 3: 已缓存 correct 字段)
      - 'llm_judge'      (走 LLM)
    """
    gt_lower = gt.lower()
    pred_lower = pred.lower()

    # 短路 1: GT 是 pred 的子串（大小写不敏感）
    if gt_lower in pred_lower:
        return {
            'correct_l3': 1, 'response_text': '',
            'short_circuit': 'l1_substr', 'ok': True, 'error': '',
        }

    # 短路 2: GT ∈ {yes, no, 0..99} → 严格相等
    if gt_lower in SHORT_GT_SET:
        is_c = 1 if gt_lower == pred_lower else 0
        return {
            'correct_l3': is_c, 'response_text': '',
            'short_circuit': 'l2_short_gt', 'ok': True, 'error': '',
        }

    # 短路 3: 记录已有 correct 字段（apply_model_adapter 之后通常没有，留兼容）
    cached = record.get('correct')
    if cached in ('0', '1'):
        return {
            'correct_l3': int(cached), 'response_text': '',
            'short_circuit': 'cached_correct', 'ok': True, 'error': '',
        }

    # 兜底 4: 调本地 LLM（温度 0）
    prompt = VRS_L3_PROMPT_TPL.format(
        question=question, ground_truth=gt, predicted=pred)
    resp = call_llm_text(model, processor, prompt, max_new_tokens)
    if not resp['ok']:
        return {
            'correct_l3': 0, 'response_text': '',
            'short_circuit': 'llm_judge', 'ok': False, 'error': resp['error'],
        }
    is_c = parse_judgment(resp['response_text'])
    return {
        'correct_l3': is_c,
        'response_text': resp['response_text'][:200],
        'short_circuit': 'llm_judge', 'ok': True, 'error': '',
    }


# ─── 批量：单文件 ──────────────────────────────────────────
def judge_file(l3_path: Path, model, processor, max_new_tokens: int,
               batch_size: int = DEFAULT_BATCH_SIZE) -> dict:
    if not l3_path.exists():
        raise FileNotFoundError(f'文件不存在: {l3_path}')

    recs = json.load(open(l3_path, encoding='utf-8'))
    if not isinstance(recs, list):
        raise ValueError(f'{l3_path} 顶层不是 list')

    out_json = l3_path.parent / 'vqa_l3_llm.json'
    metrics_path = l3_path.parent / 'vqa_l3_llm_metrics.json'

    n_total = len(recs)
    n_correct = 0
    sc_counts = {'l1_substr': 0, 'l2_short_gt': 0, 'cached_correct': 0, 'llm_judge': 0}
    sc_correct = {'l1_substr': 0, 'l2_short_gt': 0, 'cached_correct': 0, 'llm_judge': 0}
    n_ok_llm = 0
    t0 = time.time()

    # 第 1 遍扫：先做 4 步短路的前 3 步，收集需要走 LLM 的题
    pre_results = []  # [(idx, sc, is_c, response, ok, error), ...]
    llm_prompts = []  # [prompt, ...]
    llm_indices = []  # [rec_idx, ...]
    for i, r in enumerate(recs):
        question = r.get('question', '')
        gt = r.get('gt', '')
        pred = r.get('pred', '')
        gt_lower = gt.lower()
        pred_lower = pred.lower()

        # 短路 1
        if gt_lower in pred_lower:
            pre_results.append((i, 'l1_substr', 1, '', True, ''))
            continue
        # 短路 2
        if gt_lower in SHORT_GT_SET:
            is_c = 1 if gt_lower == pred_lower else 0
            pre_results.append((i, 'l2_short_gt', is_c, '', True, ''))
            continue
        # 短路 3
        cached = r.get('correct')
        if cached in ('0', '1'):
            pre_results.append((i, 'cached_correct', int(cached), '', True, ''))
            continue

        # 走 LLM
        prompt = VRS_L3_PROMPT_TPL.format(
            question=question, ground_truth=gt, predicted=pred)
        llm_prompts.append(prompt)
        llm_indices.append(i)

    print(f'  [短路] 前 3 步命中: {n_total - len(llm_prompts)}/{n_total}  '
          f'需 LLM: {len(llm_prompts)}', flush=True)

    # 第 2 遍：batch 推理
    print(f'  [Batch={batch_size}] LLM 推理 {len(llm_prompts)} 条', flush=True)
    n_done = 0
    for batch_start in range(0, len(llm_prompts), batch_size):
        batch_prompts = llm_prompts[batch_start:batch_start + batch_size]
        batch_idxs = llm_indices[batch_start:batch_start + batch_size]
        results = call_llm_text_batch(model, processor, batch_prompts, max_new_tokens)
        for rec_i, resp in zip(batch_idxs, results):
            if resp['ok']:
                is_c = parse_judgment(resp['response_text'])
                n_ok_llm += 1
            else:
                is_c = 0
            pre_results.append((rec_i, 'llm_judge', is_c,
                                resp.get('response_text', '')[:200],
                                resp['ok'], resp.get('error', '')[:200]))
            n_done += 1
        if n_done % (batch_size * 10) == 0 or n_done == len(llm_prompts):
            elapsed = time.time() - t0
            speed = n_done / max(elapsed, 0.001)
            eta = (len(llm_prompts) - n_done) / max(speed, 0.001)
            print(f'    [{n_done}/{len(llm_prompts)}] ok={n_ok_llm}  '
                  f'speed={speed:.1f}条/s  ETA={eta:.0f}s', flush=True)

    # 按 recs 顺序组装
    pre_results.sort(key=lambda x: x[0])
    out_recs = []
    for i, r in enumerate(recs):
        _, sc, is_c, resp_text, ok, err = pre_results[i]
        sc_counts[sc] += 1
        if is_c:
            n_correct += 1
            sc_correct[sc] += 1
        out_recs.append({
            **r,
            'llm_correct_l3': is_c,
            'llm_response':    resp_text,
            'llm_short_circuit': sc,
            'llm_ok':          ok,
            'llm_error':       err,
        })

    summary = {
        'total_l3':   n_total,
        'correct_l3': n_correct,
        'acc_l3':     round(n_correct / n_total, 4) if n_total else 0,
        'short_circuit_counts': sc_counts,
        'short_circuit_correct': sc_correct,
        'short_circuit_acc': {
            k: round(sc_correct[k] / sc_counts[k], 4) if sc_counts[k] else 0
            for k in sc_counts
        },
        'llm_ok_count': n_ok_llm,
        'batch_size':  batch_size,
        'method':      'vrs_4step_shortcircuit+batch',
        'judge_model': model.__class__.__name__,
        'temperature': TEMPERATURE,
        'do_sample':   DO_SAMPLE,
        'elapsed_s':   round(time.time() - t0, 1),
    }
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(out_recs, f, indent=2, ensure_ascii=False)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'prompt_tpl': VRS_L3_PROMPT_TPL,
                   'method': 'vrs_4step_shortcircuit+batch'},
                  f, indent=2, ensure_ascii=False)

    print(f'\n[完成] {l3_path.parent.name}: acc_l3={summary["acc_l3"]:.4f} ({n_correct}/{n_total})  '
          f'耗时={summary["elapsed_s"]}s', flush=True)
    print(f'  短路分布: {sc_counts}', flush=True)
    return summary


# ─── 批量：扫目录 ──────────────────────────────────────────
def judge_batch_dir(batch_dir: Path, model, processor, max_new_tokens: int,
                    batch_size: int = DEFAULT_BATCH_SIZE,
                    force: bool = False) -> dict:
    l3_files = sorted(batch_dir.glob('**/vqa_l3_raw.json'))
    if not l3_files:
        print(f'[LLM-L3] 在 {batch_dir} 下没找到 vqa_l3_raw.json')
        return {}

    print(f'[LLM-L3] 发现 {len(l3_files)} 个 vqa_l3_raw.json')
    print(f'[LLM-L3] judge model: {model.__class__.__name__}')
    print(f'[LLM-L3] 温度={TEMPERATURE}, do_sample={DO_SAMPLE}（强制温度 0）')
    print(f'[LLM-L3] batch_size={batch_size}（GPU 一次送 N 条 prompt）')

    t0 = time.time()
    total_l3, total_correct = 0, 0
    results = {}
    for i, l3f in enumerate(l3_files, 1):
        run_name = l3f.parent.name
        out_llm = l3f.parent / 'vqa_l3_llm.json'
        if out_llm.exists() and not force:
            met_p = l3f.parent / 'vqa_l3_llm_metrics.json'
            if met_p.exists():
                m = json.load(open(met_p))
                s = m.get('summary', {})
                if 'total_l3' in s:
                    total_l3 += s['total_l3']
                    total_correct += s['correct_l3']
                results[run_name] = s
            print(f'[{i}/{len(l3_files)}] 跳过 {run_name} (vqa_l3_llm.json 已存在)', flush=True)
            continue

        try:
            summary = judge_file(l3f, model, processor, max_new_tokens, batch_size)
            results[run_name] = summary
            total_l3 += summary['total_l3']
            total_correct += summary['correct_l3']
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'[{i}/{len(l3_files)}] 失败 {run_name}: {e}', flush=True)
            results[run_name] = {'error': str(e)}

    overall = {
        'total_l3':     total_l3,
        'correct_l3':   total_correct,
        'acc_l3':       round(total_correct / total_l3, 4) if total_l3 else 0,
        'elapsed_s':    round(time.time() - t0, 1),
        'judge_model':  model.__class__.__name__,
        'temperature':  TEMPERATURE,
        'do_sample':    DO_SAMPLE,
        'batch_size':   batch_size,
        'method':       'vrs_4step_shortcircuit+batch',
    }
    print(f'\n[LLM-L3] 全部完成: 累计 acc_l3={overall["acc_l3"]:.4f} ({total_correct}/{total_l3}), '
          f'耗时 {overall["elapsed_s"]}s')

    overview_path = batch_dir / 'vqa_l3_llm_overview.json'
    with open(overview_path, 'w', encoding='utf-8') as f:
        json.dump({'overall': overall, 'per_run': results, 'method': 'vrs_4step_shortcircuit+batch'},
                  f, indent=2, ensure_ascii=False)
    print(f'[LLM-L3] 总览: {overview_path}')

    return overall


# ─── 入口 ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='VRS-VQA L3 评测 — 本地大模型 (Qwen3.5-4B 等)，温度=0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('l3_json', nargs='?', help='单个 vqa_l3_raw.json 路径')
    parser.add_argument('--batch-dir', type=Path, default=None,
                        help='批量模式：扫该目录下所有 vqa_l3_raw.json')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help=f'本地模型名（默认: {DEFAULT_MODEL}），如 qwen3.5-4B / qwen3-vl-4B / gemma-4-e4b / minicpm-v-4.6')
    parser.add_argument('--model-root', type=str, default=MODEL_ROOT,
                        help=f'模型根目录（默认: {MODEL_ROOT}）')
    parser.add_argument('--max-new', type=int, default=DEFAULT_MAX_NEW,
                        help=f'LLM 生成 token 上限（默认: {DEFAULT_MAX_NEW}，判定只需 1 token）')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'batch 推理大小（默认: {DEFAULT_BATCH_SIZE}，实测 16 比 8 再快 2x）')
    parser.add_argument('--force', action='store_true', help='覆盖已存在的 vqa_l3_llm.json')

    args = parser.parse_args()

    if not args.l3_json and not args.batch_dir:
        print('[ERROR] 必须传 l3_json 或 --batch-dir')
        parser.print_help()
        sys.exit(1)

    # 加载本地 LLM（只加载一次）
    print(f'[LLM-L3] 加载模型: {args.model}  (root: {args.model_root})')
    t_load = time.time()
    import torch
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

    sys.path.insert(0, '/home/admin1/projects/remote_vlm_eval')
    os.environ['PYTHONPATH'] = '/home/admin1/projects/remote_vlm_eval'
    from inference import load_model  # noqa: E402
    model, processor = load_model(args.model, args.model_root)
    print(f'[LLM-L3] 加载完成，耗时 {time.time()-t_load:.1f}s')
    print(f'[LLM-L3] 确认参数: temperature={TEMPERATURE}, do_sample={DO_SAMPLE}')

    # 显式断言温度 0（张哥硬性要求）
    assert TEMPERATURE == 0.0, f'温度必须 = 0，当前 {TEMPERATURE}'
    assert DO_SAMPLE is False, f'do_sample 必须 = False（对应温度 0），当前 {DO_SAMPLE}'

    try:
        if args.l3_json:
            judge_file(Path(args.l3_json), model, processor, args.max_new, args.batch_size)
        else:
            judge_batch_dir(args.batch_dir, model, processor, args.max_new,
                           args.batch_size, force=args.force)
    finally:
        # 释放显存
        del model, processor
        import gc
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
