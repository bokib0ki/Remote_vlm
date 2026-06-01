#!/usr/bin/env python3
"""
从固定 selection JSON 中读取要评测的题目集合，再从对应 benchmark 的源数据集中抽取这些题目并评测。

【重要】现在使用 bench_cofig/ 目录下的官方配置进行评测，
包括官方 prompt、评测方式和生成长度。

用法:
  python3 eval_select.py --select /path/to/selection.json --model qwen3.5-4B
  python3 eval_select.py --select /path/to/selection.json --model qwen3.5-4B --thinking --max_new 512
  python3 eval_select.py --select /path/to/selection.json --all
"""
import os, sys, json, time, argparse
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import torch
from PIL import Image

from config import (
    ROOT, ALL_MODELS, DUAL_MODELS,
    VRS_IMG_DIR, LEVIR_DIR, MME_RS_DIR, TEST_IMGS,
    RAW_DIR_ROOT, RES_DIR,
)
from inference import load_model, infer, strip_thinking, extract_letter, safe_img
from metrics import compute_caption_scores
from model_store import save_run_from_raw_records
from bench_config_loader import load_bench_config, build_prompt, get_max_new_tokens, check_temperature

print = lambda *a, **kw: __builtins__.print(*a, **kw, flush=True)


def compute_caption_scores_safe(refs: dict, hyps: dict) -> dict:
    if not refs or not hyps:
        return {'bleu1': 0.0, 'bleu2': 0.0, 'bleu3': 0.0, 'bleu4': 0.0, 'rouge_l': 0.0, 'cider': 0.0}
    return compute_caption_scores(refs, hyps)


def perf_stats(times_s: list[float], tokens: list[int]) -> dict:
    total_time = float(sum(times_s)) if times_s else 0.0
    total_tokens = int(sum(tokens)) if tokens else 0
    n = int(len(times_s))
    return {
        'n': n,
        'total_time_s': round(total_time, 4),
        'total_tokens': total_tokens,
        'avg_time_s': round(total_time / n, 4) if n else 0.0,
        'avg_tokens': round(total_tokens / n, 1) if n else 0.0,
        'avg_s_per_token': round(total_time / total_tokens, 6) if total_tokens else None,
        'avg_speed': round(total_tokens / total_time, 2) if total_time else None,
    }


def get_test_img(i=0):
    if TEST_IMGS:
        return safe_img(str(TEST_IMGS[i % len(TEST_IMGS)]), fallback_dir=str(TEST_IMGS[0].parent))
    return Image.new('RGB', (512, 512), 'gray')


def _as_list(obj):
    if obj is None:
        return []
    return obj if isinstance(obj, list) else [obj]


def _load_json(path: str | Path):
    p = Path(path)
    with open(p) as f:
        return json.load(f)


def _unwrap_records(name: str, data):
    if name == 'levir_cc':
        return data.get('images', data)
    return data


def _key_getter(name: str, rec: dict, key: str):
    if key == 'index':
        return None
    if key in rec:
        return rec.get(key)
    aliases = {
        'vrs_caption': ['question_id', '_original_index', 'id'],
        'vrs_vqa': ['question_id', '_original_index', 'id'],
        'mme_rs': ['Question_id', '_original_index', 'id'],
        'levir_cc': ['imgid', '_original_index', 'id', 'filename'],
        'xlrs': ['idx', '_original_index', 'id'],
    }
    for k in aliases.get(name, []):
        if k in rec:
            return rec.get(k)
    return None


def load_selected_benchmark(bench_name: str, cfg: dict):
    source = cfg.get('source')
    if not source:
        raise ValueError(f"{bench_name}: missing 'source'")
    data = _load_json(source)
    records = _unwrap_records(bench_name, data)
    if not isinstance(records, list):
        raise ValueError(f"{bench_name}: source json is not a list (or images list)")

    if cfg.get('all') is True:
        return records

    select_by = cfg.get('select_by', 'index')
    ids = cfg.get('ids')
    items = cfg.get('items')
    if ids is None and items is None:
        raise ValueError(f"{bench_name}: need 'ids' or 'items'")
    if ids is not None and items is not None:
        raise ValueError(f"{bench_name}: only one of 'ids' or 'items' should be set")

    if items is not None:
        ids = [it.get('id') for it in _as_list(items)]

    if select_by == 'index':
        picked = []
        for i in ids:
            if i is None:
                continue
            if 0 <= int(i) < len(records):
                picked.append(records[int(i)])
        return picked

    idx = {}
    for r_i, rec in enumerate(records):
        k = _key_getter(bench_name, rec, select_by)
        if k is None:
            continue
        idx[str(k)] = rec

    picked = []
    for i in ids:
        if i is None:
            continue
        rec = idx.get(str(i))
        if rec is not None:
            picked.append(rec)
    return picked


def load_selection(select_path: str):
    doc = _load_json(select_path)
    benches = doc.get('benchmarks', {})
    if not isinstance(benches, dict) or not benches:
        raise ValueError("selection json missing 'benchmarks' dict")
    return doc


def _get_gt_vrs_vqa(ann: dict) -> str:
    """从 VRS-VQA 记录中提取 ground_truth。"""
    return str(ann.get('ground_truth', '') or ann.get('answer', '')).strip()


def _get_gt_mme_rs(ann: dict) -> str:
    """从 MME-RS 记录中提取 ground_truth（大写字母）。"""
    gt = str(ann.get('Ground truth', '') or ann.get('answer', '')).strip().upper()
    # 清理可能的前缀如 "A." -> "A"
    import re
    gt = re.sub(r'^[A-E]\.\s*', '', gt)
    return gt


def _get_gt_xlrs(ann: dict) -> str:
    """从 XLRS 记录中提取 ground_truth。"""
    gt = str(ann.get('answer', '')).strip().upper()
    import re
    gt = re.sub(r'^[A-D]\.\s*', '', gt)
    return gt


def _get_question(ann: dict, bench_name: str) -> str:
    """从记录中提取 question。"""
    if bench_name == 'mme_rs':
        return ann.get('Text', '') or ann.get('question', '')
    return ann.get('question', '') or ann.get('Text', '')


def _get_choices_mme_rs(ann: dict) -> list:
    """从 MME-RS 记录中提取 choices。"""
    return ann.get('Answer choices', []) or ann.get('choices', [])


def _get_choices_xlrs(ann: dict) -> list:
    """从 XLRS 记录中提取 choices。"""
    return ann.get('multi-choice options', []) or ann.get('choices', [])


def run_eval(model_name: str, enable_thinking: bool, select_doc: dict, max_new: int | None):
    mode_tag = '_thinkON' if enable_thinking else '_thinkOFF'
    select_name = Path(select_doc.get('name') or Path(select_doc.get('_path', 'selection')).stem).stem
    config_tag = f"{model_name}{mode_tag}_{select_name}"
    print(f"\n{'='*60}")
    print(f"  {config_tag}")
    print(f"  [bench_cofig] 官方配置模式")
    print(f"{'='*60}")

    run_started_at = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    model, processor = load_model(model_name, str(ROOT))
    total_vram = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM: {total_vram:.1f}GB")

    def _run(img, prompt, extra_imgs=None, max_new_tokens=max_new):
        t0 = time.time()
        res = infer(
            model, processor, img, prompt,
            max_new_tokens=max_new_tokens,
            extra_imgs=extra_imgs,
            enable_thinking=enable_thinking,
            do_sample=False,  # 严格 temperature=0
        )
        dt = time.time() - t0
        res['speed'] = round(res['out_tokens'] / dt, 2) if dt > 0 else 0.0
        clean = res['out_text']
        raw = res['raw']
        return clean, raw, res, dt

    # 基础配置
    effective_max_new = max_new if max_new is not None else (25000 if enable_thinking else 128)
    
    row = {
        'config': config_tag,
        'model': model_name,
        'thinking': enable_thinking,
        'max_new': effective_max_new,
        'selection': {
            'path': select_doc.get('_path'),
            'name': select_doc.get('name'),
        },
        'benchmarks': {},
        'bench_cofig_mode': True,  # 标记使用官方配置
    }
    raw_records = []

    benches = select_doc['benchmarks']
    only = set(select_doc.get('_only_benchmarks') or [])
    if only:
        benches = {k: v for k, v in benches.items() if k in only}
    benches_run = set(benches.keys())

    # 遍历每个 benchmark，使用 bench_cofig 中的官方配置
    for bench_name in ['vrs_caption', 'vrs_vqa', 'mme_rs', 'levir_cc', 'xlrs']:
        if bench_name not in benches:
            continue
        
        print(f"\n  {bench_name}...")
        
        # 加载官方配置（严格模式：必须存在）
        try:
            bench_cfg = load_bench_config(bench_name)
        except Exception as e:
            raise RuntimeError(f"[{bench_name}] bench_cofig 加载失败，必须使用官方配置: {e}")
        
        # 检查温度配置
        if bench_cfg and not check_temperature(bench_cfg):
            print(f"    [WARN] {bench_name} temperature != 0.0，已强制设为 0.0")
            bench_cfg['temperature'] = 0.0
        
        anns = load_selected_benchmark(bench_name, benches[bench_name])
        
        if bench_name == 'vrs_caption':
            # VRS-Caption: Caption 任务
            # 官方配置: PROMPT_TEMPLATE = "Describe this remote sensing image in detail."
            # 评测方式: BLEU/ROUGE/CIDER
            prompt_template = bench_cfg.get('prompt_template', 'Describe this remote sensing image in detail.')
            max_new_tokens = get_max_new_tokens(bench_name, enable_thinking, bench_cfg)
            if max_new_tokens is None:
                max_new_tokens = effective_max_new
            
            refs, hyps = {}, {}
            token_lens = []
            times_s = []
            
            for i, ann in enumerate(anns):
                img = safe_img(str(VRS_IMG_DIR / ann.get('image_id', ann.get('image', ''))),
                               fallback_dir=str(TEST_IMGS[0].parent) if TEST_IMGS else None)
                gt = str(ann.get('ground_truth', '') or ann.get('caption', '')).strip()
                if not gt:
                    continue
                
                # 使用官方 prompt
                prompt = prompt_template
                clean, raw, res, dt = _run(img, prompt, max_new_tokens=max_new_tokens)
                nt = res['out_tokens']
                token_lens.append(nt)
                times_s.append(dt)
                refs[str(i)] = [gt]
                hyps[str(i)] = [clean]
                raw_records.append({
                    'benchmark': 'VRS-Caption',
                    '_idx': ann.get('question_id', ann.get('_original_index', i)),
                    'image_id': ann.get('image_id', ann.get('image', '')),
                    'gt': gt, 'pred': clean, 'tokens': nt,
                    'pred_raw': raw,
                    'time_s': round(dt, 4),
                    'speed': res['speed'],
                    'input_tokens': res['input_tokens'],
                    'img_tokens': res['img_tokens'],
                    'prompt_tokens': res['prompt_tokens'],
                    'thinking_tokens': res['thinking_tokens'],
                    'answer_tokens': res['answer_tokens'],
                    'prompt_used': prompt,  # 记录使用的官方 prompt
                })
            scores = compute_caption_scores_safe(refs, hyps)
            stats = {'min': min(token_lens), 'max': max(token_lens),
                     'avg': round(sum(token_lens)/len(token_lens), 1)} if token_lens else {}
            row['benchmarks']['vrs_caption'] = {**scores, 'token_stats': stats, 'perf': perf_stats(times_s, token_lens), 'n': len(anns)}
            print(f"    done ({time.time()-t0:.0f}s) n={len(anns)} bleu4={scores['bleu4']}")
            
        elif bench_name == 'vrs_vqa':
            # VRS-VQA: VQA 任务
            # 官方配置: PROMPT_TEMPLATE = "{question}\nAnswer the question using a single word or phrase."
            # 评测方式: L1/L2/L3 三级评测（L1 substring → L2 yes/no/数字 → L3 Qwen3.7语义）
            prompt_template = bench_cfg.get('prompt_template', "{question}\nAnswer the question using a single word or phrase.")
            
            # 用户传了 --max_new 就用用户的，没传才用 bench_cofig 的
            # bench_cofig 定义: VRS-VQA thinkOFF=64, thinkON=None（无上限，用用户值或默认4096）
            if max_new is not None:
                max_new_tokens = max_new
            else:
                configured_max = get_max_new_tokens(bench_name, enable_thinking, bench_cfg)
                max_new_tokens = configured_max if configured_max is not None else (4096 if enable_thinking else 64)
            
            # 导入 VQA 三级评测器
            from vqa_judge import vqa_judge
            
            correct, total = 0, 0
            correct_l1, correct_l2, correct_l3 = 0, 0, 0
            vqa_records = []
            token_lens = []
            times_s = []
            
            for i, ann in enumerate(anns):
                img = safe_img(str(VRS_IMG_DIR / ann.get('image_id', ann.get('image', ''))),
                               fallback_dir=str(TEST_IMGS[0].parent) if TEST_IMGS else None)
                q = _get_question(ann, bench_name)
                gt = _get_gt_vrs_vqa(ann)
                if not gt:
                    continue
                
                # 构建 img_path 用于 L3 评测
                img_id = ann.get('image_id', ann.get('image', ''))
                img_path = str(VRS_IMG_DIR / img_id) if img_id else None
                
                # 使用官方 prompt（构建完整的格式化 prompt）
                prompt = build_prompt(bench_name, {'question': q}, bench_cfg)
                clean, raw, res, dt = _run(img, prompt, max_new_tokens=max_new_tokens)
                nt = res['out_tokens']
                
                # VRSBench 官方三级评测: L1 substring → L2 yes/no/数字 → L3 Qwen3.7语义
                judge_result = vqa_judge(q, gt, clean, image_path=img_path)
                ok = judge_result['correct']
                
                # 统计各级别
                level = judge_result.get('level', 'L1')
                if level == 'L1':
                    correct_l1 += 1
                elif level == 'L2':
                    correct_l2 += 1
                elif level == 'L3':
                    correct_l3 += 1
                
                correct += ok
                total += 1
                token_lens.append(nt)
                times_s.append(dt)
                
                rec = {
                    'benchmark': 'VRS-VQA',
                    '_idx': ann.get('question_id', ann.get('_original_index', i)),
                    'image_id': img_id,
                    'gt': gt, 'pred': clean, 'tokens': nt,
                    'question': q, 'correct': ok,
                    'judge_level': level,
                    'judge_method': judge_result.get('method', 'substring'),
                    'pred_raw': raw,
                    'time_s': round(dt, 4),
                    'speed': res['speed'],
                    'input_tokens': res['input_tokens'],
                    'img_tokens': res['img_tokens'],
                    'prompt_tokens': res['prompt_tokens'],
                    'thinking_tokens': res['thinking_tokens'],
                    'answer_tokens': res['answer_tokens'],
                    'prompt_used': prompt,
                }
                vqa_records.append(rec)
                raw_records.append(rec)
            
            row['benchmarks']['vrs_vqa'] = {
                'acc': round(correct / max(total, 1), 4),  # 总准确率（三级合并）
                'acc_l1': round(correct_l1 / max(total, 1), 4),
                'correct_l1': correct_l1,
                'correct_l2': correct_l2,
                'correct_l3': correct_l3,
                'total': total,
                'n': len(anns),
                'perf': perf_stats(times_s, token_lens),
            }
            raw_dir = RAW_DIR_ROOT / select_name / config_tag
            raw_dir.mkdir(parents=True, exist_ok=True)
            with open(raw_dir / 'vqa_raw.json', 'w') as f:
                json.dump(vqa_records, f, indent=2)
            print(f"    done ({time.time()-t0:.0f}s) n={len(anns)} acc={correct}/{total} (L1={correct_l1}/L2={correct_l2}/L3={correct_l3})")
            
        elif bench_name == 'mme_rs':
            # MME-RS: 多选 VQA
            # 官方配置: PROMPT_TEMPLATE = "{question}\nThe choices are listed below:\n{choices}\n..."
            # 评测方式: 准确率（提取选项字母）
            prompt_template = bench_cfg.get('prompt_template')
            
            # 用户传了 --max_new 就用用户的，没传才用 bench_cofig 的
            configured_max = get_max_new_tokens(bench_name, enable_thinking, bench_cfg)
            if max_new is not None:
                max_new_tokens = max_new
            else:
                max_new_tokens = configured_max if configured_max is not None else 10
            
            correct, total = 0, 0
            token_lens = []
            times_s = []
            
            for i, ann in enumerate(anns):
                img_path_str = ann.get('Image', '') or ann.get('image', '')
                if MME_RS_DIR.exists() and img_path_str:
                    fp = MME_RS_DIR / Path(img_path_str).name
                    img = safe_img(str(fp)) if fp.exists() else get_test_img(i)
                else:
                    img = get_test_img(i)
                
                q = _get_question(ann, bench_name)
                choices = _get_choices_mme_rs(ann)
                gt = _get_gt_mme_rs(ann)
                
                # 构建 prompt（必须从 bench_cofig 获取）
                if not prompt_template:
                    raise RuntimeError(f"[{bench_name}] bench_cofig 缺少 PROMPT_TEMPLATE 配置")
                choices_str = '\n'.join(choices) if isinstance(choices, list) else str(choices)
                prompt = prompt_template.format(question=q, choices=choices_str)
                
                clean, raw, res, dt = _run(img, prompt, max_new_tokens=max_new_tokens)
                nt = res['out_tokens']
                token_lens.append(nt)
                times_s.append(dt)
                pred = extract_letter(res['out_text'])
                ok = 1 if pred == gt else 0
                correct += ok
                total += 1
                raw_records.append({
                    'benchmark': 'MME-RS',
                    '_idx': ann.get('Question_id', ann.get('_original_index', i)),
                    'image_id': img_path_str,
                    'gt': gt, 'pred': pred, 'tokens': nt, 'question': q,
                    'pred_raw': raw,
                    'time_s': round(dt, 4),
                    'speed': res['speed'],
                    'input_tokens': res['input_tokens'],
                    'img_tokens': res['img_tokens'],
                    'prompt_tokens': res['prompt_tokens'],
                    'thinking_tokens': res['thinking_tokens'],
                    'answer_tokens': res['answer_tokens'],
                    'prompt_used': prompt,
                })
            
            row['benchmarks']['mme_rs'] = {
                'acc': round(correct / max(total, 1), 4),
                'correct': correct,
                'total': total,
                'n': len(anns),
                'perf': perf_stats(times_s, token_lens),
                'token_stats': {'min': min(token_lens), 'max': max(token_lens),
                                'avg': round(sum(token_lens)/len(token_lens), 1)} if token_lens else {},
            }
            print(f"    done ({time.time()-t0:.0f}s) n={len(anns)} acc={correct}/{total}")
            
        elif bench_name == 'levir_cc':
            # LEVIR-CC: 变化描述（双图）
            # 官方配置: PROMPT_TEMPLATE = "Describe the changes between these two remote sensing images..."
            # 评测方式: BLEU/ROUGE/CIDER
            prompt_template = bench_cfg.get('prompt_template', "Describe the changes between these two remote sensing images taken at different times.")
            max_new_tokens = get_max_new_tokens(bench_name, enable_thinking, bench_cfg)
            if max_new_tokens is None:
                max_new_tokens = effective_max_new
            
            refs, hyps = {}, {}
            token_lens = []
            times_s = []
            
            for i, ann in enumerate(anns):
                fname = ann.get('filename') or ann.get('image') or ''
                before = safe_img(str(LEVIR_DIR / 'A' / fname),
                                  fallback_dir=str(TEST_IMGS[0].parent) if TEST_IMGS else None)
                after = safe_img(str(LEVIR_DIR / 'B' / fname),
                                 fallback_dir=str(TEST_IMGS[0].parent) if TEST_IMGS else None)
                refs_list = [s.get('raw', '').strip() for s in ann.get('sentences', []) if s.get('raw')]
                
                # 使用官方 prompt
                prompt = prompt_template
                clean, raw, res, dt = _run(before, prompt, extra_imgs=[after], max_new_tokens=max_new_tokens)
                nt = res['out_tokens']
                token_lens.append(nt)
                times_s.append(dt)
                if refs_list:
                    refs[str(i)] = refs_list
                    hyps[str(i)] = [clean]
                raw_records.append({
                    'benchmark': 'LEVIR-CC',
                    '_idx': ann.get('imgid', ann.get('_original_index', i)),
                    'image_id': fname,
                    'gt': ' || '.join(refs_list),
                    'pred': clean, 'tokens': nt,
                    'pred_raw': raw,
                    'time_s': round(dt, 4),
                    'speed': res['speed'],
                    'input_tokens': res['input_tokens'],
                    'img_tokens': res['img_tokens'],
                    'prompt_tokens': res['prompt_tokens'],
                    'thinking_tokens': res['thinking_tokens'],
                    'answer_tokens': res['answer_tokens'],
                    'prompt_used': prompt,
                })
            
            scores = compute_caption_scores_safe(refs, hyps)
            stats = {'min': min(token_lens), 'max': max(token_lens),
                     'avg': round(sum(token_lens)/len(token_lens), 1)} if token_lens else {}
            row['benchmarks']['levir_cc'] = {**scores, 'token_stats': stats, 'perf': perf_stats(times_s, token_lens), 'n': len(anns)}
            print(f"    done ({time.time()-t0:.0f}s) n={len(anns)} bleu4={scores['bleu4']}")
            
        elif bench_name == 'xlrs':
            # XLRS: 多选 VQA（超高分辨率）
            # 官方配置: PROMPT_TEMPLATE = "{question}\nThe choices are listed below:\n{choices}\n..."
            # 评测方式: 准确率（提取选项字母）
            prompt_template = bench_cfg.get('prompt_template')
            
            # 用户传了 --max_new 就用用户的，没传才用 bench_cofig 的
            configured_max = get_max_new_tokens(bench_name, enable_thinking, bench_cfg)
            if max_new is not None:
                max_new_tokens = max_new
            else:
                max_new_tokens = configured_max if configured_max is not None else 10
            
            correct, total = 0, 0
            token_lens = []
            times_s = []
            
            for i, ann in enumerate(anns):
                local_img = ann.get('local_image', '')
                if local_img and Path(local_img).exists():
                    img = Image.open(local_img).convert('RGB')
                else:
                    img = get_test_img(i)
                
                q = ann.get('question', '')
                choices = _get_choices_xlrs(ann)
                gt = _get_gt_xlrs(ann)
                gt_set = set(gt.replace(',', ' ').split()) if gt else set()
                
                # 构建 prompt（必须从 bench_cofig 获取）
                if not prompt_template:
                    raise RuntimeError(f"[{bench_name}] bench_cofig 缺少 PROMPT_TEMPLATE 配置")
                choices_str = '\n'.join(choices) if isinstance(choices, list) else str(choices)
                prompt = prompt_template.format(question=q, choices=choices_str)
                
                clean, raw, res, dt = _run(img, prompt, max_new_tokens=max_new_tokens)
                nt = res['out_tokens']
                token_lens.append(nt)
                times_s.append(dt)
                letter = extract_letter(res['out_text'], choices='ABCD')
                pred_set = {letter} if letter else set()
                ok = 1 if gt_set and pred_set == gt_set else 0
                correct += ok
                total += 1
                raw_records.append({
                    'benchmark': 'XLRS',
                    '_idx': ann.get('idx', ann.get('_original_index', i)),
                    'image_id': ann.get('path', ''),
                    'gt': gt, 'pred': list(pred_set), 'tokens': nt,
                    'question': q[:60],
                    'pred_raw': raw,
                    'time_s': round(dt, 4),
                    'speed': res['speed'],
                    'input_tokens': res['input_tokens'],
                    'img_tokens': res['img_tokens'],
                    'prompt_tokens': res['prompt_tokens'],
                    'thinking_tokens': res['thinking_tokens'],
                    'answer_tokens': res['answer_tokens'],
                    'prompt_used': prompt,
                })
            
            row['benchmarks']['xlrs'] = {
                'acc': round(correct / max(total, 1), 4),
                'correct': correct,
                'total': total,
                'n': len(anns),
                'perf': perf_stats(times_s, token_lens),
                'token_stats': {'min': min(token_lens), 'max': max(token_lens),
                                'avg': round(sum(token_lens)/len(token_lens), 1)} if token_lens else {},
            }
            print(f"    done ({time.time()-t0:.0f}s) n={len(anns)} acc={correct}/{total}")

    res_dir = RES_DIR / select_name
    res_dir.mkdir(parents=True, exist_ok=True)
    res_file = res_dir / f'{config_tag}_metrics.json'
    row_out = row
    if benches_run and only and res_file.exists():
        try:
            with open(res_file) as f:
                prev = json.load(f)
            if isinstance(prev, dict):
                prev_bm = prev.setdefault('benchmarks', {})
                for k, v in row.get('benchmarks', {}).items():
                    prev_bm[k] = v
                prev['max_new'] = effective_max_new
                prev['thinking'] = enable_thinking
                prev['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                row_out = prev
        except Exception:
            row_out = row
    with open(res_file, 'w') as f:
        json.dump(row_out, f, indent=2)
    print(f"  Saved metrics: {res_file}")

    raw_dir = RAW_DIR_ROOT / select_name / config_tag
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / 'raw_outputs.json'
    raw_out = raw_records
    if benches_run and only and raw_file.exists():
        labels_run = {'VRS-Caption', 'VRS-VQA', 'MME-RS', 'LEVIR-CC', 'XLRS'}
        labels_run = labels_run.intersection({r.get('benchmark') for r in raw_records})
        try:
            with open(raw_file) as f:
                prev_raw = json.load(f)
            if isinstance(prev_raw, list):
                kept = [r for r in prev_raw if r.get('benchmark') not in labels_run]
                raw_out = kept + raw_records
        except Exception:
            raw_out = raw_records
    with open(raw_file, 'w') as f:
        json.dump(raw_out, f, indent=2)
    print(f"  Saved raw: {raw_file} ({len(raw_out)} records)")

    run_ended_at = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    run_meta = {
        'config_tag': config_tag,
        'thinking': enable_thinking,
        'max_new': max_new,
        'selection': select_name,
        'started_at': run_started_at,
        'ended_at': run_ended_at,
        'run_id': f'{config_tag}_{run_ended_at}',
    }
    save_run_from_raw_records(model_name, run_meta, raw_records)

    del model, processor
    torch.cuda.empty_cache()
    print(f"  [{config_tag}] DONE")
    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--select', required=True, help='selection JSON 路径')
    parser.add_argument('--model', action='append', help='模型名称（可重复/逗号分隔）')
    parser.add_argument('--all', action='store_true', help='批量评测所有模型')
    parser.add_argument('--thinking', action='store_true', help='启用 thinkON 模式')
    parser.add_argument('--max_new', type=int, default=None, help='最大生成 tokens（thinkON 默认 25000；thinkOFF 默认 128）')
    parser.add_argument('--bench', default='', help='只跑指定 benchmark（逗号分隔）：vrs_vqa,vrs_caption,mme_rs,levir_cc,xlrs')
    args = parser.parse_args()

    select_doc = load_selection(args.select)
    select_doc['_path'] = str(Path(args.select).resolve())
    if args.bench:
        select_doc['_only_benchmarks'] = [x.strip() for x in args.bench.split(',') if x.strip()]

    def _split_models(xs):
        out = []
        for x in xs or []:
            for p in str(x).split(','):
                p = p.strip()
                if p:
                    out.append(p)
        return out

    models_to_run = []
    if args.all:
        models_to_run = [(m, False) for m in ALL_MODELS]
        models_to_run += [(m, True) for m in DUAL_MODELS]
    elif args.model:
        ms = _split_models(args.model)
        models_to_run = [(m, args.thinking) for m in ms]
    else:
        parser.print_help()
        sys.exit(1)

    for model_name, enable_thinking in models_to_run:
        run_eval(model_name, enable_thinking, select_doc, args.max_new)