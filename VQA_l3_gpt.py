#!/usr/bin/env python3
"""
VQA L3 评测 — 阿里云百炼 GLM-5 (OpenAI 兼容模式)
====================================================

设计:
  - 输入: vqa_l3_raw.json (由 apply_model_adapter.py 重生成)
  - API: 阿里云百炼兼容 OpenAI Chat Completions (复用 eval_sota.py 的 call_openai_compatible)
  - 评测 prompt: 参考 VRS 官方 (vqa_judge.py 第 152-157 行)
  - 范围: 对经过 L1/L2 仍未判定的 L3 题做语义匹配
  - 输出:
      1) vqa_l3_gpt.json          - 逐条记录 + 响应 + 判定
      2) vqa_l3_gpt_metrics.json  - {summary: {total_l3, correct_l3, acc_l3}}

用法:
  export DASHSCOPE_API_KEY=sk-xxxx
  python VQA_l3_gpt.py --batch-dir /home/admin1/models/raw_outputs/lever_test2
  python VQA_l3_gpt.py --raw-path <single_vqa_l3_raw.json>
  python VQA_l3_gpt.py --model glm-4.5  # 切换到 GLM-4.5
"""
import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

# ─── 默认配置（阿里云百炼 OpenAI 兼容模式） ───────────────────
DASHSCOPE_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
DASHSCOPE_API_KEY = os.environ.get('DASHSCOPE_API_KEY', '')
DEFAULT_MODEL = 'glm-5'              # 阿里云百炼上的 GLM-5
DEFAULT_MAX_NEW = 16                  # L3 判定只需 1 个 token
DEFAULT_TIMEOUT = 60
DEFAULT_CONCURRENCY = 8               # 简单并发
DEFAULT_RETRIES = 3

# ─── VRS 官方 L3 评测 prompt (参考 vqa_judge.py:152-157) ───────
# 原文 (GPT-4o-mini):
#   Question: {question}
#   Ground Truth Answer: {gt}
#   Predicted Answer: {pred}
#   Does the predicted answer match the ground truth?
#   Consider synonyms as matches (e.g., football/soccer, building/rooftop, pond/swimming pool).
#   Answer with only "1" for match or "0" for no match. Do not explain.
VRS_L3_PROMPT_TPL = """Question: {question}
Ground Truth Answer: {gt}
Predicted Answer: {pred}
Does the predicted answer match the ground truth?
Consider synonyms as matches (e.g., football/soccer, building/rooftop, pond/swimming pool).
Answer with only "1" for match or "0" for no match. Do not explain."""


# ─── HTTP 工具（直接 urllib，不依赖 openai SDK） ──────────────
def _http_post_json(url: str, headers: dict, body: dict, timeout_s: int) -> dict:
    data = json.dumps(body).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={
        'Content-Type': 'application/json',
        **headers,
    }, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode('utf-8', errors='replace')
            return {'_http_status': resp.status, **json.loads(raw)}
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8', errors='replace') if e.fp else ''
        try:
            return {'_http_status': e.code, 'error': body, **json.loads(body)}
        except Exception:
            return {'_http_status': e.code, 'error': body}
    except Exception as e:
        return {'_http_status': -1, 'error': str(e)}


def _read_image_bytes(path: str) -> Optional[tuple[bytes, str]]:
    """读图片为 (bytes, mime)"""
    p = Path(path)
    if not p.exists():
        return None
    try:
        b = p.read_bytes()
        mime, _ = mimetypes.guess_type(str(p))
        if not mime:
            mime = 'image/png' if p.suffix.lower() == '.png' else 'image/jpeg'
        return b, mime
    except Exception:
        return None


def _data_url(b: bytes, mime: str) -> str:
    b64 = base64.b64encode(b).decode('ascii')
    return f'data:{mime};base64,{b64}'


# ─── 调用 GLM-5（OpenAI 兼容 Chat Completions） ──────────────
def call_glm(prompt: str,
             image_paths: list[str],
             model: str,
             api_key: str,
             base_url: str,
             max_new: int,
             timeout_s: int) -> dict:
    """
    返回 {
        'ok': bool,
        'response_text': str,
        'error': str,
        'http_status': int,
    }
    """
    if not api_key:
        return {'ok': False, 'error': 'DASHSCOPE_API_KEY 未设置（export DASHSCOPE_API_KEY=sk-...）',
                'response_text': '', 'http_status': -1}

    # 构建 content
    content = []
    for img_path in image_paths:
        if not img_path:
            continue
        img = _read_image_bytes(img_path)
        if not img:
            continue
        b, mime = img
        content.append({'type': 'image_url', 'image_url': {'url': _data_url(b, mime)}})
    content.append({'type': 'text', 'text': prompt})

    url = base_url.rstrip('/') + '/chat/completions'
    body = {
        'model': model,
        'messages': [{'role': 'user', 'content': content}],
        'max_tokens': max_new,
        'temperature': 0,  # 判定任务要确定性
    }
    headers = {'Authorization': f'Bearer {api_key}'}

    last_err = ''
    for attempt in range(DEFAULT_RETRIES):
        resp = _http_post_json(url, headers, body, timeout_s)
        status = resp.get('_http_status', -1)
        if status == 200 and 'choices' in resp:
            try:
                txt = resp['choices'][0]['message']['content']
                return {
                    'ok': True,
                    'response_text': txt.strip(),
                    'error': '',
                    'http_status': status,
                }
            except (KeyError, IndexError) as e:
                last_err = f'parse error: {e}, resp={json.dumps(resp)[:200]}'
        else:
            err = resp.get('error', resp.get('message', 'unknown error'))
            last_err = f'status={status} err={str(err)[:200]}'
        time.sleep(0.5 * (attempt + 1))  # backoff
    return {'ok': False, 'response_text': '', 'error': last_err, 'http_status': status}


# ─── 提取判定结果 ────────────────────────────────────────────
def parse_judgment(response_text: str) -> int:
    """
    从 GLM-5 响应中提取 0/1
    优先匹配响应开头的第一个 0/1
    兜底：扫描全文第一个 0 或 1
    模糊匹配（仅当以上都失败）：
      "not match" / "no" / "false" → 0
      "yes" / "true" / "match" → 1
    """
    if not response_text:
        return 0
    s = response_text.strip()

    # 优先级 1: 第一个字符
    first = s[0]
    if first in ('0', '1'):
        return int(first)

    # 优先级 2: 第一个 '0'/'1' 字符
    m = re.search(r'\b([01])\b', s)
    if m:
        return int(m.group(1))

    # 优先级 3: 模糊匹配（注意否定词优先级 > 肯定词）
    lower = s.lower()
    # 否定信号优先（覆盖各种缩写）
    if 'not match' in lower or 'not_match' in lower \
            or "doesn't match" in lower or 'doesnt match' in lower \
            or 'does not match' in lower or 'do not match' in lower \
            or 'no match' in lower or 'not a match' in lower \
            or 'incorrect' in lower or 'wrong' in lower:
        return 0
    # 肯定信号
    if 'match' in lower or 'yes' in lower or 'true' in lower or 'correct' in lower:
        return 1
    # 单独的 "no" / "false" 兜底
    if re.search(r'\b(no|false)\b', lower):
        return 0
    if re.search(r'\b(yes|true)\b', lower):
        return 1
    return 0  # 兜底


# ─── 评测单条 ───────────────────────────────────────────────
def judge_one(question: str, gt: str, pred: str, image_path: Optional[str],
              model: str, api_key: str, base_url: str, max_new: int, timeout_s: int) -> dict:
    prompt = VRS_L3_PROMPT_TPL.format(question=question, gt=gt, pred=pred)
    img_paths = [image_path] if image_path else []
    resp = call_glm(prompt, img_paths, model, api_key, base_url, max_new, timeout_s)
    is_c = parse_judgment(resp.get('response_text', '')) if resp['ok'] else 0
    return {
        'correct_l3': is_c,
        'response_text': resp.get('response_text', ''),
        'ok': resp['ok'],
        'error': resp.get('error', ''),
        'http_status': resp.get('http_status', -1),
    }


# ─── 批量：单文件 ──────────────────────────────────────────
def judge_file(l3_path: Path,
               model: str, api_key: str, base_url: str, max_new: int, timeout_s: int,
               vrs_img_dir: Path) -> dict:
    if not l3_path.exists():
        raise FileNotFoundError(f'文件不存在: {l3_path}')

    recs = json.load(open(l3_path, encoding='utf-8'))
    if not isinstance(recs, list):
        raise ValueError(f'{l3_path} 顶层不是 list')

    out_json = l3_path.parent / 'vqa_l3_gpt.json'
    metrics_path = l3_path.parent / 'vqa_l3_gpt_metrics.json'

    n_total = len(recs)
    n_correct = 0
    n_ok = 0
    t0 = time.time()
    out_recs = []

    for i, r in enumerate(recs):
        question = r.get('question', '')
        gt = r.get('gt', '')
        pred = r.get('pred', '')
        image_id = r.get('image_id', '')
        # 拼图路径
        img_path = str(vrs_img_dir / image_id) if (vrs_img_dir and image_id) else None

        result = judge_one(question, gt, pred, img_path, model, api_key, base_url, max_new, timeout_s)

        if result['ok']:
            n_ok += 1
            n_correct += result['correct_l3']

        out_recs.append({
            **r,
            'gpt_correct_l3':  result['correct_l3'],
            'gpt_response':     result['response_text'][:200],  # 截断
            'gpt_ok':           result['ok'],
            'gpt_error':        result['error'][:200] if result['error'] else '',
            'gpt_http_status':  result['http_status'],
            'gpt_model':        model,
        })

        # 进度
        if (i + 1) % 10 == 0 or i == n_total - 1:
            elapsed = time.time() - t0
            speed = (i + 1) / max(elapsed, 0.001)
            eta = (n_total - i - 1) / max(speed, 0.001)
            print(f'  [{i+1}/{n_total}] correct={n_correct}/{i+1}={n_correct/(i+1)*100:.2f}%  '
                  f'ok={n_ok}  speed={speed:.2f}条/s  ETA={eta:.0f}s', flush=True)

    # 写结果
    summary = {
        'total_l3':   n_total,
        'correct_l3': n_correct,
        'acc_l3':     round(n_correct / n_total, 4) if n_total else 0,
        'ok_count':   n_ok,
        'model':      model,
        'metric':     'glm5_semantic_match',
        'elapsed_s':  round(time.time() - t0, 1),
    }
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(out_recs, f, indent=2, ensure_ascii=False)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'method': 'glm5_semantic_match', 'prompt_tpl': VRS_L3_PROMPT_TPL},
                  f, indent=2, ensure_ascii=False)

    print(f'\n[完成] {l3_path.parent.name}: acc_l3={summary["acc_l3"]:.4f} ({n_correct}/{n_total})  '
          f'ok={n_ok}  耗时={summary["elapsed_s"]}s', flush=True)
    return summary


# ─── 批量：扫目录 ──────────────────────────────────────────
def judge_batch_dir(batch_dir: Path, model: str, api_key: str, base_url: str,
                    max_new: int, timeout_s: int, vrs_img_dir: Path,
                    force: bool = False) -> dict:
    """扫 batch_dir/**/vqa_l3_raw.json，跳过已存在 vqa_l3_gpt.json 的（除非 force）"""
    l3_files = sorted(batch_dir.glob('**/vqa_l3_raw.json'))
    if not l3_files:
        print(f'[GPT-L3] 在 {batch_dir} 下没找到 vqa_l3_raw.json')
        return {}

    print(f'[GPT-L3] 发现 {len(l3_files)} 个 vqa_l3_raw.json')
    print(f'[GPT-L3] model: {model}, base_url: {base_url}')
    print(f'[GPT-L3] api_key: {"已设置 (" + api_key[:8] + "...)" if api_key else "未设置!"}')

    if not api_key:
        raise ValueError('DASHSCOPE_API_KEY 未设置，请 export DASHSCOPE_API_KEY=sk-...')

    t0 = time.time()
    total_l3, total_correct = 0, 0
    results = {}
    for i, l3f in enumerate(l3_files, 1):
        run_name = l3f.parent.name
        out_gpt = l3f.parent / 'vqa_l3_gpt.json'
        if out_gpt.exists() and not force:
            # 读已存在的 metrics
            met_p = l3f.parent / 'vqa_l3_gpt_metrics.json'
            if met_p.exists():
                m = json.load(open(met_p))
                s = m.get('summary', {})
                if 'total_l3' in s:
                    total_l3 += s['total_l3']
                    total_correct += s['correct_l3']
                results[run_name] = s
            print(f'[{i}/{len(l3_files)}] 跳过 {run_name} (vqa_l3_gpt.json 已存在)', flush=True)
            continue

        try:
            summary = judge_file(l3f, model, api_key, base_url, max_new, timeout_s, vrs_img_dir)
            results[run_name] = summary
            total_l3 += summary['total_l3']
            total_correct += summary['correct_l3']
        except Exception as e:
            print(f'[{i}/{len(l3_files)}] 失败 {run_name}: {e}', flush=True)
            results[run_name] = {'error': str(e)}

    overall = {
        'total_l3':     total_l3,
        'correct_l3':   total_correct,
        'acc_l3':       round(total_correct / total_l3, 4) if total_l3 else 0,
        'elapsed_s':    round(time.time() - t0, 1),
        'model':        model,
        'metric':       'glm5_semantic_match',
    }
    print(f'\n[GPT-L3] 全部完成: 累计 acc_l3={overall["acc_l3"]:.4f} ({total_correct}/{total_l3}), '
          f'耗时 {overall["elapsed_s"]}s')

    # 写总览
    overview_path = batch_dir / 'vqa_l3_gpt_overview.json'
    with open(overview_path, 'w', encoding='utf-8') as f:
        json.dump({'overall': overall, 'per_run': results, 'method': 'glm5_semantic_match'},
                  f, indent=2, ensure_ascii=False)
    print(f'[GPT-L3] 总览: {overview_path}')

    return overall


# ─── 入口 ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='VRS-VQA L3 评测 — 阿里云百炼 GLM-5 (OpenAI 兼容)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('l3_json', nargs='?', help='单个 vqa_l3_raw.json 路径')
    parser.add_argument('--batch-dir', type=Path, default=None,
                        help='批量模式：扫该目录下所有 vqa_l3_raw.json')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help=f'模型名（默认: {DEFAULT_MODEL}，可换 glm-4.5 / qwen3-max 等）')
    parser.add_argument('--base-url', type=str, default=DASHSCOPE_BASE_URL,
                        help=f'API base url（默认: {DASHSCOPE_BASE_URL}）')
    parser.add_argument('--max-new', type=int, default=DEFAULT_MAX_NEW,
                        help=f'生成 token 上限（默认: {DEFAULT_MAX_NEW}，L3 判定只需 1 token）')
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT,
                        help=f'请求超时秒（默认: {DEFAULT_TIMEOUT}）')
    parser.add_argument('--vrs-img-dir', type=Path, default=Path('/home/admin1/models/vrsbench_images/Images_val'),
                        help='VRS 图像目录（含 image_id）')
    parser.add_argument('--force', action='store_true', help='覆盖已存在的 vqa_l3_gpt.json')
    args = parser.parse_args()

    if not DASHSCOPE_API_KEY:
        print('[ERROR] 环境变量 DASHSCOPE_API_KEY 未设置')
        print('        请先: export DASHSCOPE_API_KEY=sk-...')
        sys.exit(1)

    if args.l3_json:
        judge_file(Path(args.l3_json), args.model, DASHSCOPE_API_KEY, args.base_url,
                   args.max_new, args.timeout, args.vrs_img_dir)
    elif args.batch_dir:
        judge_batch_dir(args.batch_dir, args.model, DASHSCOPE_API_KEY, args.base_url,
                       args.max_new, args.timeout, args.vrs_img_dir, force=args.force)
    else:
        print('[ERROR] 必须传 l3_json 或 --batch-dir')
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
