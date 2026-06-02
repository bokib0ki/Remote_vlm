"""
VRS-VQA L3 语义评测脚本
=======================
读取 eval_select.py 输出的 vqa_l3_raw.json，
对需要 L3 评测的记录调用 Qwen3.7（或 GPT-4o-mini）语义匹配，
输出最终 acc（含 L3）。

API Key 填写位置：
    方法1: 环境变量 DASHSCOPE_API_KEY
           Linux/macOS: export DASHSCOPE_API_KEY=你的阿里云百炼key
           Windows WSL:  export DASHSCOPE_API_KEY=你的阿里云百炼key
           持久化写入 ~/.bashrc: echo 'export DASHSCOPE_API_KEY=...' >> ~/.bashrc
    方法2: 直接在本文件第 16 行填入: QWEN_API_KEY = '你的key'
    方法3: 百炼 API Key 也可以从 DASHSCOPE_API_KEY 环境变量读取
"""

import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

# =============================================================================
# API Key 配置（修改这里填入你的 key）
# =============================================================================
# 方法1: 直接在这里填入 key（不推荐共享给他人）
QWEN_API_KEY = os.environ.get('DASHSCOPE_API_KEY', '')
# 方法2: 设为空字符串，使用环境变量 DASHSCOPE_API_KEY
# QWEN_API_KEY = ''

# 模型选择: 'qwen3.7'（阿里百炼） 或 'gpt-4o-mini'（OpenAI）
MODEL = 'qwen3.7'

# OpenAI GPT-4o-mini 配置（如果 MODEL = 'gpt-4o-mini' 则使用这个）
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
OPENAI_BASE_URL = 'https://api.openai.com/v1'

# =============================================================================

def _guess_mime(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    return mt or 'image/png'


def _read_image_bytes(path: Path) -> tuple[bytes, str]:
    if path.exists():
        b = path.read_bytes()
        return b, _guess_mime(path.name)
    return b'', 'image/png'


def _data_url(img_bytes: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(img_bytes).decode('ascii')}"


def _http_post_json(url: str, headers: dict, body: dict, timeout_s: int = 120) -> dict:
    data = json.dumps(body).encode('utf-8')
    req = urllib.request.Request(url, data=data, method='POST')
    for k, v in headers.items():
        req.add_header(k, v)
    req.add_header('Content-Type', 'application/json')
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode('utf-8', errors='replace')
                return json.loads(raw)
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            if attempt < 2:
                wait = 2 ** attempt
                time.sleep(wait)
                continue
            return {'_http_error': str(e)}


def _call_qwen3_api(prompt: str, images: list = None, max_tokens: int = 10, timeout_s: int = 120) -> str:
    """调用阿里云百炼 Qwen3.7 API 进行语义匹配判断。"""
    api_key = QWEN_API_KEY
    if not api_key:
        raise RuntimeError('[VQA_V3] 未设置 DASHSCOPE_API_KEY 环境变量，L3 评测无法进行。'
                           '请先: export DASHSCOPE_API_KEY=你的阿里云百炼key')

    base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    model = 'qwen3.7'
    
    content = []
    if images:
        for img_path, mime in images:
            b = _read_image_bytes(Path(img_path))[0] if isinstance(img_path, str) else img_path
            if b:
                content.append({'type': 'image_url', 'image_url': {'url': _data_url(b, mime)}})
    content.append({'type': 'text', 'text': prompt})
    
    url = base_url.rstrip('/') + '/chat/completions'
    body = {
        'model': model,
        'messages': [{'role': 'user', 'content': content}],
        'max_tokens': max_tokens,
        'temperature': 0,
    }
    headers = {'Authorization': f'Bearer {api_key}'}
    
    resp = _http_post_json(url, headers=headers, body=body, timeout_s=timeout_s)
    
    text = ''
    if isinstance(resp, dict) and 'choices' in resp:
        try:
            choice0 = (resp.get('choices') or [{}])[0]
            msg = choice0.get('message') or {}
            c = msg.get('content', '')
            if isinstance(c, list):
                text = ''.join([p.get('text', '') for p in c if isinstance(p, dict)]).strip()
            else:
                text = str(c or '').strip()
        except Exception:
            text = ''
    elif isinstance(resp, dict) and 'output' in resp:
        try:
            out = resp.get('output', [])
            for item in out:
                if item.get('type') == 'message':
                    for c in item.get('content', []):
                        if c.get('type') == 'output_text':
                            text += c.get('text', '') or ''
            text = text.strip()
        except Exception:
            text = ''
    
    return text


def _call_openai_api(prompt: str, images: list = None, max_tokens: int = 10, timeout_s: int = 120) -> str:
    """调用 OpenAI GPT-4o-mini API 进行语义匹配判断。"""
    api_key = OPENAI_API_KEY
    if not api_key:
        raise RuntimeError('[VQA_V3] 未设置 OPENAI_API_KEY 环境变量，L3 评测无法进行。'
                           '请先: export OPENAI_API_KEY=你的OpenAI key')

    base_url = OPENAI_BASE_URL
    
    content = []
    if images:
        for img_path, mime in images:
            b = _read_image_bytes(Path(img_path))[0] if isinstance(img_path, str) else img_path
            if b:
                content.append({'type': 'image_url', 'image_url': {'url': _data_url(b, mime)}})
    content.append({'type': 'text', 'text': prompt})
    
    url = base_url.rstrip('/') + '/chat/completions'
    body = {
        'model': 'gpt-4o-mini',
        'messages': [{'role': 'user', 'content': content}],
        'max_tokens': max_tokens,
        'temperature': 0,
    }
    headers = {'Authorization': f'Bearer {api_key}'}
    
    resp = _http_post_json(url, headers=headers, body=body, timeout_s=timeout_s)
    
    text = ''
    if isinstance(resp, dict) and 'choices' in resp:
        try:
            choice0 = (resp.get('choices') or [{}])[0]
            msg = choice0.get('message') or {}
            c = msg.get('content', '')
            text = str(c or '').strip()
        except Exception:
            text = ''
    
    return text


def vqa_judge_l3(question: str, gt: str, pred: str, image_path: str = None) -> int:
    """
    L3 评测: 语义匹配
    返回: 1（匹配）或 0（不匹配）
    """
    prompt = f"""Question: {question}
Ground Truth Answer: {gt}
Predicted Answer: {pred}
Does the predicted answer match the ground truth? 
Consider synonyms as matches (e.g., football/soccer, building/rooftop, pond/swimming pool).
Answer with only "1" for match or "0" for no match. Do not explain."""

    images = []
    if image_path and Path(image_path).exists():
        mime = _guess_mime(str(image_path))
        images.append((str(image_path), mime))
    
    if MODEL == 'gpt-4o-mini':
        result = _call_openai_api(prompt, images=images if images else None, max_tokens=10, timeout_s=60)
    else:
        result = _call_qwen3_api(prompt, images=images if images else None, max_tokens=10, timeout_s=60)
    
    result_clean = result.strip().upper()
    return 1 if '1' in result_clean else 0


def run_l3_eval(l3_json_path: str, model: str = 'qwen3.7'):
    """
    读取 vqa_l3_raw.json，评测 L3，返回结果。
    
    Args:
        l3_json_path: eval_select.py 输出的 vqa_l3_raw.json 路径
        model: 'qwen3.7' 或 'gpt-4o-mini'
    
    Returns:
        dict: 包含 L3 评测结果的 dict
    """
    global MODEL
    MODEL = model
    
    l3_path = Path(l3_json_path)
    if not l3_path.exists():
        raise FileNotFoundError(f'[VQA_V3] 文件不存在: {l3_json_path}')
    
    with open(l3_path, 'r', encoding='utf-8') as f:
        l3_records = json.load(f)
    
    if not l3_records:
        print('[VQA_V3] 无 L3 待评测记录')
        return {'correct_l3': 0, 'total_l3': 0, 'acc_l3': 0, 'results': []}
    
    print(f'[VQA_V3] 共 {len(l3_records)} 条 L3 待评测，使用模型: {MODEL}')
    
    correct_l3 = 0
    results = []
    
    for i, rec in enumerate(l3_records):
        gt = rec.get('gt', '')
        pred = rec.get('pred', '')
        question = rec.get('question', '')
        image_path = rec.get('image_path', '')
        
        l3_ok = vqa_judge_l3(question, gt, pred, image_path=image_path)
        correct_l3 += l3_ok
        
        results.append({
            'image_id': rec.get('image_id', ''),
            'gt': gt,
            'pred': pred,
            'question': question,
            'correct_l3': l3_ok,
            'image_path': image_path,
        })
        
        if (i + 1) % 10 == 0:
            print(f'  [{i+1}/{len(l3_records)}] L3_acc={correct_l3}/{i+1}')
    
    total_l3 = len(l3_records)
    print(f'[VQA_V3] L3 完成: correct_l3={correct_l3}/{total_l3} acc_l3={correct_l3/max(total_l3,1):.4f}')
    
    return {
        'correct_l3': correct_l3,
        'total_l3': total_l3,
        'acc_l3': round(correct_l3 / max(total_l3, 1), 4),
        'results': results,
    }


def merge_results(vqa_raw_path: str, l3_json_path: str, output_path: str):
    """
    合并 L1/L2 结果（来自 eval_select.py 的 vqa_raw.json）
    和 L3 结果（来自 vqa_l3_raw.json 的评测），
    输出完整的含三级评测结果的 JSON。
    """
    with open(vqa_raw_path, 'r', encoding='utf-8') as f:
        vqa_records = json.load(f)
    
    with open(l3_json_path, 'r', encoding='utf-8') as f:
        l3_results = json.load(f)
    
    # 建立 L3 结果索引
    l3_map = {r['image_id']: r for r in l3_results}
    
    correct_l1, correct_l2, correct_l3 = 0, 0, 0
    total = len(vqa_records)
    
    for rec in vqa_records:
        level = rec.get('judge_level', '')
        if level == 'L1':
            correct_l1 += 1
        elif level == 'L2':
            correct_l2 += 1
        elif level == 'L3':
            img_id = rec.get('image_id', '')
            l3_res = l3_map.get(img_id, {})
            l3_ok = l3_res.get('correct_l3', 0)
            correct_l3 += l3_ok
            rec['correct'] = l3_ok
            rec['judge_method'] = 'semantic'
    
    correct_total = correct_l1 + correct_l2 + correct_l3
    acc_total = correct_total / max(total, 1)
    
    merged = {
        'summary': {
            'total': total,
            'correct_total': correct_total,
            'acc_total': round(acc_total, 4),
            'correct_l1': correct_l1,
            'correct_l2': correct_l2,
            'correct_l3': correct_l3,
            'acc_l1': round(correct_l1 / max(total, 1), 4),
            'acc_l2': round(correct_l2 / max(total, 1), 4),
            'acc_l3': round(correct_l3 / max(total, 1), 4),
        },
        'records': vqa_records,
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    
    print(f'[VQA_V3] 合并完成: acc_total={acc_total:.4f} (L1={correct_l1}/L2={correct_l2}/L3={correct_l3})')
    print(f'[VQA_V3] 结果已保存: {output_path}')
    
    return merged


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='VRS-VQA L3 语义评测')
    parser.add_argument('l3_json', nargs='?', 
                        help='vqa_l3_raw.json 路径（eval_select.py 输出）')
    parser.add_argument('--model', '-m', default='qwen3.7',
                        choices=['qwen3.7', 'gpt-4o-mini'],
                        help='L3 语义评测使用的模型（默认 qwen3.7）')
    parser.add_argument('--merge', action='store_true',
                        help='合并 L1/L2 和 L3 结果，输出完整 JSON')
    parser.add_argument('--vqa_raw', 
                        help='eval_select.py 输出的 vqa_raw.json 路径（配合 --merge 使用）')
    parser.add_argument('--output', '-o', 
                        help='合并后的输出 JSON 路径（配合 --merge 使用）')
    args = parser.parse_args()
    
    if not args.l3_json:
        # 自动查找最新的 vqa_l3_raw.json
        import glob
        candidates = sorted(glob.glob('results/**/vqa_l3_raw.json', recursive=True), key=os.path.getmtime, reverse=True)
        if candidates:
            args.l3_json = candidates[0]
            print(f'[VQA_V3] 自动找到: {args.l3_json}')
        else:
            print('[VQA_V3] 请提供 vqa_l3_raw.json 路径，或确保 results/ 目录下有该文件')
            parser.print_help()
            exit(1)
    
    l3_res = run_l3_eval(args.l3_json, model=args.model)
    
    if args.merge and args.vqa_raw and args.output:
        merge_results(args.vqa_raw, args.l3_json, args.output)