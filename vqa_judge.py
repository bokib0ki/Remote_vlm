"""
VRSBench VQA 三级评测器
=======================
L1: substring 匹配
L2: yes/no/数字 精确匹配
L3: Qwen3.7 语义匹配（非 L1/L2 时升级）
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

# Qwen3.7 配置（使用阿里云百炼 API）
QWEN3_DASHSCOPE_API_KEY = os.environ.get('DASHSCOPE_API_KEY', '')
QWEN3_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
QWEN3_MODEL = 'qwen3.7'  # 注意：是 qwen3.7 不是 qwen3.6


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


def _call_qwen3_api(prompt: str, images: list = None, max_tokens: int = 512, timeout_s: int = 120) -> str:
    """调用 Qwen3.7 API 进行语义匹配判断。"""
    api_key = QWEN3_DASHSCOPE_API_KEY
    base_url = QWEN3_BASE_URL
    model = QWEN3_MODEL
    
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
    headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
    
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


def vqa_judge_l1(gt: str, pred: str) -> int:
    """
    L1 评测: substring 匹配
    如果 ground_truth (lowercase) in predicted (lowercase) → 1
    """
    return 1 if gt.lower() in pred.lower() else 0


def vqa_judge_l2(gt: str, pred: str) -> Optional[int]:
    """
    L2 评测: yes/no/数字 精确匹配
    如果 gt 是 yes/no 或数字，进行精确匹配
    返回 1（正确）、0（错误）或 None（不适用）
    """
    gt_lower = gt.lower().strip()
    pred_lower = pred.lower().strip()
    
    # yes/no 精确匹配
    if gt_lower in ['yes', 'no']:
        return 1 if gt_lower == pred_lower else 0
    
    # 数字匹配（0-99）
    try:
        gt_num = int(gt_lower)
        pred_num = int(pred_lower) if pred_lower.isdigit() else -1
        return 1 if gt_num == pred_num else 0
    except ValueError:
        pass
    
    return None  # 不适用 L2


def vqa_judge_l3(question: str, gt: str, pred: str, image_path: str = None) -> int:
    """
    L3 评测: Qwen3.7 语义匹配
    用于非 yes/no/数字 的情况，判断 pred 是否语义匹配 gt
    
    VRSBench 源码使用 GPT-4o-mini，这里用 Qwen3.7 替代。
    
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
    
    result = _call_qwen3_api(prompt, images=images if images else None, max_tokens=10, timeout_s=60)
    
    # 提取结果
    result_clean = result.strip().upper()
    if '1' in result_clean:
        return 1
    return 0


def vqa_judge(question: str, gt: str, pred: str, image_path: str = None) -> dict:
    """
    VQA 三级评测（按 VRSBench 官方流程）
    
    1. 先试 L1: substring 匹配
    2. 不通过则试 L2: yes/no/数字 精确匹配
    3. 还不通过则升级到 L3: Qwen3.7 语义匹配
    
    Returns:
        dict: {
            'correct': 1 or 0,
            'level': 'L1' or 'L2' or 'L3',
            'method': 'substring' or 'exact' or 'qwen3_semantic'
        }
    """
    # L1: substring 匹配
    if vqa_judge_l1(gt, pred):
        return {'correct': 1, 'level': 'L1', 'method': 'substring'}
    
    # L2: yes/no/数字 精确匹配
    l2_result = vqa_judge_l2(gt, pred)
    if l2_result is not None:
        return {'correct': l2_result, 'level': 'L2', 'method': 'exact'}
    
    # L3: Qwen3.7 语义匹配
    l3_result = vqa_judge_l3(question, gt, pred, image_path)
    return {'correct': l3_result, 'level': 'L3', 'method': 'qwen3_semantic'}