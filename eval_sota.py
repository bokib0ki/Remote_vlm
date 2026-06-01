#!/usr/bin/env python3
import argparse
import base64
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from config import (
    RAW_DIR_ROOT,
    RES_DIR,
    ROOT,
    TEST_IMGS,
    VRS_IMG_DIR,
    LEVIR_DIR,
    MME_RS_DIR,
    CAPTION_PROMPT,
    LEVIR_PROMPT,
    VRS_VQA_PROMPT_TPL,
    MME_PROMPT_TPL,
    XLRS_PROMPT_TPL,
)
try:
    from model_store import save_run_from_raw_records as _save_run_from_raw_records  # type: ignore
except Exception:
    _save_run_from_raw_records = None


def save_run_from_raw_records(model_name: str, run_meta: dict, raw_records: list[dict]):
    if _save_run_from_raw_records is None:
        return
    return _save_run_from_raw_records(model_name, run_meta, raw_records)

try:
    from metrics import compute_caption_scores as _compute_caption_scores  # type: ignore
except Exception:
    _compute_caption_scores = None

try:
    from inference import extract_letter as _extract_letter  # type: ignore
    from inference import strip_thinking as _strip_thinking  # type: ignore
except Exception:
    _extract_letter = None
    _strip_thinking = None


def strip_thinking(text: str) -> str:
    if _strip_thinking is not None:
        return _strip_thinking(text)
    if not text:
        return ''
    s = text
    lower = s.lower()
    for close_tag in ('</think>', '</thinking>'):
        idx = lower.rfind(close_tag)
        if idx != -1:
            s = s[idx + len(close_tag):]
            lower = s.lower()
    s = re.sub(r'<think>.*?</think>\s*\n?\s*', '', s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r'<thinking>.*?</thinking>\s*\n?\s*', '', s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r'</?think(?:ing)?>', '', s, flags=re.IGNORECASE)
    return s.strip()


def extract_letter(text: str, choices: str = 'ABCDE') -> str:
    if _extract_letter is not None:
        return _extract_letter(text, choices=choices)
    s = text.strip().upper()
    for pat in [
        r'(?:ANSWER|OPTION)\s+(?:IS|:)\s*[(]?([' + choices + r'])[)]?',
        r'CORRECT\s+(?:ANSWER|OPTION)\s+IS\s+[(]?([' + choices + r'])[)]?',
        r'THE\s+BEST\s+ANSWER\s+IS\s+[(]?([' + choices + r'])[)]?',
        r'IS\s+CORRECT\s*[.:]?\s*[(]?([' + choices + r'])[)]?',
    ]:
        m = re.findall(pat, s)
        if m:
            return m[-1]
    parenthesized = re.findall(r'\(' + choices + r'\)', s)
    if parenthesized:
        return parenthesized[-1].strip('()')
    letters = re.findall(
        r'(?:^|[\s(])([' + choices + r'])(?:\)|\]|\}|\.|,|\s|$)',
        s,
    )
    if letters:
        return letters[-1]
    return ""

try:
    from config_sota import (
        OPENAI_BASE_URL as _CFG_OPENAI_BASE_URL,
        OPENAI_API_KEY as _CFG_OPENAI_API_KEY,
        GEMINI_BASE_URL as _CFG_GEMINI_BASE_URL,
        GEMINI_API_KEY as _CFG_GEMINI_API_KEY,
        DEFAULT_PROVIDER as _CFG_DEFAULT_PROVIDER,
        DEFAULT_MODEL as _CFG_DEFAULT_MODEL,
        DEFAULT_MAX_NEW as _CFG_DEFAULT_MAX_NEW,
        DEFAULT_TIMEOUT as _CFG_DEFAULT_TIMEOUT,
    )
except Exception:
    _CFG_OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    _CFG_OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
    _CFG_GEMINI_BASE_URL = os.environ.get('GEMINI_BASE_URL', 'https://generativelanguage.googleapis.com')
    _CFG_GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
    _CFG_DEFAULT_PROVIDER = ''
    _CFG_DEFAULT_MODEL = ''
    _CFG_DEFAULT_MAX_NEW = 512
    _CFG_DEFAULT_TIMEOUT = 180


print = lambda *a, **kw: __builtins__.print(*a, **kw, flush=True)


def _now_iso():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def compute_caption_scores_safe(refs: dict, hyps: dict) -> dict:
    if not refs or not hyps:
        return {'bleu1': 0.0, 'bleu2': 0.0, 'bleu3': 0.0, 'bleu4': 0.0, 'rouge_l': 0.0, 'cider': 0.0}
    if _compute_caption_scores is None:
        return {'bleu1': 0.0, 'bleu2': 0.0, 'bleu3': 0.0, 'bleu4': 0.0, 'rouge_l': 0.0, 'cider': 0.0}
    return _compute_caption_scores(refs, hyps)


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
    }


def _read_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _load_json(path: str | Path):
    return _read_json(Path(path))


def _as_list(obj):
    if obj is None:
        return []
    return obj if isinstance(obj, list) else [obj]


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
    for rec in records:
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


def _guess_mime(path: str):
    mt, _ = mimetypes.guess_type(path)
    return mt or 'image/png'


def _read_image_bytes(path: Path) -> tuple[bytes, str]:
    if path.exists():
        b = path.read_bytes()
        return b, _guess_mime(path.name)
    if TEST_IMGS:
        b = Path(TEST_IMGS[0]).read_bytes()
        return b, _guess_mime(str(TEST_IMGS[0]))
    return b'', 'image/png'


def _data_url(img_bytes: bytes, mime: str):
    return f"data:{mime};base64,{base64.b64encode(img_bytes).decode('ascii')}"


def _http_post_json(url: str, headers: dict, body: dict, timeout_s: int = 120):
    data = json.dumps(body).encode('utf-8')
    req = urllib.request.Request(url, data=data, method='POST')
    for k, v in headers.items():
        req.add_header(k, v)
    req.add_header('Content-Type', 'application/json')
    for attempt in range(4):  # 0=first try, 1-3=retries
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode('utf-8', errors='replace')
                return json.loads(raw)
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            if attempt < 3:
                wait = 2 ** attempt
                time.sleep(wait)
                continue
            raw = b''
            code = 0
            if isinstance(e, urllib.error.HTTPError):
                raw = e.read().decode('utf-8', errors='replace')
                code = e.code
            try:
                j = json.loads(raw) if raw else {}
            except Exception:
                j = {'error': {'message': raw}}
            if isinstance(j, dict):
                j['_http_status'] = int(code or 0)
            return j


def call_openai_compatible(model: str, prompt: str, images: list[tuple[bytes, str]], max_new: int, api_key: str, base_url: str, timeout_s: int, extra_body: dict | None = None):
    def _call_responses():
        content = []
        for b, mime in images:
            if not b:
                continue
            content.append({'type': 'input_image', 'image_url': _data_url(b, mime)})
        content.append({'type': 'input_text', 'text': prompt})

        url = base_url.rstrip('/') + '/responses'
        body = {
            'model': model,
            'input': [{'role': 'user', 'content': content}],
            'max_output_tokens': max_new,
            'temperature': 0,
        }
        if extra_body:
            body['extra_body'] = extra_body
        headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
        return _http_post_json(url, headers=headers, body=body, timeout_s=timeout_s)

    def _call_chat_completions():
        content = []
        for b, mime in images:
            if not b:
                continue
            content.append({'type': 'image_url', 'image_url': {'url': _data_url(b, mime)}})
        content.append({'type': 'text', 'text': prompt})

        url = base_url.rstrip('/') + '/chat/completions'
        body = {
            'model': model,
            'messages': [{'role': 'user', 'content': content}],
            'max_tokens': max_new,
            'temperature': 0,
        }
        headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
        return _http_post_json(url, headers=headers, body=body, timeout_s=timeout_s)

    resp = _call_responses()
    if isinstance(resp, dict) and resp.get('_http_status') in (404, 405):
        resp = _call_chat_completions()

    text = ''
    reasoning_text = ''
    tokens = None
    thinking_enabled = extra_body and extra_body.get('enable_thinking')
    if isinstance(resp, dict) and 'output' in resp:
        try:
            out = resp.get('output', [])
            for item in out:
                item_type = item.get('type')
                if item_type == 'reasoning':
                    # summary is at top level
                    for s in item.get('summary', []):
                        if isinstance(s, dict) and s.get('type') == 'summary_text':
                            reasoning_text += s.get('text', '') or ''
                    # also try nested content
                    for c in item.get('content', []):
                        if c.get('type') == 'summary_text':
                            reasoning_text += c.get('text', '') or ''
                elif item_type == 'message':
                    for c in item.get('content', []):
                        if c.get('type') == 'output_text':
                            text += c.get('text', '') or ''
            text = text.strip()
            if not text and reasoning_text and thinking_enabled:
                text = _extract_answer_from_reasoning(reasoning_text)
        except Exception as exc:
            text = ''
            reasoning_text = ''
            resp['_parse_error'] = str(exc)
        usage = resp.get('usage') or {}
        tokens = usage.get('output_tokens')
        if tokens is None:
            tokens = usage.get('completion_tokens')
    elif isinstance(resp, dict) and 'choices' in resp:
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
        usage = resp.get('usage') or {}
        tokens = usage.get('completion_tokens')
        if tokens is None:
            tokens = usage.get('output_tokens')
    return text, tokens, resp, reasoning_text


def _extract_answer_from_reasoning(reasoning: str) -> str:
    """Extract final short answer from reasoning text (for thinking mode when no output_text)."""
    if not reasoning:
        return ''
    # Try to find "The answer is: XXX" or "Answer: XXX" near the end
    for pattern in [
        r'(?:final\s+)?answer\s*(?:is)?\s*[:=]\s*["\']?([A-Za-z0-9]+)["\']?',
        r'(?:final\s+)?answer\s+(?:is\s+)?["\']?([A-Za-z0-9 ]+)["\']?',
        r'(?:so|the)\s+(?:answer|count|result)\s*(?:is)?\s*[:=]\s*["\']?([A-Za-z0-9]+)["\']?',
    ]:
        matches = list(re.finditer(pattern, reasoning, re.IGNORECASE))
        if matches:
            # Use the last match (most recent/authoritative)
            last = matches[-1]
            ans = last.group(1).strip().strip('"\'')
            if ans:
                return ans
    # Fallback: take the last 100 chars, strip thinking tags, return short phrase
    chunk = reasoning[-800:].strip()
    chunk = re.sub(r'<[^>]+>', '', chunk)
    lines = [l.strip() for l in chunk.split('\n') if l.strip()]
    if lines:
        last_line = lines[-1]
        # If last line is short and looks like an answer, use it
        if 1 <= len(last_line) <= 60:
            return last_line
    # Last resort: return last 200 chars stripped
    return reasoning[-200:].strip()


def call_gemini(model: str, prompt: str, images: list[tuple[bytes, str]], max_new: int, api_key: str, base_url: str, timeout_s: int):
    parts = []
    for b, mime in images:
        if not b:
            continue
        parts.append({'inline_data': {'mime_type': mime, 'data': base64.b64encode(b).decode('ascii')}})
    parts.append({'text': prompt})
    base = (base_url or 'https://generativelanguage.googleapis.com').rstrip('/')
    url = f"{base}/v1beta/models/{model}:generateContent"
    headers = {}
    if base.endswith('googleapis.com'):
        url = url + f"?key={api_key}"
    else:
        if api_key:
            headers = {'Authorization': f'Bearer {api_key}'}
    body = {
        'contents': [{'role': 'user', 'parts': parts}],
        'generationConfig': {'maxOutputTokens': max_new, 'temperature': 0},
    }
    resp = _http_post_json(url, headers=headers, body=body, timeout_s=timeout_s)
    if (not base.endswith('googleapis.com')) and isinstance(resp, dict):
        err = resp.get('error') or {}
        if isinstance(err, dict) and err.get('code') == 'missing_api_key' and api_key:
            resp = _http_post_json(url, headers={'x-api-key': api_key}, body=body, timeout_s=timeout_s)
    text = ''
    try:
        cand = (resp.get('candidates') or [{}])[0]
        c = cand.get('content', {})
        text = ''.join([p.get('text', '') for p in c.get('parts', []) if isinstance(p, dict)]).strip()
    except Exception:
        text = ''

    usage = resp.get('usageMetadata') or {}
    tokens = usage.get('candidatesTokenCount')
    return text, tokens, resp, ''


def _sanitize_model_tag(s: str):
    s = s.strip().replace(' ', '_')
    s = s.replace('/', '__').replace('\\', '__').replace(':', '_')
    return s


def _bench_filter(benches: dict, allow: set[str] | None):
    if not allow:
        return benches
    return {k: v for k, v in benches.items() if k in allow}


def run_eval(provider: str, model: str, select_doc: dict, max_new: int, benches_allow: set[str] | None, timeout_s: int, api_key: str, base_url: str | None, save_raw_json: bool, thinking: bool = False, save_reasoning: bool = False):
    select_name = Path(select_doc.get('name') or Path(select_doc.get('_path', 'selection')).stem).stem
    run_started_at = _now_iso()
    run_suffix = time.strftime('%Y%m%d_%H%M%S', time.gmtime())
    model_tag = _sanitize_model_tag(f'{provider}__{model}')
    config_tag = f'{model_tag}_{select_name}_{run_suffix}'

    print(f"\n{'='*60}")
    print(f"  {config_tag}")
    print(f"{'='*60}")

    benches = _bench_filter(select_doc['benchmarks'], benches_allow)

    row = {
        'config': config_tag,
        'provider': provider,
        'model': model,
        'max_new': max_new,
        'selection': {'path': select_doc.get('_path'), 'name': select_doc.get('name')},
        'benchmarks': {},
    }
    raw_records = []

    extra_body = {'enable_thinking': True} if thinking else None

    def _call(prompt: str, images: list[tuple[bytes, str]]):
        t0 = time.time()
        if provider == 'gemini':
            out_text, out_tokens, resp, reasoning = call_gemini(model, prompt, images, max_new=max_new, api_key=api_key, base_url=base_url or _CFG_GEMINI_BASE_URL, timeout_s=timeout_s)
        else:
            out_text, out_tokens, resp, reasoning = call_openai_compatible(model, prompt, images, max_new=max_new, api_key=api_key, base_url=base_url, timeout_s=timeout_s, extra_body=extra_body)
        dt = time.time() - t0
        return out_text, out_tokens, dt, resp, reasoning

    if 'vrs_caption' in benches:
        print(f"\n  VRS-Caption...")
        t0 = time.time()
        anns = load_selected_benchmark('vrs_caption', benches['vrs_caption'])
        refs, hyps = {}, {}
        token_lens, times_s = [], []
        for i, ann in enumerate(anns):
            img_id = ann.get('image_id', ann.get('image', ''))
            b, mime = _read_image_bytes(VRS_IMG_DIR / str(img_id))
            gt = str(ann.get('ground_truth', '') or ann.get('caption', '')).strip()
            if not gt:
                continue
            pred_raw, tokens, dt, resp, reasoning = _call(CAPTION_PROMPT, [(b, mime)])
            pred = strip_thinking(pred_raw)
            token_lens.append(int(tokens or 0))
            times_s.append(dt)
            refs[str(i)] = [gt]
            hyps[str(i)] = [pred]
            rec = {
                'benchmark': 'VRS-Caption',
                '_idx': ann.get('question_id', ann.get('_original_index', i)),
                'image_id': img_id,
                'gt': gt,
                'pred': pred,
                'pred_raw': pred_raw,
                'time_s': round(dt, 4),
                'tokens': tokens,
                'provider': provider,
                'model': model,
            }
            if save_reasoning:
                rec['reasoning'] = reasoning
            if save_raw_json:
                rec['api_response'] = resp
            raw_records.append(rec)
            print(f"    [{i+1}/{len(anns)}] qid={rec['_idx']} gt={gt[:30]!r} pred={pred[:50]!r} dt={dt:.1f}s")
        scores = compute_caption_scores_safe(refs, hyps)
        row['benchmarks']['vrs_caption'] = {**scores, 'n': len(anns), 'perf': perf_stats(times_s, token_lens)}
        print(f"    done ({time.time()-t0:.0f}s) n={len(anns)} bleu4={scores['bleu4']}")

    if 'vrs_vqa' in benches:
        print(f"\n  VRS-VQA...")
        t0 = time.time()
        anns = load_selected_benchmark('vrs_vqa', benches['vrs_vqa'])
        correct, total = 0, 0
        token_lens, times_s = [], []
        vqa_records = []
        for i, ann in enumerate(anns):
            img_id = ann.get('image_id', ann.get('image', ''))
            b, mime = _read_image_bytes(VRS_IMG_DIR / str(img_id))
            q = ann.get('question', '') or ann.get('text', '')
            gt = str(ann.get('ground_truth', '') or ann.get('answer', '')).strip()
            if not gt:
                continue
            prompt = VRS_VQA_PROMPT_TPL.format(question=q)
            pred_raw, tokens, dt, resp, reasoning = _call(prompt, [(b, mime)])
            pred = strip_thinking(pred_raw)
            ok = 1 if gt.lower() in pred.lower() else 0
            correct += ok
            total += 1
            token_lens.append(int(tokens or 0))
            times_s.append(dt)
            rec = {
                'benchmark': 'VRS-VQA',
                '_idx': ann.get('question_id', ann.get('_original_index', i)),
                'image_id': img_id,
                'question': q,
                'gt': gt,
                'pred': pred,
                'pred_raw': pred_raw,
                'correct': ok,
                'time_s': round(dt, 4),
                'tokens': tokens,
                'provider': provider,
                'model': model,
            }
            if save_reasoning:
                rec['reasoning'] = reasoning
            if save_raw_json:
                rec['api_response'] = resp
            vqa_records.append(rec)
            raw_records.append(rec)
            print(f"    [{i+1}/{len(anns)}] qid={rec.get('_idx','?')} gt={gt[:30]!r} pred={pred[:50]!r} correct={ok} dt={dt:.1f}s")
        row['benchmarks']['vrs_vqa'] = {
            'acc_l1': round(correct / max(total, 1), 4),
            'total': total,
            'correct_l1': correct,
            'n': len(anns),
            'perf': perf_stats(times_s, token_lens),
        }
        print(f"    done ({time.time()-t0:.0f}s) n={len(anns)} acc_l1={correct}/{total}")

    if 'mme_rs' in benches:
        print(f"\n  MME-RS...")
        t0 = time.time()
        anns = load_selected_benchmark('mme_rs', benches['mme_rs'])
        correct, total = 0, 0
        token_lens, times_s = [], []
        for i, ann in enumerate(anns):
            img_path_str = ann.get('Image', '') or ann.get('image', '')
            fp = MME_RS_DIR / Path(img_path_str).name if img_path_str else None
            b, mime = _read_image_bytes(fp) if fp else _read_image_bytes(Path(''))
            q = ann.get('Text', '') or ann.get('question', '')
            choices = ann.get('Answer choices', []) or ann.get('choices', [])
            prompt = MME_PROMPT_TPL.format(question=q, choices='\n'.join(choices))
            gt = str(ann.get('Ground truth', '') or ann.get('answer', '')).strip().upper()
            pred_raw, tokens, dt, resp, reasoning = _call(prompt, [(b, mime)])
            pred = extract_letter(strip_thinking(pred_raw))
            ok = 1 if pred == gt else 0
            correct += ok
            total += 1
            token_lens.append(int(tokens or 0))
            times_s.append(dt)
            rec = {
                'benchmark': 'MME-RS',
                '_idx': ann.get('Question_id', ann.get('_original_index', i)),
                'image_id': img_path_str,
                'question': q,
                'gt': gt,
                'pred': pred,
                'pred_raw': pred_raw,
                'correct': ok,
                'time_s': round(dt, 4),
                'tokens': tokens,
                'provider': provider,
                'model': model,
            }
            if save_reasoning:
                rec['reasoning'] = reasoning
            if save_raw_json:
                rec['api_response'] = resp
            raw_records.append(rec)
            print(f"    [{i+1}/{len(anns)}] qid={rec.get('_idx','?')} gt={gt[:30]!r} pred={pred[:50]!r} correct={ok} dt={dt:.1f}s")
        row['benchmarks']['mme_rs'] = {
            'acc': round(correct / max(total, 1), 4),
            'correct': correct,
            'total': total,
            'n': len(anns),
            'perf': perf_stats(times_s, token_lens),
        }
        print(f"    done ({time.time()-t0:.0f}s) n={len(anns)} acc={correct}/{total}")

    if 'levir_cc' in benches:
        print(f"\n  LEVIR-CC...")
        t0 = time.time()
        anns = load_selected_benchmark('levir_cc', benches['levir_cc'])
        refs, hyps = {}, {}
        token_lens, times_s = [], []
        for i, ann in enumerate(anns):
            fname = ann.get('filename') or ann.get('image') or ''
            b1, m1 = _read_image_bytes(LEVIR_DIR / 'A' / fname)
            b2, m2 = _read_image_bytes(LEVIR_DIR / 'B' / fname)
            refs_list = [s.get('raw', '').strip() for s in ann.get('sentences', []) if s.get('raw')]
            pred_raw, tokens, dt, resp, reasoning = _call(LEVIR_PROMPT, [(b1, m1), (b2, m2)])
            pred = strip_thinking(pred_raw)
            token_lens.append(int(tokens or 0))
            times_s.append(dt)
            if refs_list:
                refs[str(i)] = refs_list
                hyps[str(i)] = [pred]
            rec = {
                'benchmark': 'LEVIR-CC',
                '_idx': ann.get('imgid', ann.get('_original_index', i)),
                'image_id': fname,
                'gt': ' || '.join(refs_list),
                'pred': pred,
                'pred_raw': pred_raw,
                'time_s': round(dt, 4),
                'tokens': tokens,
                'provider': provider,
                'model': model,
            }
            if save_reasoning:
                rec['reasoning'] = reasoning
            if save_raw_json:
                rec['api_response'] = resp
            raw_records.append(rec)
            print(f"    [{i+1}/{len(anns)}] img={fname} pred={pred[:50]!r} dt={dt:.1f}s")
        scores = compute_caption_scores_safe(refs, hyps)
        row['benchmarks']['levir_cc'] = {**scores, 'n': len(anns), 'perf': perf_stats(times_s, token_lens)}
        print(f"    done ({time.time()-t0:.0f}s) n={len(anns)} bleu4={scores['bleu4']}")

    if 'xlrs' in benches:
        print(f"\n  XLRS...")
        t0 = time.time()
        anns = load_selected_benchmark('xlrs', benches['xlrs'])
        correct, total = 0, 0
        token_lens, times_s = [], []
        for i, ann in enumerate(anns):
            local_img = ann.get('local_image', '')
            if local_img and Path(local_img).exists():
                b, mime = _read_image_bytes(Path(local_img))
            else:
                b, mime = _read_image_bytes(Path(''))
            q = ann.get('question', '')
            choices = ann.get('multi-choice options', []) or ann.get('choices', [])
            prompt = XLRS_PROMPT_TPL.format(question=q, choices='\n'.join(choices))
            gt = str(ann.get('answer', '')).strip().upper()
            gt_set = set(gt.replace(',', ' ').split()) if gt else set()
            pred_raw, tokens, dt, resp, reasoning = _call(prompt, [(b, mime)])
            letter = extract_letter(strip_thinking(pred_raw), choices='ABCD')
            pred_set = {letter} if letter else set()
            ok = 1 if gt_set and pred_set == gt_set else 0
            correct += ok
            total += 1
            token_lens.append(int(tokens or 0))
            times_s.append(dt)
            rec = {
                'benchmark': 'XLRS',
                '_idx': ann.get('idx', ann.get('_original_index', i)),
                'image_id': ann.get('path', ''),
                'question': q[:120],
                'gt': gt,
                'pred': list(pred_set),
                'pred_raw': pred_raw,
                'time_s': round(dt, 4),
                'tokens': tokens,
                'provider': provider,
                'model': model,
            }
            if save_reasoning:
                rec['reasoning'] = reasoning
            if save_raw_json:
                rec['api_response'] = resp
            raw_records.append(rec)
            print(f"    [{i+1}/{len(anns)}] qid={rec.get('_idx','?')} gt={gt} pred={pred} correct={ok} dt={dt:.1f}s")
        row['benchmarks']['xlrs'] = {
            'acc': round(correct / max(total, 1), 4),
            'correct': correct,
            'total': total,
            'n': len(anns),
            'perf': perf_stats(times_s, token_lens),
        }
        print(f"    done ({time.time()-t0:.0f}s) n={len(anns)} acc={correct}/{total}")

    res_dir = RES_DIR / select_name
    res_dir.mkdir(parents=True, exist_ok=True)
    res_file = res_dir / f'{config_tag}_metrics.json'
    with open(res_file, 'w') as f:
        json.dump(row, f, ensure_ascii=False, indent=2)
    print(f"  Saved metrics: {res_file}")

    raw_dir = RAW_DIR_ROOT / select_name / config_tag
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / 'raw_outputs.json'
    with open(raw_file, 'w') as f:
        json.dump(raw_records, f, ensure_ascii=False, indent=2)
    print(f"  Saved raw: {raw_file} ({len(raw_records)} records)")

    run_ended_at = _now_iso()
    run_meta = {
        'config_tag': config_tag,
        'thinking': None,
        'max_new': max_new,
        'selection': select_name,
        'started_at': run_started_at,
        'ended_at': run_ended_at,
        'run_id': f'{config_tag}_{run_ended_at}',
    }
    save_run_from_raw_records(model_tag, run_meta, raw_records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--select', required=True, help='selection JSON 路径')
    parser.add_argument('--provider', choices=['openai', 'gemini'], default=_CFG_DEFAULT_PROVIDER, required=not bool(_CFG_DEFAULT_PROVIDER))
    parser.add_argument('--model', default=_CFG_DEFAULT_MODEL, required=not bool(_CFG_DEFAULT_MODEL), help='provider 对应的模型名')
    parser.add_argument('--max_new', type=int, default=_CFG_DEFAULT_MAX_NEW)
    parser.add_argument('--bench', default='', help='逗号分隔：vrs_vqa,vrs_caption,mme_rs,levir_cc,xlrs')
    parser.add_argument('--timeout', type=int, default=_CFG_DEFAULT_TIMEOUT)
    parser.add_argument('--save_raw_json', action='store_true')
    parser.add_argument('--openai_base_url', default=_CFG_OPENAI_BASE_URL)
    parser.add_argument('--openai_api_key', default=_CFG_OPENAI_API_KEY)
    parser.add_argument('--gemini_base_url', default=_CFG_GEMINI_BASE_URL)
    parser.add_argument('--gemini_api_key', default=_CFG_GEMINI_API_KEY)
    parser.add_argument('--thinking', action='store_true', help='开启 qwen3.6-plus 思考模式 (enable_thinking)')
    parser.add_argument('--save_reasoning', action='store_true', help='记录原始思考过程到 raw_records')
    args = parser.parse_args()

    select_doc = load_selection(args.select)
    select_doc['_path'] = str(Path(args.select).resolve())

    allow = None
    if args.bench.strip():
        allow = {x.strip() for x in args.bench.split(',') if x.strip()}

    if args.provider == 'openai':
        if not args.openai_api_key:
            raise SystemExit('missing OPENAI_API_KEY (or --openai_api_key)')
        api_key = args.openai_api_key
        base_url = args.openai_base_url
    else:
        if not args.gemini_api_key:
            raise SystemExit('missing GEMINI_API_KEY (or --gemini_api_key)')
        api_key = args.gemini_api_key
        base_url = args.gemini_base_url

    run_eval(
        provider=args.provider,
        model=args.model,
        select_doc=select_doc,
        max_new=args.max_new,
        benches_allow=allow,
        timeout_s=args.timeout,
        api_key=api_key,
        base_url=base_url,
        save_raw_json=args.save_raw_json,
        thinking=args.thinking,
        save_reasoning=args.save_reasoning,
    )
