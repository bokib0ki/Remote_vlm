"""
VQA_L3_bert.py
================
VRS-VQA L3 语义评测 — DeBERTa-xlarge-mnli NLI 头版本

设计:
  - 读 eval_select.py 输出的 vqa_l3_raw.json（L3 pending 记录）
  - 用 microsoft/deberta-xlarge-mnli (DeBERTa-v2-xlarge) 的 NLI 分类头
    输入格式: [GT] [SEP] [Pred]
    输出 3-class logits: 0=entailment, 1=neutral, 2=contradiction
  - correct = entailment_prob >= 0.5（且 contradiction_prob < entailment_prob）
  - 输出:
      1) vqa_l3_nli.json              - 逐条记录 + entailment/neutral/contradiction 概率 + correct_l3
      2) vqa_l3_nli_metrics.json      - {summary: {correct_l3, total_l3, acc_l3, ...}}

为什么用 NLI 头（vs 之前失败的 BERTScore 方案）:
  - BERTScore 用 token-level cosine sim，反义词在 embedding 空间天然接近，
    smaller/larger 在 embedding 空间 cosine=0.99 但语义反义
  - DeBERTa-xlarge-mnli 本身是个 NLI 分类模型，3-class 头（entailment/neutral/contradiction）
    在 MNLI 上 fine-tune 过，天然能识别 entailment vs contradiction
  - 用 NLI 头不依赖阈值调参，逻辑更接近 VQA_V3.py 的 Qwen3.7 思路

用法:
  # 单文件
  python3 VQA_L3_bert.py <vqa_l3_raw.json>

  # 批量
  python3 VQA_L3_bert.py --batch-dir /home/admin1/models/raw_outputs/lever_test2

  # 重跑（覆盖已有 vqa_l3_nli.json）
  python3 VQA_L3_bert.py file.json --force
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
os.environ.setdefault('HF_HOME', '/home/admin1/models/bert_cache')
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')


# ─── 全局常量 ──────────────────────────────────────────────
DEFAULT_MODEL = 'microsoft/deberta-xlarge-mnli'
DEFAULT_THRESHOLD = 0.5  # entailment prob 阈值
DEFAULT_BATCH_SIZE = 32
DEFAULT_DEVICE = 'cuda:0'


# ─── 工具函数 ──────────────────────────────────────────────
def strip_special_tokens(text: str) -> str:
    """清理 pred 文本里常见的特殊 token / 标点"""
    if not text:
        return ''
    text = re.sub(r'<\|[^>]*?\|>', ' ', text)
    text = re.sub(r'</?s>', ' ', text)
    text = re.sub(r'<think>.*?</think>', ' ', text, flags=re.DOTALL)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_pred(pred: str) -> str:
    s = strip_special_tokens(pred)
    s = s.rstrip('.,;:!?\n\t ')
    return s


def load_l3_records(l3_json_path: Path) -> List[dict]:
    if not l3_json_path.exists():
        raise FileNotFoundError(f'文件不存在: {l3_json_path}')
    with open(l3_json_path, encoding='utf-8') as f:
        recs = json.load(f)
    if not isinstance(recs, list):
        raise ValueError(f'{l3_json_path} 顶层不是 list（实际: {type(recs).__name__}）')
    return recs


def prepare_pairs(recs: List[dict]) -> Tuple[List[str], List[str]]:
    """准备 (premise=GT, hypothesis=Pred) 对"""
    premises, hypotheses = [], []
    for r in recs:
        gt = r.get('gt', '').strip()
        pred = normalize_pred(r.get('pred', ''))
        if not pred:
            pred = '<empty>'
        if not gt:
            gt = '<empty>'
        # 全部转小写，避免大小写干扰（gt "Grey" vs pred "gray" 应判 entailment）
        premises.append(gt.lower())
        hypotheses.append(pred.lower())
    return premises, hypotheses


# ─── NLI 评测 ──────────────────────────────────────────────
class NliScorer:
    """
    DeBERTa-v2-xlarge NLI 分类头做 entailment 判断

    label 映射（DeBERTa-mnli 标准）:
      0: ENTAILMENT
      1: NEUTRAL
      2: CONTRADICTION
    """

    LABEL_NAMES = ['entailment', 'neutral', 'contradiction']

    def __init__(self, model_path: str, device: str = DEFAULT_DEVICE, batch_size: int = DEFAULT_BATCH_SIZE):
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self._tokenizer = None
        self._model = None
        self._entail_id = None  # 从 config.id2label 自动识别

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        print(f'[L3-NLI] 加载 tokenizer + model: {self.model_path}', flush=True)
        t0 = time.time()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path, torch_dtype=torch.float32
        )
        # 自动识别 entailment label id
        id2label = self._model.config.id2label
        self._entail_id = None
        for lid, lname in id2label.items():
            if 'entail' in lname.lower():
                self._entail_id = int(lid)
                break
        if self._entail_id is None:
            # DeBERTa-mnli 默认是 0
            self._entail_id = 0
        print(f'[L3-NLI]   id2label: {id2label}  →  entailment_id={self._entail_id}')

        self._model.eval()
        self._model.to(self.device)
        print(f'[L3-NLI] 加载完成 ({time.time()-t0:.1f}s)', flush=True)

    def score(self, premises: List[str], hypotheses: List[str]) -> Tuple[List[float], List[float], List[float]]:
        """
        返回 (entailment_probs, neutral_probs, contradiction_probs) 三个 list
        """
        import torch
        self._ensure_loaded()
        if not premises:
            return [], [], []

        n = len(premises)
        all_e, all_neu, all_c = [], [], []
        t0 = time.time()
        for s in range(0, n, self.batch_size):
            e = min(s + self.batch_size, n)
            batch_p = premises[s:e]
            batch_h = hypotheses[s:e]

            enc = self._tokenizer(
                batch_p, batch_h,
                padding=True, truncation=True, max_length=128, return_tensors='pt'
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self._model(**enc, return_dict=True)
            # logits: (B, 3)
            logits = out.logits
            probs = torch.softmax(logits, dim=-1)  # (B, 3)
            all_e.extend(probs[:, self._entail_id].tolist())
            # neutral / contradiction id 找一下
            for lid, lname in self._model.config.id2label.items():
                if 'neutral' in lname.lower():
                    all_neu.extend(probs[:, int(lid)].tolist())
                elif 'contradict' in lname.lower():
                    all_c.extend(probs[:, int(lid)].tolist())

            if (s // self.batch_size) % 5 == 0:
                done = e
                dt = time.time() - t0
                print(f'    [L3-NLI] {done}/{n}  ({done/dt:.0f} 条/s)', flush=True)

        dt = time.time() - t0
        print(f'[L3-NLI] 算完 {n} 条 ({dt:.1f}s, {n/dt:.0f} 条/s)', flush=True)
        return all_e, all_neu, all_c


# ─── 主评测函数 ────────────────────────────────────────────
def run_l3_nli_eval(l3_json_path: Path,
                    model_path: str = DEFAULT_MODEL,
                    threshold: float = DEFAULT_THRESHOLD,
                    batch_size: int = DEFAULT_BATCH_SIZE,
                    device: str = DEFAULT_DEVICE,
                    force: bool = False) -> dict:
    l3_json_path = Path(l3_json_path)
    out_json_path = l3_json_path.parent / 'vqa_l3_nli.json'
    out_metrics_path = l3_json_path.parent / 'vqa_l3_nli_metrics.json'

    if out_json_path.exists() and not force:
        print(f'[L3-NLI] 跳过（已存在）: {out_json_path}', flush=True)
        return {'skipped': True, 'path': str(out_json_path)}

    recs = load_l3_records(l3_json_path)
    n = len(recs)
    if n == 0:
        return {'total_l3': 0, 'acc_l3': 0, 'correct_l3': 0}

    premises, hypotheses = prepare_pairs(recs)
    scorer = NliScorer(model_path=model_path, device=device, batch_size=batch_size)
    E_list, N_list, C_list = scorer.score(premises, hypotheses)

    correct_l3 = 0
    out_recs = []
    for i, r in enumerate(recs):
        e_p = float(E_list[i])
        n_p = float(N_list[i])
        c_p = float(C_list[i])
        # correct = entailment prob >= threshold AND entailment > contradiction
        is_c = 1 if (e_p >= threshold and e_p > c_p) else 0
        correct_l3 += is_c
        out_recs.append({
            **r,
            'nli_entailment':    round(e_p, 4),
            'nli_neutral':       round(n_p, 4),
            'nli_contradiction': round(c_p, 4),
            'correct_l3':        is_c,
            'judge_method':      'deberta_nli_entailment',
        })

    summary = {
        'total_l3':   n,
        'correct_l3': correct_l3,
        'acc_l3':     round(correct_l3 / n, 4) if n else 0.0,
        'threshold':  threshold,
        'model':      model_path,
        'metric':     'nli_entailment',
    }

    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(out_recs, f, indent=2, ensure_ascii=False)
    with open(out_metrics_path, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'method': 'nli_entailment'}, f, indent=2, ensure_ascii=False)

    print(f'[L3-NLI] 完成 {l3_json_path.name}: acc_l3={summary["acc_l3"]:.4f} '
          f'({correct_l3}/{n})  → {out_json_path.name}', flush=True)
    return summary


# ─── 批量扫目录 ────────────────────────────────────────────
def batch_run(raw_root: Path,
              model_path: str = DEFAULT_MODEL,
              threshold: float = DEFAULT_THRESHOLD,
              batch_size: int = DEFAULT_BATCH_SIZE,
              device: str = DEFAULT_DEVICE,
              force: bool = False) -> dict:
    raw_root = Path(raw_root)
    if not raw_root.exists():
        raise FileNotFoundError(f'raw_root 不存在: {raw_root}')

    l3_files = sorted(raw_root.glob('**/vqa_l3_raw.json'))
    if not l3_files:
        print(f'[L3-NLI] 在 {raw_root} 下没找到 vqa_l3_raw.json', flush=True)
        return {}

    print(f'[L3-NLI] 发现 {len(l3_files)} 个 vqa_l3_raw.json', flush=True)
    scorer = NliScorer(model_path=model_path, device=device, batch_size=batch_size)
    scorer._ensure_loaded()

    total_l3, total_correct = 0, 0
    results = {}
    t0 = time.time()
    for i, l3f in enumerate(l3_files, 1):
        run_name = l3f.parent.name
        out_json = l3f.parent / 'vqa_l3_nli.json'
        if out_json.exists() and not force:
            print(f'[{i}/{len(l3_files)}] 跳过 {run_name} (vqa_l3_nli.json 已存在)', flush=True)
            met_p = l3f.parent / 'vqa_l3_nli_metrics.json'
            if met_p.exists():
                with open(met_p) as f:
                    existing = json.load(f)
                    results[run_name] = existing.get('summary', {})
                    if 'total_l3' in results[run_name]:
                        total_l3 += results[run_name]['total_l3']
                        total_correct += results[run_name]['correct_l3']
            continue

        try:
            recs = load_l3_records(l3f)
            n = len(recs)
            if n == 0:
                results[run_name] = {'total_l3': 0, 'acc_l3': 0, 'correct_l3': 0}
                continue
            premises, hypotheses = prepare_pairs(recs)
            E_list, N_list, C_list = scorer.score(premises, hypotheses)

            correct_l3 = 0
            out_recs = []
            for j, r in enumerate(recs):
                e_p = float(E_list[j])
                n_p = float(N_list[j])
                c_p = float(C_list[j])
                is_c = 1 if (e_p >= threshold and e_p > c_p) else 0
                correct_l3 += is_c
                out_recs.append({
                    **r,
                    'nli_entailment':    round(e_p, 4),
                    'nli_neutral':       round(n_p, 4),
                    'nli_contradiction': round(c_p, 4),
                    'correct_l3':        is_c,
                    'judge_method':      'deberta_nli_entailment',
                })

            summary = {
                'total_l3':   n,
                'correct_l3': correct_l3,
                'acc_l3':     round(correct_l3 / n, 4),
                'threshold':  threshold,
                'model':      model_path,
                'metric':     'nli_entailment',
            }
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(out_recs, f, indent=2, ensure_ascii=False)
            with open(l3f.parent / 'vqa_l3_nli_metrics.json', 'w', encoding='utf-8') as f:
                json.dump({'summary': summary, 'method': 'nli_entailment'}, f, indent=2, ensure_ascii=False)

            results[run_name] = summary
            total_l3 += n
            total_correct += correct_l3

            print(f'[{i}/{len(l3_files)}] {run_name}: '
                  f'acc_l3={summary["acc_l3"]:.4f} ({correct_l3}/{n})  '
                  f'[累计 {(time.time()-t0)/60:.1f}min]', flush=True)
        except Exception as e:
            import traceback
            print(f'[{i}/{len(l3_files)}] {run_name}: FAILED - {e}', flush=True)
            traceback.print_exc()
            results[run_name] = {'error': str(e)}

    overall = {
        'total_l3': total_l3,
        'correct_l3': total_correct,
        'acc_l3': round(total_correct / total_l3, 4) if total_l3 else 0,
        'threshold': threshold,
        'model': model_path,
        'metric': 'nli_entailment',
    }
    print(f'\n[L3-NLI] 全部完成: {len(results)} 个 run, '
          f'总 L3 acc={overall["acc_l3"]:.4f} ({total_correct}/{total_l3}), '
          f'总耗时 {(time.time()-t0)/60:.1f} min', flush=True)

    overview_path = raw_root / 'vqa_l3_nli_overview.json'
    with open(overview_path, 'w', encoding='utf-8') as f:
        json.dump({'overall': overall, 'per_run': results}, f, indent=2, ensure_ascii=False)
    print(f'[L3-NLI] 总览: {overview_path}', flush=True)
    return {'overall': overall, 'per_run': results}


# ─── 入口 ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='VRS-VQA L3 评测 — DeBERTa-xlarge-mnli NLI 头（entailment 判断）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('l3_json', nargs='?', help='单个 vqa_l3_raw.json 路径')
    parser.add_argument('--batch-dir', type=Path, default=None,
                        help='批量模式：扫该目录下所有 vqa_l3_raw.json')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help=f'NLI 模型路径（默认: {DEFAULT_MODEL}）')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help=f'entailment 概率阈值（默认: {DEFAULT_THRESHOLD}）')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'batch size（默认: {DEFAULT_BATCH_SIZE}）')
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE,
                        help=f'推理设备（默认: {DEFAULT_DEVICE}）')
    parser.add_argument('--force', action='store_true',
                        help='覆盖已存在的 vqa_l3_nli.json')
    args = parser.parse_args()

    if args.batch_dir:
        batch_run(args.batch_dir, model_path=args.model, threshold=args.threshold,
                  batch_size=args.batch_size, device=args.device, force=args.force)
    elif args.l3_json:
        run_l3_nli_eval(Path(args.l3_json), model_path=args.model, threshold=args.threshold,
                        batch_size=args.batch_size, device=args.device, force=args.force)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
