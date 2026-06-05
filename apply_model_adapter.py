"""
后处理：按 model_adapter 重新裁剪 pred，重算 L1/L2 correct 字段

只处理 lever_test2 目录（张哥要求）

处理流程:
  1. 扫 raw_outputs/lever_test2/lever_k=*_vqa/*/raw_outputs.json
  2. 解析目录名得到 model
  3. 找对应 adapter
  4. 备份原 raw_outputs.json → raw_outputs.json.bak
  5. 用 adapter.strip_thinking + post_process_pred 重新生成 pred
     （原 pred 保存到 pred_raw，如已有 pred_raw 则保留）
  6. 重跑 L1 (substring) + L2 (yes/no/数字精确)
  7. 更新 correct, judge_level, judge_method 字段
  8. 重生成 vqa_l3_raw.json (用新 pred)
  9. 删除 vqa_l3_nli.json (让 L3 NLI 重跑)

用法:
  python apply_model_adapter.py                # 默认处理 lever_test2
  python apply_model_adapter.py --no-backup    # 不备份
  python apply_model_adapter.py --dry-run      # 只统计不修改
"""
import argparse
import json
import re
import shutil
from pathlib import Path
from collections import Counter
import sys

# 让 model_adapter 可以 import
sys.path.insert(0, str(Path(__file__).parent))
from model_adapter import get_adapter, list_adapters


# ── L1/L2 判别（从 vqa_judge.py 复制，独立 import） ─────────────
def vqa_judge_l1(gt: str, pred: str) -> int:
    """L1: substring 匹配"""
    return 1 if gt.lower() in pred.lower() else 0


def vqa_judge_l2(gt: str, pred: str):
    """L2: yes/no/数字 精确匹配。返回 1/0/None"""
    gt_lower = gt.lower().strip()
    pred_lower = pred.lower().strip()
    if gt_lower in ('yes', 'no'):
        return 1 if gt_lower == pred_lower else 0
    try:
        gt_num = int(gt_lower)
        pred_num = int(pred_lower) if pred_lower.isdigit() else -1
        return 1 if gt_num == pred_num else 0
    except ValueError:
        pass
    return None


# ── 解析目录名（与 gen_vqa_type_report_v3.py 一致） ─────────────
def parse_dir_name(dirname: str):
    for pat in [
        r'^(.+?)_(think(?:ON|OFF))_lever_test2_lever_k=(\d+)_vqa$',
        r'^(.+?)_(think(?:ON|OFF))_lever_k=(\d+)$',
    ]:
        m = re.match(pat, dirname)
        if m:
            return m.group(1), m.group(2), int(m.group(3))
    return None


# ── 主流程 ──────────────────────────────────────────────
def process_run(run_dir: Path, dry_run: bool = False) -> dict:
    raw_path = run_dir / 'raw_outputs.json'
    l3_raw_path = run_dir / 'vqa_l3_raw.json'
    l3_nli_path = run_dir / 'vqa_l3_nli.json'

    if not raw_path.exists():
        return {'skipped': True, 'reason': 'no raw_outputs.json'}

    parsed = parse_dir_name(run_dir.name)
    if not parsed:
        return {'skipped': True, 'reason': f'目录名不匹配: {run_dir.name}'}
    model, mode, k = parsed

    try:
        adapter = get_adapter(model)
    except KeyError as e:
        return {'skipped': True, 'reason': f'无 adapter: {e}'}

    recs = json.load(open(raw_path))
    n_total = len(recs)

    # 统计旧字段
    n_old_correct = sum(1 for r in recs if r.get('correct') == 1)
    n_old_l1 = sum(1 for r in recs if r.get('judge_level') == 'L1')
    n_old_l2 = sum(1 for r in recs if r.get('judge_level') == 'L2')
    n_old_l3 = sum(1 for r in recs if r.get('judge_level') == 'L3')

    # 重新处理每条记录
    n_changed = 0
    n_l1 = n_l2 = n_l3 = 0
    n_correct = 0
    new_recs = []
    for r in recs:
        # pred_raw: 如果已经有，就用旧的；否则把 pred 当 raw
        old_pred = str(r.get('pred', ''))
        old_pred_raw = r.get('pred_raw', '')
        if old_pred_raw:
            raw_text = old_pred_raw
        else:
            raw_text = old_pred

        # 跑 adapter 清洗
        new_pred = adapter.clean_pred(raw_text)

        # 重跑 L1/L2
        gt = str(r.get('gt', '')).strip()
        l1_ok = vqa_judge_l1(gt, new_pred)
        if l1_ok:
            new_level = 'L1'
            new_method = 'substring'
            new_correct = 1
            new_ok = 1
            n_l1 += 1
        else:
            l2_res = vqa_judge_l2(gt, new_pred)
            if l2_res is not None:
                new_level = 'L2'
                new_method = 'exact'
                new_correct = l2_res
                new_ok = l2_res
                n_l2 += 1
            else:
                new_level = 'L3'
                new_method = 'pending'
                new_correct = 0
                new_ok = 0
                n_l3 += 1
        n_correct += new_correct

        # 检测是否真改了
        if new_pred != old_pred or new_correct != int(r.get('correct', 0)) or new_level != r.get('judge_level'):
            n_changed += 1

        # 更新记录
        new_r = dict(r)
        new_r['pred'] = new_pred
        new_r['pred_raw'] = raw_text  # 强制保存原始
        new_r['correct'] = new_correct
        new_r['judge_level'] = new_level
        new_r['judge_method'] = new_method
        new_recs.append(new_r)

    if dry_run:
        return {
            'model': model, 'mode': mode, 'k': k,
            'n_total': n_total, 'n_changed': n_changed,
            'old_correct': n_old_correct, 'new_correct': n_correct,
            'old_l1/l2/l3': f'{n_old_l1}/{n_old_l2}/{n_old_l3}',
            'new_l1/l2/l3': f'{n_l1}/{n_l2}/{n_l3}',
            'dry_run': True,
        }

    # 备份
    backup = raw_path.with_suffix('.json.bak')
    if not backup.exists():
        shutil.copy2(raw_path, backup)

    # 写回
    with open(raw_path, 'w', encoding='utf-8') as f:
        json.dump(new_recs, f, indent=2, ensure_ascii=False)

    # 重生成 vqa_l3_raw.json
    l3_records = [r for r in new_recs if r['judge_level'] == 'L3']
    if l3_records:
        # 只保留 VQA_L3_bert.py 需要的字段
        l3_out = []
        for r in l3_records:
            l3_out.append({
                'benchmark': 'VRS-VQA',
                '_idx': r.get('_idx'),
                'image_id': r.get('image_id', ''),
                'gt': r.get('gt', ''),
                'pred': r.get('pred', ''),
                'question': r.get('question', ''),
                'pred_raw': r.get('pred_raw', ''),
            })
        if l3_raw_path.exists():
            shutil.copy2(l3_raw_path, l3_raw_path.with_suffix('.json.bak'))
        with open(l3_raw_path, 'w', encoding='utf-8') as f:
            json.dump(l3_out, f, indent=2, ensure_ascii=False)

    # 删 NLI 结果（让 VQA_L3_bert.py 重跑）
    if l3_nli_path.exists():
        l3_nli_bak = l3_nli_path.with_suffix('.json.bak')
        if not l3_nli_bak.exists():
            shutil.copy2(l3_nli_path, l3_nli_bak)
        l3_nli_path.unlink()
    # metrics 也删
    met_path = run_dir / 'vqa_l3_nli_metrics.json'
    if met_path.exists():
        met_bak = met_path.with_suffix('.json.bak')
        if not met_bak.exists():
            shutil.copy2(met_path, met_bak)
        met_path.unlink()

    return {
        'model': model, 'mode': mode, 'k': k,
        'n_total': n_total, 'n_changed': n_changed,
        'old_correct': n_old_correct, 'new_correct': n_correct,
        'old_l1/l2/l3': f'{n_old_l1}/{n_old_l2}/{n_old_l3}',
        'new_l1/l2/l3': f'{n_l1}/{n_l2}/{n_l3}',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-root', type=Path, default=Path('/home/admin1/models/raw_outputs/lever_test2'))
    parser.add_argument('--dry-run', action='store_true', help='只统计不修改')
    parser.add_argument('--no-backup', action='store_true', help='不备份原文件')
    args = parser.parse_args()

    print(f"raw_root:    {args.raw_root}")
    print(f"dry_run:     {args.dry_run}")
    print(f"adapters:    {list_adapters()}")
    print()

    run_dirs = sorted([d for d in args.raw_root.glob('lever_k=*_vqa/*') if d.is_dir()])
    print(f"扫到 {len(run_dirs)} 个 run 目录\n")

    summary = []
    for run_dir in run_dirs:
        result = process_run(run_dir, dry_run=args.dry_run)
        summary.append(result)
        if 'skipped' in result:
            print(f"  [SKIP] {run_dir.name}: {result['reason']}")
        else:
            old_acc = result['old_correct']/result['n_total']*100
            new_acc = result['new_correct']/result['n_total']*100
            delta = new_acc - old_acc
            sign = '+' if delta > 0 else ''
            print(f"  {result['model']:25} {result['mode']:9} k={result['k']:>3}: "
                  f"改 {result['n_changed']:>3} 条  "
                  f"acc: {old_acc:5.2f}% → {new_acc:5.2f}% ({sign}{delta:.2f}pp)  "
                  f"L1/L2/L3: {result['old_l1/l2/l3']} → {result['new_l1/l2/l3']}")

    # 总体
    not_skip = [r for r in summary if 'skipped' not in r]
    if not_skip:
        n_total = sum(r['n_total'] for r in not_skip)
        n_old = sum(r['old_correct'] for r in not_skip)
        n_new = sum(r['new_correct'] for r in not_skip)
        n_changed = sum(r['n_changed'] for r in not_skip)
        print(f"\n=== 汇总 ===")
        print(f"  处理 {len(not_skip)} 个 run, 共 {n_total} 条")
        print(f"  改动 {n_changed} 条 ({n_changed/n_total*100:.1f}%)")
        print(f"  整体 acc: {n_old/n_total*100:.2f}% → {n_new/n_total*100:.2f}% ({(n_new-n_old)/n_total*100:+.2f}pp)")
        print(f"  {'(DRY RUN)' if args.dry_run else '(已修改原文件)'}")


if __name__ == '__main__':
    main()
