#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from config import RAW_DIR_ROOT, OUT_DIR


def _read_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _parse_tag(tag: str):
    if '_thinkON_' in tag:
        model, rest = tag.split('_thinkON_', 1)
        return model, 'thinkON', rest
    if '_thinkOFF_' in tag:
        model, rest = tag.split('_thinkOFF_', 1)
        return model, 'thinkOFF', rest
    return None


def _safe_div(a: int, b: int):
    return (a / b) if b else None


def _fmt_acc(v):
    if v is None:
        return ''
    return round(v, 4)


def load_vrs_type_map():
    path = Path('/home/admin1/models/VRSBench_EVAL_vqa.json')
    if not path.exists():
        return {}
    rows = _read_json(path)
    m = {}
    for r in rows:
        qid = r.get('question_id')
        if qid is None:
            continue
        m[str(qid)] = r.get('type', '')
    return m


def load_mme_category_map():
    path = Path('/home/admin1/models/mme_rs_annotations.json')
    if not path.exists():
        return {}
    rows = _read_json(path)
    m = {}
    for r in rows:
        qid = r.get('Question_id')
        if not qid:
            continue
        m[str(qid)] = r.get('Category', '')
    return m


def collect_runs(selection_name: str):
    root = Path(RAW_DIR_ROOT) / selection_name
    if not root.exists():
        raise SystemExit(f'not found: {root}')
    runs = []
    for fp in sorted(root.glob('*/raw_outputs.json')):
        tag = fp.parent.name
        parsed = _parse_tag(tag)
        if not parsed:
            continue
        model, mode, _ = parsed
        runs.append((model, mode, tag, fp))
    return runs


def aggregate(selection_name: str):
    vrs_type = load_vrs_type_map()
    mme_cat = load_mme_category_map()

    runs = collect_runs(selection_name)
    model_modes = sorted({f'{m} ({mode})' for m, mode, _, _ in runs})

    vrs = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    mme = defaultdict(lambda: defaultdict(lambda: [0, 0]))

    for model, mode, tag, fp in runs:
        key = f'{model} ({mode})'
        rows = _read_json(fp)
        if not isinstance(rows, list):
            continue

        for r in rows:
            bench = r.get('benchmark')
            if bench == 'VRS-VQA':
                qid = r.get('_idx')
                t = vrs_type.get(str(qid), '')
                if not t:
                    t = 'unknown'
                ok = int(r.get('correct', 0) == 1)
                vrs[t][key][0] += ok
                vrs[t][key][1] += 1
            elif bench == 'MME-RS':
                qid = r.get('_idx')
                c = mme_cat.get(str(qid), '')
                if not c:
                    c = 'unknown'
                gt = str(r.get('gt', '')).strip().upper()
                pred = str(r.get('pred', '')).strip().upper()
                ok = int(bool(gt) and pred == gt)
                mme[c][key][0] += ok
                mme[c][key][1] += 1

    return model_modes, vrs, mme


def _write_wide_csv(path: Path, title: str, model_modes: list[str], rows: list[tuple[str, dict]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([title])
        w.writerow(['task_type', 'n'] + model_modes)
        for task_type, stats in rows:
            n = max((b for _, b in stats.values()), default=0)
            line = [task_type, n]
            for mm in model_modes:
                a, b = stats.get(mm, (0, 0))
                line.append(_fmt_acc(_safe_div(a, b)))
            w.writerow(line)


def _print_table(title: str, model_modes: list[str], rows: list[tuple[str, dict]]):
    widths = [max(len('task_type'), max((len(t) for t, _ in rows), default=0)), len('n')]
    for mm in model_modes:
        widths.append(max(len(mm), len('acc')))

    def _row(vals):
        return ' | '.join(str(v).ljust(widths[i]) for i, v in enumerate(vals))

    print(title)
    print(_row(['task_type', 'n'] + model_modes))
    print(_row(['-' * widths[0], '-' * widths[1]] + ['-' * w for w in widths[2:]]))
    for task_type, stats in rows:
        n = max((b for _, b in stats.values()), default=0)
        vals = [task_type, n]
        for mm in model_modes:
            a, b = stats.get(mm, (0, 0))
            vals.append(_fmt_acc(_safe_div(a, b)))
        print(_row(vals))
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection', default='batch3_sel', help='selectionName（raw_outputs/{selectionName}）')
    parser.add_argument('--out', default='', help='输出 CSV 前缀（默认 selectionName）')
    args = parser.parse_args()

    selection_name = args.selection
    out_prefix = args.out or selection_name

    model_modes, vrs, mme = aggregate(selection_name)

    vrs_rows = []
    for t in sorted(vrs.keys()):
        vrs_rows.append((t, {k: tuple(v) for k, v in vrs[t].items()}))
    mme_rows = []
    for c in sorted(mme.keys()):
        mme_rows.append((c, {k: tuple(v) for k, v in mme[c].items()}))

    if vrs_rows:
        _print_table('VRS-VQA', model_modes, vrs_rows)
        _write_wide_csv(OUT_DIR / f'{out_prefix}_VRS_VQA.csv', 'VRS-VQA', model_modes, vrs_rows)
    if mme_rows:
        _print_table('MME-RS', model_modes, mme_rows)
        _write_wide_csv(OUT_DIR / f'{out_prefix}_MME_RS.csv', 'MME-RS', model_modes, mme_rows)

    if not vrs_rows and not mme_rows:
        raise SystemExit('no VRS-VQA/MME-RS records found')

