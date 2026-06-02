#!/usr/bin/env python3
"""
自动化批量评测脚本 v2
=====================
扫描指定目录下的所有 selection JSON，
对每个 JSON 调用 eval_select.py 的 run_eval()（在同一进程内），
每跑完一个 JSON 立即按 eval_select.py 的逻辑保存结果。

用法:
  python3 run_bench_dir.py
  python3 run_bench_dir.py annotation_data/sampled_eval/lever_test/
  python3 run_bench_dir.py /path/to/selections/ --model qwen3.5-4B --max_new 4096
"""
import json
import os
import sys
import time
from pathlib import Path

# 确保项目根目录在 Python path
PROJ = Path('/home/admin1/projects/remote_vlm_eval')
sys.path.insert(0, str(PROJ))
os.environ['PYTHONPATH'] = str(PROJ)

import torch

# 默认参数
DEFAULT_SEL_DIR = PROJ / 'annotation_data/sampled_eval/lever_test'
DEFAULT_MODEL = 'qwen3.5-4B'
DEFAULT_MAX_NEW = 4096

# think-off 模式（不开启 thinking）
ENABLE_THINKING = False


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='批量跑 selection JSON 评测')
    parser.add_argument('sel_dir', nargs='?', type=Path, default=DEFAULT_SEL_DIR,
                        help='selection JSON 所在目录（默认: %s）' % DEFAULT_SEL_DIR)
    parser.add_argument('--model', '-m', default=DEFAULT_MODEL,
                        help='模型名称（默认: %s）' % DEFAULT_MODEL)
    parser.add_argument('--max_new', type=int, default=DEFAULT_MAX_NEW,
                        help='max_new_tokens（默认: %d）' % DEFAULT_MAX_NEW)
    return parser.parse_args()


def scan_selections(sel_dir: Path):
    """扫描目录下所有 .json 文件，按文件名排序。"""
    jsons = sorted(sel_dir.glob('*.json'))
    return [p for p in jsons if p.name != '_batch_progress.json']
    # 跳过 _batch_progress.json 等临时文件


def load_selection_json(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    args = parse_args()
    sel_dir: Path = args.sel_dir
    model_name: str = args.model
    max_new: int = args.max_new

    if not sel_dir.exists():
        print('[ERROR] 目录不存在: %s' % sel_dir)
        sys.exit(1)

    selections = scan_selections(sel_dir)
    if not selections:
        print('[ERROR] 目录为空，没有找到 .json 文件: %s' % sel_dir)
        sys.exit(1)

    print('=' * 60)
    print('BATCH START')
    print('  sel_dir : %s' % sel_dir)
    print('  model   : %s' % model_name)
    print('  max_new : %d' % max_new)
    print('  thinking: %s' % ENABLE_THINKING)
    print('  files   : %d' % len(selections))
    print('=' * 60)

    # 导入 eval_select.py 的核心函数
    from eval_select import run_eval
    from config import RAW_DIR_ROOT, RES_DIR

    results = []  # 汇总

    for i, sel_path in enumerate(selections):
        sel_name = sel_path.stem
        print('')
        print('[%d/%d] ===== %s =====' % (i + 1, len(selections), sel_name))
        t0 = time.time()

        try:
            sel_doc = load_selection_json(sel_path)
        except Exception as e:
            print('[ERROR] 无法加载 %s: %s' % (sel_path, e))
            results.append({'selection': sel_name, 'status': 'load_error', 'error': str(e)})
            continue

        # 给 select_doc 补上 _path（run_eval 用这个算 select_name）
        sel_doc['_path'] = str(sel_path)

        try:
            row = run_eval(
                model_name=model_name,
                enable_thinking=ENABLE_THINKING,
                select_doc=sel_doc,
                max_new=max_new,
            )
            elapsed = time.time() - t0
            status = 'ok'
            # 提取各 bench 结果摘要
            bms = row.get('benchmarks', {}) if isinstance(row, dict) else {}
            summary = {}
            for bm, vals in bms.items():
                if isinstance(vals, dict):
                    if 'bleu4' in vals:
                        summary[bm] = 'bleu4=%.4f' % vals.get('bleu4', 0)
                    elif 'acc' in vals:
                        summary[bm] = 'acc=%.4f' % vals.get('acc', 0)
            print('[OK] %s 完成 (%ds) -- %s' % (sel_name, elapsed, summary))
            results.append({
                'selection': sel_name,
                'status': 'ok',
                'elapsed_s': round(elapsed, 1),
                'benchmarks': summary,
            })
        except Exception as e:
            import traceback
            elapsed = time.time() - t0
            traceback.print_exc()
            print('[ERROR] %s 失败 (%ds): %s' % (sel_name, elapsed, e))
            results.append({'selection': sel_name, 'status': 'error', 'error': str(e)})

        # 每个 JSON 跑完立即打印进度
        done = i + 1
        remain = len(selections) - done
        print('[PROGRESS] %d/%d done, %d remaining' % (done, len(selections), remain))

    # 打印最终汇总
    print('')
    print('================== ALL DONE ==================')
    ok_count = sum(1 for r in results if r['status'] == 'ok')
    print('  Total: %d | OK: %d | Error: %d' % (len(results), ok_count, len(results) - ok_count))
    for r in results:
        tag = '[OK]' if r['status'] == 'ok' else '[FAIL]'
        bms = r.get('benchmarks', {})
        bm_str = ', '.join('%s=%s' % (k, v) for k, v in bms.items()) if bms else r.get('error', '')
        print('  %s %s -- %s' % (tag, r['selection'], bm_str))


if __name__ == '__main__':
    main()