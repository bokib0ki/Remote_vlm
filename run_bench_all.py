#!/usr/bin/env python3
"""
自动化批量评测脚本
==================
Qwen3.5-4B 和 MiniCPM-V 推理所有 bench，think-off 模式，max_new=4096。
每完成一个 bench 就保存一次结果（incremental save），不怕中断。

用法:
  python3 run_bench_all.py
  python3 run_bench_all.py --max_new 2048
"""
import json
import os
import sys
import time
import subprocess
from pathlib import Path

PROJ = Path('/home/admin1/projects/remote_vlm_eval')
# 所有要跑的 selection 文件
SELECTIONS = [
    PROJ / 'annotation_data/sampled_eval/lever_test/lever_k=50_sel.json',
    PROJ / 'annotation_data/sampled_eval/lever_test/lever_k=100_sel.json',
    PROJ / 'annotation_data/sampled_eval/lever_test/lever_k=200_sel.json',
    PROJ / 'annotation_data/sampled_eval/lever_test/lever_k=400_sel.json',
    PROJ / 'annotation_data/sampled_eval/lever_test/lever_k=500_sel.json',
]
MAX_NEW = 4096

MODELS = [
    'qwen3.5-4B',
    'minicpm-v',
]

BENCH_ORDER = ['vrs_caption', 'vrs_vqa', 'mme_rs', 'levir_cc', 'xlrs']

PROGRESS_FILE = PROJ / 'results' / '_batch_progress.json'
LOG_FILE = PROJ / 'results' / '_batch_run.log'


def log(msg):
    ts = time.strftime('%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {}


def save_progress(data):
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def run_eval(model: str, selection: str, bench: str, max_new: int) -> dict:
    """运行单次 eval_select.py。"""
    cmd = [
        sys.executable, str(PROJ / 'eval_select.py'),
        '--select', selection,
        '--model', model,
        '--bench', bench,
        '--max_new', str(max_new),
    ]
    # think-off: 不传 --thinking 参数
    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJ)

    cmd_str = ' '.join(cmd)
    log(f'  -> CMD: {cmd_str}')
    result = subprocess.run(
        cmd,
        cwd=str(PROJ),
        env=env,
        capture_output=True,
        text=True,
        timeout=7200,  # 2h per bench
    )
    return {
        'returncode': result.returncode,
        'stdout': result.stdout[-3000:],
        'stderr': result.stderr[-1000:],
    }


def extract_done_line(bench: str, result: dict) -> str:
    out = result.get('stdout', '')
    for line in out.split('\n'):
        if f'done [{bench}]' in line:
            return line.strip()
    return ''


def already_done(progress: dict, model: str, selection: str, bench: str) -> bool:
    """检查是否已完成（returncode==0）。"""
    for r in progress.get('runs', []):
        if r.get('model') == model and r.get('bench') == bench and r.get('selection') == selection:
            return r.get('status') == 'ok'
    return False


def main():
    log('=' * 60)
    log('BATCH START: models=%s, selections=%d, benches=%s, max_new=%d' % (
        MODELS, len(SELECTIONS), BENCH_ORDER, MAX_NEW))
    log('=' * 60)

    progress = load_progress()
    if 'runs' not in progress:
        progress['runs'] = []

    total_runs = 0
    skip_runs = 0

    for selection_path in SELECTIONS:
        selection = str(selection_path)
        sel_name = selection_path.stem  # e.g. lever_k=50_sel

        if not Path(selection).exists():
            log('[SKIP] 文件不存在: %s' % selection)
            continue

        with open(selection) as f:
            sel = json.load(f)
        benches_in_sel = list(sel.get('benchmarks', {}).keys())

        log('')
        log('>>> Selection: %s (benches: %s)' % (sel_name, benches_in_sel))

        for model in MODELS:
            for bench in BENCH_ORDER:
                if bench not in benches_in_sel:
                    log('  [SKIP] %s/%s -- 不在此 selection 中' % (model, bench))
                    continue

                run_key = '%s/%s/%s' % (model, sel_name, bench)

                if already_done(progress, model, selection, bench):
                    log('  [SKIP] %s -- 已完成，跳过' % run_key)
                    skip_runs += 1
                    continue

                total_runs += 1
                log('')
                log('========== %s START ==========' % run_key)
                t0 = time.time()

                try:
                    res = run_eval(model, selection, bench, MAX_NEW)
                except subprocess.TimeoutExpired:
                    log('[TIMEOUT] %s 超过2小时，被强制终止' % run_key)
                    progress['runs'].append({
                        'model': model, 'bench': bench,
                        'selection': selection, 'selection_name': sel_name,
                        'status': 'timeout',
                        'elapsed_s': time.time() - t0,
                    })
                    save_progress(progress)
                    continue

                elapsed = time.time() - t0
                done_line = ''
                status = 'ok' if res['returncode'] == 0 else 'error'

                if res['returncode'] == 0:
                    done_line = extract_done_line(bench, res)
                    log('[OK] %s 完成 (%ds) -- %s' % (run_key, elapsed, done_line))
                else:
                    log('[ERROR] %s 失败 (rc=%d)' % (run_key, res['returncode']))
                    log('  stderr: %s' % res['stderr'][-500:])

                progress['runs'].append({
                    'model': model,
                    'bench': bench,
                    'selection': selection,
                    'selection_name': sel_name,
                    'status': status,
                    'elapsed_s': round(elapsed, 1),
                    'done_line': done_line,
                })
                save_progress(progress)
                log('========== %s END (%ds) ==========' % (run_key, elapsed))

    log('')
    log('================== ALL DONE ==================')
    log('Progress saved: %s' % PROGRESS_FILE)
    log('Total runs: %d | Skipped (already done): %d' % (total_runs, skip_runs))
    # 打印汇总
    runs = progress.get('runs', [])
    for r in runs:
        tag = 'OK' if r['status'] == 'ok' else 'FAIL'
        sel = r.get('selection_name', r.get('selection', ''))
        log('  [%s] %s/%s/%s -- %s' % (
            tag, r.get('model', ''), sel, r.get('bench', ''), r.get('done_line', '')))


if __name__ == '__main__':
    main()