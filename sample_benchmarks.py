#!/usr/bin/env python3
"""
多 bench 灵活采样器 — 从 5 个 benchmark 数据集按指定规则采样，输出 selection JSON。

采样规则：
- VRS-VQA（--vqa N）: per-type 采样，每个 type 抽 N，共 12*N 道题
- 其他 bench（--mme/--vrs-cap/--levir/--xlrs N）: 整体采样，从池中随机抽 N 道
  （触顶则取完，warn 提示）

输出（eval_select.py 兼容）：
  {out_dir}/{name}.json          # 总览（含 benchmarks.<key>.{source, select_by, ids}）
  {out_dir}/<bench>_<mode>{N}.json  # 单 bench JSON，可直接 --select

用法：
  # VQA 每类 20（240 题）+ MME 100 题
  python3 sample_benchmarks.py --vqa 20 --mme 100

  # 全 5 bench，自定义路径
  python3 sample_benchmarks.py --vqa 50 --mme 200 --vrs-cap 100 --levir 200 --xlrs 42 \\
      --name full_run1 --out /home/admin1/projects/remote_vlm_eval/annotation_data/sampled_eval

  # 自定义种子
  python3 sample_benchmarks.py --vqa 20 --mme 100 --seed 7 --out ./my_seeds
"""
import argparse
import json
import random
import sys
from pathlib import Path
from collections import defaultdict
from typing import Any

# ─── 数据源（与 config.py 保持一致） ─────────────────────────
DATA_SOURCES = {
    'vrs_vqa':     '/home/admin1/models/VRSBench_EVAL_vqa.json',
    'vrs_caption': '/home/admin1/models/VRSBench_EVAL_Cap.json',
    'mme_rs':      '/home/admin1/models/mme_rs_annotations.json',
    'levir_cc':    '/home/admin1/models/levircc_data/extracted/LevirCCcaptions.json',
    'xlrs':        '/home/admin1/models/xlrs_arrow/xlrs_samples_42.json',
}

# ─── 各 bench 的分组字段（per-type 采样的依据） ─────────────
GROUP_FIELDS = {
    'vrs_vqa':     'type',         # 12 类
    'vrs_caption': None,           # 1 类（不分子任务）
    'mme_rs':      'Category',     # 3 类 (color/count/position)
    'levir_cc':    'changeflag',   # 2 类 (0/1)
    'xlrs':        'category',     # 1 类
}

# ─── 各 bench 的 id 字段（写入 select_by，必须 unique） ─────
ID_FIELDS = {
    'vrs_vqa':     'question_id',
    'vrs_caption': 'question_id',
    'mme_rs':      'Question_id',
    'levir_cc':    'imgid',
    'xlrs':        'idx',
}

# ─── 预过滤：LEVIR-CC 只用 test split ─────────────────────
PRE_FILTERS = {
    'levir_cc': lambda r: r.get('filepath') == 'test' or r.get('split') == 'test',
}


def load_records(bench: str) -> list[dict]:
    """加载并预过滤某个 bench 的全部 records。"""
    src = DATA_SOURCES[bench]
    with open(src) as f:
        data = json.load(f)
    if bench == 'levir_cc':
        recs = data.get('images', data)
    else:
        recs = data
    pf = PRE_FILTERS.get(bench)
    if pf:
        recs = [r for r in recs if pf(r)]
    return recs


def group_records(records: list[dict], group_field: str | None) -> dict[Any, list[dict]]:
    """把 records 按 group_field 分组，返回 {group_value: [record, ...]}。"""
    if group_field is None:
        return {'_all': list(records)}
    groups: dict[Any, list[dict]] = defaultdict(list)
    for r in records:
        g = r.get(group_field, '_NO_FIELD_')
        groups[g].append(r)
    return dict(groups)


def sample_per_type(records: list[dict], k: int, rng: random.Random,
                    group_field: str) -> tuple[list[dict], dict, dict]:
    """
    per-type 采样：每组抽 k 个 record（不足则全取）。
    返回 (picked_records, per_type_count, pool_per_type)。
    """
    groups = group_records(records, group_field)
    picked: list[dict] = []
    per_type = {}
    pool = {g: len(rec_list) for g, rec_list in groups.items()}
    for g, rec_list in sorted(groups.items()):
        pool_size = len(rec_list)
        kk = min(k, pool_size)
        chosen = rng.sample(rec_list, kk) if pool_size > 0 else []
        picked.extend(chosen)
        per_type[g] = len(chosen)
        if kk < k:
            print(f"    [WARN] {g!r} 池仅 {pool_size} < k={k}，已取完（{kk}）")
    return picked, per_type, pool


def sample_total(records: list[dict], n: int, rng: random.Random,
                 group_field: str | None) -> tuple[list[dict], dict, dict]:
    """
    整体采样：从所有 records 抽 n 个 record。
    返回 (picked_records, per_type_count, pool_per_type)。
    """
    groups = group_records(records, group_field)
    all_recs = []
    for rec_list in groups.values():
        all_recs.extend(rec_list)
    pool_size = len(all_recs)
    kk = min(n, pool_size)
    picked = rng.sample(all_recs, kk) if pool_size > 0 else []
    if kk < n:
        print(f"    [WARN] 池仅 {pool_size} < n={n}，已取完（{kk}）")
    # 统计 picked 的 per_type 分布
    picked_per_type: dict[Any, int] = defaultdict(int)
    for r in picked:
        if group_field is None:
            picked_per_type['_all'] += 1
        else:
            picked_per_type[r.get(group_field, '_NO_FIELD_')] += 1
    return picked, dict(picked_per_type), \
           {g: len(rec_list) for g, rec_list in groups.items()}


def build_bench_block(bench: str, picked_records: list[dict], per_type: dict, pool: dict,
                      mode: str, n: int) -> dict:
    """构造 selection JSON 中一个 bench 的 block（eval_select.py 兼容）。"""
    id_field = ID_FIELDS[bench]
    ids = [r.get(id_field) for r in picked_records]
    # 校验 id 都存在且非 None
    if any(i is None for i in ids):
        miss = sum(1 for i in ids if i is None)
        print(f"    [WARN] {bench}: {miss} 条 record 缺少 {id_field!r} 字段")
    return {
        'source': DATA_SOURCES[bench],
        'select_by': id_field,
        'ids': ids,
        'mode': mode,                # 'k' = per-type, 'n' = total
        'requested': n,
        'actual_count': len(picked_records),
        'per_type': per_type,
        'pool': pool,
    }


def main():
    ap = argparse.ArgumentParser(
        description='多 bench 灵活采样器（生成 eval_select.py 兼容的 selection JSON）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument('--vqa', type=int, default=None,
                    help='VRS-VQA 每 type 采样数（per-type，12*N 题）')
    ap.add_argument('--mme', type=int, default=None,
                    help='MME-RS 整体采样数（total）')
    ap.add_argument('--vrs-cap', type=int, default=None,
                    help='VRSBench Caption 整体采样数（total）')
    ap.add_argument('--levir', type=int, default=None,
                    help='LEVIR-CC 整体采样数（total，test split 内）')
    ap.add_argument('--xlrs', type=int, default=None,
                    help='XLRS 整体采样数（total，触顶 42）')
    ap.add_argument('--out', type=Path, required=True,
                    help='保存目录（必填）')
    ap.add_argument('--name', type=str, default='selection',
                    help='总览文件名（不含 .json），默认 selection')
    ap.add_argument('--seed', type=int, default=42,
                    help='随机种子，默认 42')
    ap.add_argument('--overwrite', action='store_true',
                    help='允许覆盖已存在文件（默认不覆盖）')
    args = ap.parse_args()

    # 至少指定一个 bench
    bench_args = {
        'vrs_vqa':     args.vqa,
        'vrs_caption': args.vrs_cap,
        'mme_rs':      args.mme,
        'levir_cc':    args.levir,
        'xlrs':        args.xlrs,
    }
    chosen = {b: n for b, n in bench_args.items() if n is not None and n > 0}
    if not chosen:
        ap.error('至少指定一个 bench 的采样数（--vqa / --mme / --vrs-cap / --levir / --xlrs）')

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    overview_path = args.out / f'{args.name}.json'
    if overview_path.exists() and not args.overwrite:
        print(f'[ERROR] {overview_path} 已存在，加 --overwrite 覆盖')
        sys.exit(1)

    overview: dict = {
        'name': args.name,
        'seed': args.seed,
        'created_at': __import__('time').strftime('%Y-%m-%dT%H:%M:%SZ', __import__('time').gmtime()),
        'sampling': 'per-type' if 'vrs_vqa' in chosen else 'total',
        'benchmarks': {},
    }

    print(f'\n{"="*60}')
    print(f'  sample_benchmarks — name={args.name}, seed={args.seed}')
    print(f'  out: {args.out}')
    print(f'{"="*60}')

    # 逐个 bench 采样
    for bench, n in chosen.items():
        print(f'\n  [{bench}] 采样 {n}（{"per-type" if bench == "vrs_vqa" else "total"}）...')
        records = load_records(bench)
        print(f'    池: {len(records)} 条')

        if bench == 'vrs_vqa':
            picked, per_type, pool = sample_per_type(
                records, n, rng, GROUP_FIELDS[bench])
            mode = 'k'
        else:
            picked, per_type, pool = sample_total(
                records, n, rng, GROUP_FIELDS[bench])
            mode = 'n'

        block = build_bench_block(bench, picked, per_type, pool, mode, n)
        overview['benchmarks'][bench] = block

        # 打印摘要
        print(f'    实际抽到: {len(picked)} 题')
        if per_type:
            for g, c in sorted(per_type.items(), key=lambda x: -x[1])[:5]:
                pool_n = pool.get(g, '?')
                print(f'      {g}: {c} / {pool_n}')
            if len(per_type) > 5:
                print(f'      ... (共 {len(per_type)} 组)')

        # 写单 bench JSON
        single_path = args.out / f'{bench}_{mode}{n}.json'
        if single_path.exists() and not args.overwrite:
            print(f'    [SKIP] {single_path} 已存在')
            continue
        single_doc = {
            'name': single_path.stem,
            'benchmark': bench,
            'source': block['source'],
            'select_by': block['select_by'],
            'ids': block['ids'],
            'mode': mode,
            'requested': n,
            'actual_count': len(picked),
            'per_type': per_type,
            'pool': pool,
        }
        with open(single_path, 'w') as f:
            json.dump(single_doc, f, indent=2, ensure_ascii=False)
        print(f'    写: {single_path}')

    # 写总览
    with open(overview_path, 'w') as f:
        json.dump(overview, f, indent=2, ensure_ascii=False)
    print(f'\n  总览: {overview_path}')
    print(f'  完成。共 {len(overview["benchmarks"])} 个 bench，{sum(b["actual_count"] for b in overview["benchmarks"].values())} 道题。')


if __name__ == '__main__':
    main()
