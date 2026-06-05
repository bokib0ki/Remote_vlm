#!/usr/bin/env python3
"""
从 VRSBench_EVAL_vqa.json 抽 3 份新题（不放回），与现有 lever_k=400_vqa.json 不重复。

规则：
  - 每类型每份 400 道
  - 4 份（1 老 + 3 新）不重复
  - 4 份总和需 1600；VRS 剩余 < 1600 的类型，3 份新文件都填空数组
  - 同款 json 结构（name/seed/created_at/sampling/benchmarks.vrs_vqa.ids）

用法：
  python gen_lever_k400_vqa_split.py
  python gen_lever_k400_vqa_split.py --base-seed 9001
  python gen_lever_k400_vqa_split.py --dry-run   # 只算不算不写
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

VRS_PATH = Path('/home/admin1/models/VRSBench_EVAL_vqa.json')
ROOT = Path('/home/admin1/projects/remote_vlm_eval/annotation_data/sampled_eval/lever_test2')
OLD_PATH = ROOT / 'lever_k=400_vqa.json'
NEW_PATHS = [ROOT / f'lever_k=400_vqa_{i}.json' for i in (1, 2, 3)]

TYPES_ORDER = [
    'object existence', 'object quantity', 'object position',
    'object category', 'object color', 'scene type',
    'object shape', 'image', 'object size',
    'reasoning', 'object direction', 'rural or urban',
]
N_PER_TYPE_PER_FILE = 400
N_FILES_NEW = 3
N_FILES_TOTAL = 4  # 1 老 + 3 新


def load_vrs():
    recs = json.load(open(VRS_PATH))
    by_type = defaultdict(list)
    for r in recs:
        by_type[r['type']].append(r)
    return recs, by_type


def load_old_used_ids():
    """从老的 lever_k=400_vqa.json 读出已用 question_id 集合。"""
    if not OLD_PATH.exists():
        return set()
    data = json.load(open(OLD_PATH))
    return set(data['benchmarks']['vrs_vqa']['ids'])


def plan(by_type, used_ids):
    """对每个类型算：
        - VRS 总量
        - 老文件已用数
        - 剩余可抽
        - 是否够 4 份 (1600)
        - 3 份新文件各抽几题
    """
    plan_rows = []
    for t in TYPES_ORDER:
        all_t = by_type.get(t, [])
        vrs_n = len(all_t)
        used_n = sum(1 for r in all_t if r['question_id'] in used_ids)
        remain = vrs_n - used_n
        need_4 = N_PER_TYPE_PER_FILE * N_FILES_TOTAL  # 1600
        can_4 = remain >= need_4
        if can_4:
            per_file = [N_PER_TYPE_PER_FILE] * N_FILES_NEW  # 3 份各 400
        else:
            per_file = [0] * N_FILES_NEW  # 全空
        plan_rows.append({
            'type': t,
            'vrs_n': vrs_n,
            'used_n': used_n,
            'remain': remain,
            'need_4': need_4,
            'can_4': can_4,
            'per_file': per_file,
        })
    return plan_rows


def sample_ids(by_type, used_ids, plan_rows, base_seed):
    """对每个类型抽 3 份的 ids。
    抽法：先把 VRS 该类型所有 id 减去老已用 → 候选池；按种子打乱；分 3 段。
    """
    sampled = []
    for row in plan_rows:
        t = row['type']
        candidates = [r['question_id'] for r in by_type[t]
                      if r['question_id'] not in used_ids]
        rng = random.Random(base_seed + hash(t) % 100000)
        rng.shuffle(candidates)

        if not row['can_4']:
            sampled.append((t, [[], [], []]))
            continue

        # can_4 时 3 份各 400 题
        n1 = N_PER_TYPE_PER_FILE
        n2 = N_PER_TYPE_PER_FILE
        n3 = N_PER_TYPE_PER_FILE
        part1 = candidates[0:n1]
        part2 = candidates[n1:n1 + n2]
        part3 = candidates[n1 + n2:n1 + n2 + n3]
        sampled.append((t, [part1, part2, part3]))

    return sampled


def check_disjoint(sampled, used_ids):
    """校验 4 份（老 + 3 新）之间不重复。"""
    counts = Counter()
    counts.update(used_ids)
    for t, parts in sampled:
        for i, ids in enumerate(parts):
            # 每份内的 ids 互不重复
            dups_in_file = [qid for qid, c in Counter(ids).items() if c > 1]
            if dups_in_file:
                raise ValueError(f"type={t} file={i+1} has internal dup: {dups_in_file[:3]}")
            counts.update(ids)

    dups = [qid for qid, c in counts.items() if c > 1]
    if dups:
        raise ValueError(f"4 份之间有重复 ({len(dups)} 个): {dups[:5]}")
    return True


def write_files(sampled, base_seed, dry_run=False):
    today = datetime.now().strftime('%Y-%m-%d')
    for i, new_path in enumerate(NEW_PATHS):
        ids_i = []
        for t, parts in sampled:
            ids_i.extend(parts[i])
        meta = {
            'name': f'lever_k=400_vqa_{i+1}',
            'seed': base_seed + i,
            'created_at': today,
            'sampling': f'non-replacement per type, {N_PER_TYPE_PER_FILE}/type, {N_FILES_NEW} files, disjoint with lever_k=400_vqa',
            'benchmarks': {
                'vrs_vqa': {
                    'source': str(VRS_PATH),
                    'select_by': 'question_id',
                    'ids': ids_i,
                },
            },
        }
        if dry_run:
            print(f'[DRY-RUN] {new_path}  ids={len(ids_i)}')
        else:
            new_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
            print(f'[WROTE] {new_path}  ids={len(ids_i)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-seed', type=int, default=9001)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    recs, by_type = load_vrs()
    used_ids = load_old_used_ids()
    print(f'VRS 总数: {len(recs)}')
    print(f'老 lever_k=400_vqa.json 已用: {len(used_ids)}')

    plan_rows = plan(by_type, used_ids)
    print('\n=== 抽样计划 ===')
    print(f'{"type":30}  {"VRS":>5}  {"已用":>5}  {"剩":>5}  {"够4份":>6}  per_file')
    for row in plan_rows:
        per = '/'.join(str(x) for x in row['per_file'])
        print(f"{row['type']:30}  {row['vrs_n']:>5}  {row['used_n']:>5}  "
              f"{row['remain']:>5}  {'YES' if row['can_4'] else 'NO':>6}  {per}")

    sampled = sample_ids(by_type, used_ids, plan_rows, args.base_seed)
    print('\n=== 抽样结果 ===')
    total_each = [0, 0, 0]
    for t, parts in sampled:
        for i, ids in enumerate(parts):
            total_each[i] += len(ids)
        sizes = '/'.join(str(len(p)) for p in parts)
        print(f'  {t:30}  {sizes}')

    print(f'\n每份总题数: {total_each}  (期望: [2800, 2800, 2800])')

    try:
        check_disjoint(sampled, used_ids)
        print('✓ 4 份不重复 (老 + 3 新) 检查通过')
    except ValueError as e:
        print(f'✗ 检查失败: {e}')
        return

    write_files(sampled, args.base_seed, dry_run=args.dry_run)

    if not args.dry_run:
        # 总览
        print('\n=== 生成完成 ===')
        for p in NEW_PATHS:
            d = json.load(open(p))
            ids = d['benchmarks']['vrs_vqa']['ids']
            tps = Counter()
            vrs_by = {r['question_id']: r for r in recs}
            for qid in ids:
                if qid in vrs_by:
                    tps[vrs_by[qid]['type']] += 1
            print(f'\n{p.name}  ({len(ids)} 题):')
            for t in TYPES_ORDER:
                print(f'  {t:30}  {tps.get(t, 0):>4}')


if __name__ == '__main__':
    main()
