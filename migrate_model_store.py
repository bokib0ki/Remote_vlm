#!/usr/bin/env python3
import json, argparse, time
from pathlib import Path

from config import RAW_DIR_ROOT
from model_store import save_run_from_raw_records
from inference import strip_thinking


def _now_iso():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _parse_model_mode(tag: str):
    if '_thinkON_' in tag:
        model, rest = tag.split('_thinkON_', 1)
        return model, True, rest
    if '_thinkOFF_' in tag:
        model, rest = tag.split('_thinkOFF_', 1)
        return model, False, rest
    return None


def _safe_load(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _iter_raw_outputs(root: Path):
    for fp in root.rglob('raw_outputs.json'):
        yield fp


def _clean_record(rec: dict):
    for k in ('pred', 'pred_raw'):
        v = rec.get(k)
        if isinstance(v, str) and ('<think' in v.lower() or '</think' in v.lower()):
            rec[k] = strip_thinking(v)
    return rec


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(RAW_DIR_ROOT), help='raw_outputs 根目录')
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f'not found: {root}')

    n_files = 0
    n_runs = 0
    for fp in _iter_raw_outputs(root):
        n_files += 1
        tag_dir = fp.parent
        tag = tag_dir.name
        parsed = _parse_model_mode(tag)
        if not parsed:
            continue
        model, thinking, _ = parsed
        mtime = time.gmtime(fp.stat().st_mtime)
        ended_at = time.strftime('%Y-%m-%dT%H:%M:%SZ', mtime)

        raw = _safe_load(fp)
        if not isinstance(raw, list) or not raw:
            continue
        raw = [_clean_record(dict(r)) for r in raw]

        selection = None
        try:
            rel = fp.relative_to(root)
            parts = rel.parts
            if parts and parts[0] != 'batch1' and parts[0] != 'batch2':
                selection = parts[0]
        except Exception:
            pass

        run_meta = {
            'run_id': f'legacy_{ended_at}_{tag}',
            'config_tag': tag,
            'thinking': thinking,
            'max_new': None,
            'selection': selection,
            'started_at': None,
            'ended_at': ended_at,
            'saved_at': _now_iso(),
        }
        save_run_from_raw_records(model, run_meta, raw)
        n_runs += 1

    print(f'migrated_files={n_files} migrated_runs={n_runs}')

