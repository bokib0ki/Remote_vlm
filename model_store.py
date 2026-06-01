import json, time, hashlib
from pathlib import Path

from config import MODEL_SAVE_DIR, LEVIR_DIR
from inference import strip_thinking


BENCH_LABEL_TO_KEY = {
    'VRS-VQA': 'VRS_VQA',
    'VRS-Caption': 'VRS_CAPTION',
    'MME-RS': 'MME_RS',
    'LEVIR-CC': 'LEVIR_CC',
    'XLRS': 'XLRS',
}

INDEX_SOURCES = {
    'VRS_VQA': '/home/admin1/models/VRSBench_EVAL_vqa.json',
    'VRS_CAPTION': '/home/admin1/models/vrsbench_cap.json',
    'MME_RS': '/home/admin1/models/mme_rs_annotations.json',
    'XLRS': '/home/admin1/models/sampled_eval/full/xlrs_full.json',
}


def _now_iso():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8', errors='ignore')).hexdigest()[:12]


def _read_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _index_path(bench_key: str) -> Path:
    return MODEL_SAVE_DIR / '_index' / f'{bench_key}.json'


def ensure_index(bench_key: str):
    ip = _index_path(bench_key)
    if ip.exists():
        return

    if bench_key == 'LEVIR_CC':
        _build_levir_index(ip)
        return

    src = INDEX_SOURCES.get(bench_key)
    if not src:
        _write_json(ip, {
            'schema_version': 1,
            'benchmark': bench_key,
            'created_at': _now_iso(),
            'source': None,
            'id_field': None,
            'items': {},
        })
        return

    data = _read_json(Path(src))
    items = {}
    if bench_key == 'VRS_VQA':
        for r in data:
            qid = r.get('question_id')
            if qid is None:
                continue
            items[str(qid)] = {
                'image_id': r.get('image_id', ''),
                'question': r.get('question', ''),
                'ground_truth': r.get('ground_truth', ''),
                'type': r.get('type', ''),
            }
        id_field = 'question_id'
    elif bench_key == 'VRS_CAPTION':
        for r in data:
            qid = r.get('question_id')
            if qid is None:
                continue
            items[str(qid)] = {
                'image_id': r.get('image_id', ''),
                'question': r.get('question', ''),
                'ground_truth': r.get('ground_truth', ''),
                'type': r.get('type', ''),
            }
        id_field = 'question_id'
    elif bench_key == 'MME_RS':
        for r in data:
            qid = r.get('Question_id')
            if not qid:
                continue
            items[str(qid)] = {
                'image': r.get('Image', ''),
                'text': r.get('Text', ''),
                'ground_truth': r.get('Ground truth', ''),
                'category': r.get('Category', ''),
                'subtask': r.get('Subtask', ''),
                'task': r.get('Task', ''),
            }
        id_field = 'Question_id'
    elif bench_key == 'XLRS':
        for r in data:
            qid = r.get('idx')
            if qid is None:
                continue
            items[str(qid)] = {
                'path': r.get('path', ''),
                'question': r.get('question', ''),
                'answer': r.get('answer', ''),
                'category': r.get('category', ''),
            }
        id_field = 'idx'
    else:
        id_field = None

    _write_json(ip, {
        'schema_version': 1,
        'benchmark': bench_key,
        'created_at': _now_iso(),
        'source': src,
        'id_field': id_field,
        'items': items,
    })


def _build_levir_index(ip: Path):
    items = {}
    ann_map = {}
    ann_path = Path('/home/admin1/models/sampled_eval/full/levir_cc_sampled.json')
    if ann_path.exists():
        data = _read_json(ann_path)
        anns = data.get('images', data)
        for a in anns:
            fname = a.get('filename')
            if fname:
                ann_map[fname] = a

    a_dir = LEVIR_DIR / 'A'
    if a_dir.exists():
        for fp in sorted(a_dir.glob('*.*')):
            fname = fp.name
            a = ann_map.get(fname, {})
            items[fname] = {
                'filename': fname,
                'split': a.get('split', 'test'),
                'changeflag': a.get('changeflag'),
                'sentences': [s.get('raw', '').strip() for s in a.get('sentences', []) if s.get('raw')],
            }

    _write_json(ip, {
        'schema_version': 1,
        'benchmark': 'LEVIR_CC',
        'created_at': _now_iso(),
        'source': str(a_dir) if a_dir.exists() else None,
        'id_field': 'filename',
        'items': items,
    })


def _model_file(model_name: str, bench_key: str) -> Path:
    return MODEL_SAVE_DIR / model_name / f'{bench_key}.json'


def _load_or_init_model_doc(model_name: str, bench_key: str):
    ensure_index(bench_key)
    fp = _model_file(model_name, bench_key)
    if fp.exists():
        return _read_json(fp), fp
    return {
        'schema_version': 1,
        'benchmark': bench_key,
        'model': model_name,
        'index_ref': str(Path('..') / '_index' / f'{bench_key}.json'),
        'runs': [],
        'records': {},
    }, fp


def save_run_from_raw_records(model_name: str, run_meta: dict, raw_records: list[dict]):
    grouped = {}
    for r in raw_records:
        bench_key = BENCH_LABEL_TO_KEY.get(r.get('benchmark', ''), '')
        if not bench_key:
            continue
        grouped.setdefault(bench_key, []).append(r)

    for bench_key, recs in grouped.items():
        save_run(model_name, bench_key, run_meta, recs)


def save_run(model_name: str, bench_key: str, run_meta: dict, recs: list[dict]):
    doc, fp = _load_or_init_model_doc(model_name, bench_key)

    run_id = run_meta.get('run_id')
    if not run_id:
        seed = f"{model_name}|{bench_key}|{run_meta.get('config_tag','')}|{run_meta.get('ended_at','')}|{len(recs)}"
        run_id = f"run_{_sha1(seed)}"

    run_entry = {
        'run_id': run_id,
        'config_tag': run_meta.get('config_tag'),
        'thinking': run_meta.get('thinking'),
        'max_new': run_meta.get('max_new'),
        'selection': run_meta.get('selection'),
        'started_at': run_meta.get('started_at'),
        'ended_at': run_meta.get('ended_at'),
        'saved_at': _now_iso(),
        'n_records': len(recs),
    }
    doc['runs'].append(run_entry)

    store = doc.setdefault('records', {})
    for r in recs:
        rid = r.get('_idx')
        if rid is None:
            rid = r.get('id')
        if rid is None:
            continue
        rid = str(rid)

        out = r.get('pred', '')
        out_raw = r.get('pred_raw', None)
        if isinstance(out, str) and ('<think' in out.lower() or '</think' in out.lower()):
            out = strip_thinking(out)
        if isinstance(out_raw, str) and ('<think' in out_raw.lower() or '</think' in out_raw.lower()):
            out_raw = strip_thinking(out_raw)

        entry = {
            'run_id': run_id,
            'created_at': run_meta.get('ended_at') or _now_iso(),
            'time_s': r.get('time_s'),
            'tokens': r.get('tokens'),
            'output': out,
            'output_raw': out_raw,
        }
        for k in ('gt', 'question', 'image_id', 'correct'):
            if k in r:
                entry[k] = r.get(k)

        store.setdefault(rid, []).append(entry)

    _write_json(fp, doc)

