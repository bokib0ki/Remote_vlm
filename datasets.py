"""
数据集加载 — 读取采样文件/完整标注，统一返回记录列表。

每条记录包含评测所需字段：image_path, question, ground_truth, choices 等。
"""
import json
from pathlib import Path

from config import (
    ROOT, VRS_IMG_DIR, LEVIR_DIR, MME_RS_DIR,
    SAMP,
)


def load_vrs_caption(sample_dir=None, full=False):
    """
    加载 VRSBench Caption 数据集。

    完整标注：VRSBench_EVAL_Cap.json (4.6MB, ~877 条)
    采样文件：sampled_eval/batch1/vrs_cap_sampled.json (50 条)

    每个记录字段: image_id, ground_truth
    """
    if sample_dir or (not full):
        path = (sample_dir or SAMP) / 'vrs_cap_sampled.json'
        if not path.exists():
            path = ROOT / 'sampled_eval' / 'batch1' / 'vrs_cap_sampled.json'
    else:
        path = ROOT / 'VRSBench_EVAL_Cap.json'
    with open(path) as f:
        anns = json.load(f)
    return anns


def load_vrs_vqa(sample_dir=None, full=False):
    """
    加载 VRSBench VQA 数据集。

    完整标注：VRSBench_EVAL_vqa.json (9.0MB, ~1958 条)
    采样文件：vrs_vqa_sampled.json (50 条)

    每个记录字段: image_id, question, ground_truth/answer, type/question_type
    """
    if sample_dir or (not full):
        path = (sample_dir or SAMP) / 'vrs_vqa_sampled.json'
        if not path.exists():
            path = ROOT / 'sampled_eval' / 'batch1' / 'vrs_vqa_sampled.json'
    else:
        path = ROOT / 'VRSBench_EVAL_vqa.json'
    with open(path) as f:
        anns = json.load(f)
    return anns


def load_mme_rs(sample_dir=None, full=False):
    """
    加载 MME-RealWorld Remote Sensing 子集。

    完整标注：mme_rs_annotations.json (2.1MB, ~1265 条 RS 问题)
    采样文件：mme_rs_sampled.json (50 条)

    注意：MME-RS 的 "Image" 字段是相对路径，需拼接 MME_RS_DIR 取文件名。
    """
    if sample_dir or (not full):
        path = (sample_dir or SAMP) / 'mme_rs_sampled.json'
        if not path.exists():
            path = ROOT / 'sampled_eval' / 'batch1' / 'mme_rs_sampled.json'
    else:
        path = ROOT / 'mme_rs_annotations.json'
    with open(path) as f:
        anns = json.load(f)
    return anns


def load_xlrs(sample_dir=None, full=False):
    """
    加载 XLRS-Bench 数据集。

    完整标注：xlrs_arrow/xlrs_samples_42.json (39KB, 42 条)
    采样文件：xlrs_full.json (42 条，全量)

    每个记录字段: question, multi-choice options, answer, local_image, path
    注意：图片是超高分辨率（~2500x2500）卫星图。
    """
    if sample_dir or (not full):
        path = (sample_dir or SAMP) / 'xlrs_full.json'
        if not path.exists():
            path = ROOT / 'sampled_eval' / 'batch1' / 'xlrs_full.json'
    else:
        path = ROOT / 'xlrs_arrow' / 'xlrs_samples_42.json'
    with open(path) as f:
        anns = json.load(f)
    return anns


def load_levir_cc(sample_dir=None, full=False):
    """
    加载 LEVIR-CC 变化描述数据集。

    完整标注：levircc_data/extracted/LevirCCcaptions.json (~2135 条 test)
    采样文件：levir_cc_sampled.json (50 条, 含 "images" 外壳)

    每个记录字段: filename, sentences[{raw}], filepath='test'
    双图路径：LEVIR_DIR/A/{filename} (before) + LEVIR_DIR/B/{filename} (after)
    """
    if sample_dir or (not full):
        path = (sample_dir or SAMP) / 'levir_cc_sampled.json'
        if not path.exists():
            path = ROOT / 'sampled_eval' / 'batch1' / 'levir_cc_sampled.json'
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            anns = [a for a in data.get('images', data) if a.get('filepath') == 'test']
            if not anns:
                anns = data.get('images', data)
            return anns
    # fallback: 从完整标注加载
    from config import LEVIR_ANN_PATH
    if LEVIR_ANN_PATH and LEVIR_ANN_PATH.exists():
        from datasets import load_levir as _ll
        return _ll(full=True)
    return []


# ─── 图片路径解析 ──────────────────────────────────────────

def get_vrs_image(image_id: str) -> str:
    """返回 VRSBench 图片的绝对路径。"""
    return str(VRS_IMG_DIR / image_id)


def get_levir_image(filename: str, which: str = 'A') -> str:
    """返回 LEVIR-CC 图片路径。which = 'A'(before) 或 'B'(after)。"""
    return str(LEVIR_DIR / which / filename)


def get_mme_image(img_path_str: str) -> str:
    """返回 MME-RS 图片的绝对路径（文件名匹配）。"""
    from pathlib import Path
    return str(MME_RS_DIR / Path(img_path_str).name)
