"""
评分指标 — BLEU, ROUGE-L, CIDER。
封装 pycocoevalcap，提供统一接口。
"""
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


def compute_caption_scores(refs_dict: dict, hyps_dict: dict) -> dict:
    """
    计算 caption 类任务的 BLEU-1~4, ROUGE-L, CIDER。

    Args:
        refs_dict: {sample_id: [ref1, ref2, ...]}
        hyps_dict: {sample_id: [hypothesis]}

    Returns:
        {'bleu1': ..., 'bleu4': ..., 'rouge_l': ..., 'cider': ...}
    """
    bs, _ = Bleu(4).compute_score(refs_dict, hyps_dict)
    rouge_s, _ = Rouge().compute_score(refs_dict, hyps_dict)
    cider_s, _ = Cider().compute_score(refs_dict, hyps_dict)
    return {
        **{f'bleu{i+1}': round(bs[i], 4) for i in range(4)},
        'rouge_l': round(rouge_s, 4),
        'cider': round(cider_s, 4),
    }
