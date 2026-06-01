"""
全局配置 — 路径、模型列表、prompt 模板。
运行前请根据实际环境修改 ROOT。
"""
from pathlib import Path

# ─── 根目录 ───────────────────────────────────────────────
ROOT = Path('/home/admin1/models')

# ─── 模型列表 ─────────────────────────────────────────────
ALL_MODELS = [
    'minicpm-v-4.6',  # MiniCPM-V 4.6B (thinkON 支持)
    'qwen3.5-0.8B',   # Qwen3.5-VL 0.8B (thinkON 支持)
    'qwen3.5-2B',     # Qwen3.5-VL 2B (thinkON 支持)
    'qwen3.5-4B',     # Qwen3.5-VL 4B (thinkON 支持)
    'qwen3-vl-2B',    # Qwen3-VL 2B
    'qwen3-vl-4B',    # Qwen3-VL 4B
    'gemma-4-e2b',    # Gemma-4 2B
    'gemma-4-e4b',    # Gemma-4 4B
]
DUAL_MODELS = [
    'minicpm-v-4.6',
    'qwen3.5-0.8B',
    'qwen3.5-2B',
    'qwen3.5-4B',
    'qwen3-vl-2B',
    'qwen3-vl-4B',
    'gemma-4-e2b',
    'gemma-4-e4b',
]
# ↑ 支持 thinkON/thinkOFF 双模式的模型

# ─── 图片目录 ─────────────────────────────────────────────
VRS_IMG_DIR   = ROOT / 'vrsbench_images' / 'Images_val'        # 9351 张
LEVIR_DIR     = ROOT / 'levircc_data' / 'extracted' / 'images' / 'test'  # 2135 对 (A/B)
MME_RS_DIR    = ROOT / 'mme_images' / 'remote_sensing_full' / 'remote_sensing'  # 1300 张
TEST_IMGS     = sorted((ROOT / 'test_images').glob('*.*'))      # 备用占位图

# ─── 采样文件目录 ──────────────────────────────────────────
# batch1 = 前50样本, batch2 = 后50样本, full = 全部
SAMP = ROOT / 'sampled_eval' / 'batch1'   # 切换时改这里

# ─── 输出目录 ─────────────────────────────────────────────
RAW_DIR_ROOT = ROOT / 'raw_outputs'       # 推理原始输出
RES_DIR      = ROOT / 'results'           # JSON 指标
OUT_DIR      = ROOT / '..' / 'projects'   # Excel 报告
MODEL_SAVE_DIR = ROOT / 'model_save_output'  # 统一的模型输出历史库

# ─── 推理参数 ─────────────────────────────────────────────
MAX_NEW_TOKENS = 15000      # caption/change 任务
MAX_NEW_VQA    = 64        # VQA 短答
DO_SAMPLE      = False     # greedy 解码

# ─── Prompt 模板（官方/论文标准） ────────────────────────

CAPTION_PROMPT = "Describe this remote sensing image in detail."
# VRSBench Caption 官方 prompt

LEVIR_PROMPT = (
    "Describe the changes between these two remote sensing images "
    "taken at different times."
)
# LEVIR-CC 官方 prompt（双图输入）

VRS_VQA_PROMPT_TPL = "{question}\nAnswer the question using a single word or phrase."
# GeoChat / VRSBench 官方 VQA prompt

MME_PROMPT_TPL = (
    "{question}\nThe choices are listed below:\n{choices}\n"
    "Select the best answer to the above multiple-choice question based on the image. "
    "Respond with only the letter (A, B, C, D, or E) of the correct option.\n"
    "The best answer is:"
)
# MME-RealWorld 官方 prompt

XLRS_PROMPT_TPL = (
    "{question}\nThe choices are listed below:\n{choices}\n"
    "Select the best answer for the multiple-choice question based on the image. "
    "Only respond with the letter corresponding to the correct answer (A, B, C, D).\n"
    "The answer is:"
)
# XLRS-Bench 官方 prompt

# ─── 指标定义（Excel 列顺序） ─────────────────────────────
METRICS = [
    'mme_rs', 'xlrs',
    'vrs_cap_bleu4', 'vrs_cap_rouge_l', 'vrs_cap_cider',
    'levir_bleu4', 'levir_rouge_l', 'levir_cider',
    'vrs_vqa',
]
MLABELS = [
    'MME-RS', 'XLRS',
    'VRS-Cap B4', 'VRS-Cap R-L', 'VRS-Cap CIDER',
    'LEVIR B4', 'LEVIR R-L', 'LEVIR CIDER',
    'VRS-VQA',
]
MINFO = {
    'mme_rs':           'MME-RealWorld RS 多选VQA准确率',
    'xlrs':             'XLRS-Bench 超高分辨率多选VQA',
    'vrs_cap_bleu4':    'VRSBench Caption (BLEU-4)',
    'vrs_cap_rouge_l':  'VRSBench Caption (ROUGE-L)',
    'vrs_cap_cider':    'VRSBench Caption (CIDER)',
    'levir_bleu4':      'LEVIR-CC 变化描述 (BLEU-4)',
    'levir_rouge_l':    'LEVIR-CC 变化描述 (ROUGE-L)',
    'levir_cider':      'LEVIR-CC 变化描述 (CIDER)',
    'vrs_vqa':          'VRSBench VQA (L1 substring 准确率)',
}
