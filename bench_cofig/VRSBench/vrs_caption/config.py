"""
VRSBench Caption Benchmark Configuration
=========================================
官方评测方式：
- BLEU-1/2/3/4, ROUGE-L, CIDER
- GPT-based CLAIR评估（用于长caption）

源码: bench_cofig/bench_src/VRSBench_src/eval_fianl/
"""

# 评测任务类型
TASK_TYPE = "caption"  # vqa | caption | multi_choice | change_detection

# Prompt模板 - VRSBench官方Caption prompt
# 源码: "Describe this remote sensing image in detail."
PROMPT_TEMPLATE = "Describe this remote sensing image in detail."

# 答案后处理
ANSWER_EXTRACT = "raw"  # substring | letter | exact | raw

# 评测方法（官方标准）
EVAL_METHOD = "caption_metrics"  # caption_metrics | clair_gpt

# Caption指标
CAPTION_METRICS = ["bleu1", "bleu2", "bleu3", "bleu4", "rouge_l", "cider"]

# CLAIR GPT评估（可选，用于长描述）
USE_CLAIR = False
CLAIR_MODEL = "gpt-4o-mini"

# 最大生成长度
# thinkOFF: 128 tokens
# thinkON: 用户指定
MAX_NEW_TOKENS_THINK_OFF = 128
MAX_NEW_TOKENS_THINK_ON = None  # 用户指定

# 温度
TEMPERATURE = 0.0  # 严格=0

# 数据源
ANNOTATION_FILE = "annotation_data/full/VRSBench_EVAL_Cap.json"
IMAGE_DIR = "vrsbench_images/Images_val"

# 源码评测逻辑（来自compute_metrics.ipynb）:
# 使用pycocoevalcap计算BLEU/ROUGE/CIDER
# CLAIR评分用于长描述评估（可选）