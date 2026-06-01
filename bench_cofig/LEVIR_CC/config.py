"""
LEVIR-CC Benchmark Configuration
===============================
官方评测方式：变化描述（双图输入），BLEU/ROUGE/CIDER

官方Prompt格式
"""

# 评测任务类型
TASK_TYPE = "change_detection"  # vqa | caption | multi_choice | change_detection

# Prompt模板 - LEVIR-CC官方（双图输入）
PROMPT_TEMPLATE = (
    "Describe the changes between these two remote sensing images "
    "taken at different times."
)

# 答案后处理
ANSWER_EXTRACT = "raw"  # substring | letter | exact | raw

# 评测方法
EVAL_METHOD = "caption_metrics"  # caption_metrics | clair_gpt

# Caption指标（多参考）
CAPTION_METRICS = ["bleu1", "bleu2", "bleu3", "bleu4", "rouge_l", "cider"]

# 最大生成长度
# thinkOFF: 128 tokens
# thinkON: 用户指定
MAX_NEW_TOKENS_THINK_OFF = 128
MAX_NEW_TOKENS_THINK_ON = None

# 温度
TEMPERATURE = 0.0  # 严格=0

# 数据源
ANNOTATION_FILE = "annotation_data/full/LevirCCcaptions.json"
IMAGE_DIR_BEFORE = "levircc_data/extracted/images/test/A"  # before image
IMAGE_DIR_AFTER = "levircc_data/extracted/images/test/B"   # after image

# 指标
METRICS = ["bleu4", "rouge_l", "cider"]