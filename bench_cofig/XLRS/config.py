"""
XLRS-Bench Benchmark Configuration
=================================
官方评测方式：超高分辨率多选VQA，准确率

官方Prompt格式
"""

# 评测任务类型
TASK_TYPE = "multi_choice"  # vqa | caption | multi_choice | change_detection

# Prompt模板 - XLRS-Bench官方格式
PROMPT_TEMPLATE = (
    "{question}\n"
    "The choices are listed below:\n"
    "{choices}\n"
    "Select the best answer for the multiple-choice question based on the image. "
    "Only respond with the letter corresponding to the correct answer (A, B, C, D).\n"
    "The answer is:"
)

# 答案后处理 - 提取选项字母
ANSWER_EXTRACT = "letter"  # substring | letter | exact | raw
CHOICES_PATTERN = "ABCD"

# 评测方法
EVAL_METHOD = "accuracy"  # accuracy | substring | llm_judge

# 最大生成长度
# thinkOFF: 10 tokens（单字母）
# thinkON: 用户指定
MAX_NEW_TOKENS_THINK_OFF = 10
MAX_NEW_TOKENS_THINK_ON = None

# 温度
TEMPERATURE = 0.0  # 严格=0

# 数据源
ANNOTATION_FILE = "annotation_data/full/xlrs_samples_42.json"
IMAGE_DIR = "xlrs_arrow/images"

# 指标
METRICS = ["accuracy"]