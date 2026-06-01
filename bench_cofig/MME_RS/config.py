"""
MME-RealWorld RS Benchmark Configuration
========================================
官方评测方式：多选VQA，准确率

官方Prompt格式（来自MME-RealWorld论文）
"""

# 评测任务类型
TASK_TYPE = "multi_choice"  # vqa | caption | multi_choice | change_detection

# Prompt模板 - MME-RealWorld官方格式
PROMPT_TEMPLATE = (
    "{question}\n"
    "The choices are listed below:\n"
    "{choices}\n"
    "Select the best answer to the above multiple-choice question based on the image. "
    "Respond with only the letter (A, B, C, D, or E) of the correct option.\n"
    "The best answer is:"
)

# 答案后处理 - 提取选项字母
ANSWER_EXTRACT = "letter"  # substring | letter | exact | raw
CHOICES_PATTERN = "ABCDE"

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
ANNOTATION_FILE = "annotation_data/full/mme_rs_annotations.json"
IMAGE_DIR = "mme_images/remote_sensing_full/remote_sensing"

# 指标
METRICS = ["accuracy"]