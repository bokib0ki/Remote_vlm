"""
VRSBench VQA Benchmark Configuration
=====================================
官方评测方式：
- L1: ground_truth substring 匹配
- L2: yes/no/数字 精确匹配  
- L3: GPT-4o-mini 语义匹配（同义词、football/soccer等）

源码: bench_cofig/bench_src/VRSBench_src/eval_fianl/eval_vqa_gpt.ipynb
"""

# 评测任务类型
TASK_TYPE = "vqa"  # vqa | caption | multi_choice | change_detection

# Prompt模板 - VRSBench官方VQA prompt
# 源码使用: 直接question + "Answer the question using a single word or phrase."
PROMPT_TEMPLATE = "{question}\nAnswer the question using a single word or phrase."

# 答案后处理
ANSWER_EXTRACT = "substring"  # substring | letter | exact

# 评测方法（官方标准）
EVAL_METHOD = "vrsbench_vqa"  # vrsbench_vqa | substring | llm_judge | exact_match

# L1 substring匹配（VRSBench官方方法）
# 如果 ground_truth (lowercase) in predicted (lowercase) → 正确
SUBSTRING_MATCH = True

# GPT语义评估（用于处理同义词等）
# VRSBench源码: 当substring不匹配且非yes/no/数字时，调用GPT判断语义匹配
USE_GPT_JUDGE = False  # 默认不启用，需要单独跑GPT评估流程

# 最大生成长度
# thinkOFF: 64 tokens（短答）
# thinkON: 用户指定
MAX_NEW_TOKENS_THINK_OFF = 64
MAX_NEW_TOKENS_THINK_ON = None  # 用户指定或默认4096

# 温度
TEMPERATURE = 0.0  # 严格=0

# 数据源
ANNOTATION_FILE = "annotation_data/full/VRSBench_EVAL_vqa.json"
IMAGE_DIR = "vrsbench_images/Images_val"

# 指标定义
METRICS = ["acc_l1"]  # VRSBench官方只用L1准确率

# 源码评测逻辑（来自eval_vqa_gpt.ipynb）:
# if ground_truth in predicted:
#     match_result = '1'
# elif ground_truth in ['yes', 'no'] + list(map(str, range(100))):
#     match_result = '1' if ground_truth == predicted else '0'
# else:
#     match_result = check_match_with_gpt(question, ground_truth, predicted)  # GPT评估