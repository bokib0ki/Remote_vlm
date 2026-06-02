"""
Benchmark Configuration Loader
=============================
从 bench_cofig/ 目录加载各 benchmark 的标准化配置。
每个 benchmark 包含一个 config.py 定义其官方评测方式。
"""
import importlib.util
import sys
from pathlib import Path
from typing import Any

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.resolve()
BENCH_CONFIG_DIR = PROJECT_ROOT / "bench_cofig"

# 支持的 benchmark 列表
BENCHMARK_MAP = {
    # VRSBench 系列
    "vrs_vqa": "VRSBench/vrs_vqa",
    "vrs_caption": "VRSBench/vrs_caption",
    # 其他 benchmark
    "mme_rs": "MME_RS",
    "levir_cc": "LEVIR_CC",
    "xlrs": "XLRS",
}


def load_bench_config(bench_name: str) -> dict:
    """
    加载指定 benchmark 的配置。
    
    Args:
        bench_name: benchmark 名称（如 'vrs_vqa', 'mme_rs'）
        
    Returns:
        配置字典，包含:
        - task_type: 任务类型
        - prompt_template: prompt 模板
        - answer_extract: 答案提取方式
        - eval_method: 评测方法
        - max_new_tokens_think_off: thinkOFF 最大生成长度
        - max_new_tokens_think_on: thinkON 最大生成长度
        - temperature: 温度（应为 0.0）
        - metrics: 评测指标
        - annotation_file: 标注文件路径
        - image_dir: 图片目录
    """
    if bench_name not in BENCHMARK_MAP:
        raise ValueError(
            f"Unknown benchmark: {bench_name}. "
            f"Available: {list(BENCHMARK_MAP.keys())}"
        )
    
    config_path = BENCH_CONFIG_DIR / BENCHMARK_MAP[bench_name] / "config.py"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    # 动态加载配置模块
    spec = importlib.util.spec_from_file_location(f"bench_config_{bench_name}", config_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"bench_config_{bench_name}"] = module
    spec.loader.exec_module(module)
    
    # 提取配置
    cfg = {
        # 任务类型
        "task_type": getattr(module, "TASK_TYPE", None),
        # Prompt
        "prompt_template": getattr(module, "PROMPT_TEMPLATE", None),
        "prompt_variables": getattr(module, "PROMPT_VARIABLES", ["question", "choices"]),
        # 答案提取
        "answer_extract": getattr(module, "ANSWER_EXTRACT", "raw"),
        "choices_pattern": getattr(module, "CHOICES_PATTERN", "ABCDE"),
        # 评测方法
        "eval_method": getattr(module, "EVAL_METHOD", None),
        # 指标
        "metrics": getattr(module, "METRICS", []),
        "caption_metrics": getattr(module, "CAPTION_METRICS", []),
        # 生成长度
        "max_new_tokens_think_off": getattr(module, "MAX_NEW_TOKENS_THINK_OFF", None),
        "max_new_tokens_think_on": getattr(module, "MAX_NEW_TOKENS_THINK_ON", None),
        # 温度
        "temperature": getattr(module, "TEMPERATURE", 0.0),
        # 数据源
        "annotation_file": getattr(module, "ANNOTATION_FILE", None),
        "image_dir": getattr(module, "IMAGE_DIR", None),
        "image_dir_before": getattr(module, "IMAGE_DIR_BEFORE", None),
        "image_dir_after": getattr(module, "IMAGE_DIR_AFTER", None),
        # 额外选项
        "substring_match": getattr(module, "SUBSTRING_MATCH", True),
        "use_gpt_judge": getattr(module, "USE_GPT_JUDGE", False),
        "use_clair": getattr(module, "USE_CLAIR", False),
        "clair_model": getattr(module, "CLAIR_MODEL", "gpt-4o-mini"),
    }
    
    return cfg


def build_prompt(bench_name: str, record: dict, cfg: dict = None) -> str:
    """
    根据 benchmark 配置和记录构建 prompt。
    
    Args:
        bench_name: benchmark 名称
        record: 数据记录（包含 question, choices 等字段）
        cfg: 可选，预先加载的配置
        
    Returns:
        格式化后的 prompt
    """
    if cfg is None:
        cfg = load_bench_config(bench_name)
    
    template = cfg.get("prompt_template", "")
    
    # 准备模板变量
    variables = cfg.get("prompt_variables", ["question", "choices"])
    ctx = {}
    for var in variables:
        if var == "question":
            # 尝试多个可能的字段名
            ctx["question"] = (
                record.get("question") or 
                record.get("Text") or 
                record.get("text", "")
            )
        elif var == "choices":
            # 尝试多个可能的字段名和格式
            raw_choices = (
                record.get("choices") or 
                record.get("Answer choices") or 
                record.get("multi-choice options") or 
                []
            )
            if isinstance(raw_choices, list):
                ctx["choices"] = "\n".join(raw_choices)
            else:
                ctx["choices"] = str(raw_choices)
        else:
            ctx[var] = record.get(var, "")
    
    # 格式化
    try:
        prompt = template.format(**ctx)
    except KeyError as e:
        # 如果缺少变量，使用原始模板
        prompt = template
    
    return prompt


def get_max_new_tokens(bench_name: str, enable_thinking: bool, cfg: dict = None) -> int:
    """
    获取最大生成长度。
    """
    if cfg is None:
        cfg = load_bench_config(bench_name)
    
    if enable_thinking:
        default = cfg.get("max_new_tokens_think_on")
        return default if default is not None else 4096
    else:
        return cfg.get("max_new_tokens_think_off", 64)


def check_temperature(cfg: dict) -> bool:
    """
    检查温度配置是否合规（应为 0.0）。
    
    Returns:
        True 如果温度正确设置为 0.0
    """
    temp = cfg.get("temperature", None)
    if temp is None:
        return False
    return abs(temp - 0.0) < 1e-6


def get_all_benchmarks() -> list:
    """返回所有可用的 benchmark 名称。"""
    return list(BENCHMARK_MAP.keys())