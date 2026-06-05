"""
model_adapter 包 — 按模型分类的加载、推理、pred 裁剪规则

每个模型对应一个 .py 文件，提供：
  - MODEL_NAME: 唯一标识（与目录名匹配）
  - display_name: 报表里的友好名
  - family: 模型家族
  - has_thinking: thinkON 模型
  - AdapterClass: 实际类（继承自 BaseAdapter）

使用：
    from model_adapter import get_adapter
    adapter = get_adapter('qwen3-vl-4B-thinking')
    cleaned = adapter.strip_thinking(text)
"""
from .base import BaseAdapter

# 显式列出所有 adapter 子模块（确保 import）
from . import (
    qwen3_vl_4b_thinking,
    qwen3_vl_4b,
    qwen3_5_4b,
    qwen3_5_2b,
    qwen2_5_vl_3b,
    minicpm_v_4_6,
    gemma_4_e4b,
)

ADAPTERS = {}


def register(mod):
    """从子模块找继承自 BaseAdapter 的类，注册到 ADAPTERS"""
    cls = getattr(mod, 'AdapterClass', None)
    if cls is None:
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if isinstance(attr, type) and issubclass(attr, BaseAdapter) and attr is not BaseAdapter:
                cls = attr
                break
    if cls is None:
        raise AttributeError(f'模块 {mod.__name__} 找不到 BaseAdapter 子类')
    if not hasattr(cls, 'MODEL_NAME') or not cls.MODEL_NAME:
        raise AttributeError(f'类 {cls.__name__} 未设置 MODEL_NAME 类属性')
    name = cls.MODEL_NAME
    if name in ADAPTERS:
        raise ValueError(f'重复注册: {name}')
    ADAPTERS[name] = cls
    return cls


for _mod in [qwen3_vl_4b_thinking, qwen3_vl_4b, qwen3_5_4b, qwen3_5_2b,
             qwen2_5_vl_3b, minicpm_v_4_6, gemma_4_e4b]:
    register(_mod)


def get_adapter(model_name: str) -> BaseAdapter:
    """根据 model 标识（目录名前缀）找 adapter 类"""
    for key, cls in ADAPTERS.items():
        if model_name.startswith(key):
            return cls()
    raise KeyError(f'找不到 adapter: {model_name!r}（已知: {list(ADAPTERS.keys())}）')


def list_adapters():
    return list(ADAPTERS.keys())


__all__ = ['ADAPTERS', 'get_adapter', 'list_adapters', 'register', 'BaseAdapter']
