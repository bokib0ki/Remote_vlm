"""
model_adapter 公共基类
"""
from typing import Any, Tuple
# from PIL import Image  # 仅在 infer() 中需要，移到子类的局部 import


class BaseAdapter:
    """所有模型 adapter 的基类"""

    # 子类必须填
    MODEL_NAME: str = ''        # 唯一标识（与目录名起始匹配）
    display_name: str = ''      # 报表里的友好名
    family: str = ''            # 模型家族（Qwen3 / Qwen3.5 / Gemma / MiniCPM）
    has_thinking: bool = False  # thinkON 时是否需要 strip_thinking

    def load_model(self, model_path: str, device: str = 'cuda:0') -> Tuple[Any, Any]:
        """加载模型 + processor。子类可重写。"""
        raise NotImplementedError

    def infer(self, model, processor, image, prompt: str,
              max_new_tokens: int = 128) -> dict:
        """推理。返回 {'out_text', 'out_tokens', 'raw', 'input_tokens', 'img_tokens', ...}"""
        raise NotImplementedError

    def strip_thinking(self, text: str) -> str:
        """
        裁剪思考过程，保留最终答案。
        thinkOFF 模型返回原文（不需要 strip）。
        thinkON 模型必须实现。
        """
        return text.strip()

    def post_process_pred(self, text: str) -> str:
        """
        标准化 pred：去 im_end/endoftext/标点/首尾空白。
        所有模型通用。
        """
        if not text:
            return ''
        import re
        # 去特殊 token
        s = re.sub(r'<\|im_(?:end|start)\|>', '', text)
        s = re.sub(r'<\|endoftext\|>', '', s)
        s = re.sub(r'<think>.*?</think>', '', s, flags=re.DOTALL)
        s = re.sub(r'<thinking>.*?</thinking>', '', s, flags=re.DOTALL)
        s = re.sub(r'</?think(?:ing)?>', '', s, flags=re.IGNORECASE)
        # 去首尾标点和空白
        s = s.strip().rstrip('.,;:!?\n\t ')
        return s

    def clean_pred(self, raw_text: str) -> str:
        """
        全流程清洗：先 strip_thinking 再 post_process_pred
        默认实现就是这个流程，子类可重写以定制顺序或逻辑
        """
        return self.post_process_pred(self.strip_thinking(raw_text))
