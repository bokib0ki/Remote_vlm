"""
MiniCPM-V 4.6 适配器（thinkOFF）
"""
from .base import BaseAdapter


class MiniCPMV46Adapter(BaseAdapter):
    MODEL_NAME = 'minicpm-v-4.6'
    display_name = 'MiniCPM-V 4.6'
    family = 'MiniCPM-V'
    has_thinking = False

    def load_model(self, model_path, device='cuda:0'):
        import torch
        from transformers import AutoModel, AutoTokenizer
        # MiniCPM-V 4.6 用 AutoModel 加载
        model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=device
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return model, tokenizer

    def infer(self, model, processor, image, prompt, max_new_tokens=128):
        # MiniCPM-V 推理接口不同（chat 调用）
        import torch
        msgs = [{'role': 'user', 'content': [image, prompt]}]
        # 实际推理: model.chat(image=..., msgs=..., tokenizer=..., max_new_tokens=...)
        res = model.chat(image=image, msgs=msgs, tokenizer=processor, max_new_tokens=max_new_tokens)
        out_text = res if isinstance(res, str) else str(res)
        return {
            'out_text': out_text,
            'out_tokens': len(processor.encode(out_text)),
            'raw': out_text,
            'input_tokens': 0,  # MiniCPM-V chat 接口不返回
            'img_tokens': 0,
        }

    def strip_thinking(self, text: str) -> str:
        return text.strip()
