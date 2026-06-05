"""
Qwen3.5-4B 适配器（thinkOFF，但 17.4% 旧数据 pred 包含思考）
"""
from .base import BaseAdapter


class Qwen35_4BAdapter(BaseAdapter):
    MODEL_NAME = 'qwen3.5-4B'
    display_name = 'Qwen3.5-4B'
    family = 'Qwen3.5'
    has_thinking = False  # 实际是 thinkOFF

    def load_model(self, model_path, device='cuda:0'):
        # 实际加载逻辑（Qwen3.5 还没正式发布，目前用 Qwen2.5 兼容接口）
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer

    def infer(self, model, processor, image, prompt, max_new_tokens=128):
        # TODO: Qwen3.5 多模态
        import torch
        inputs = processor(text=prompt, return_tensors='pt').to(model.device)
        input_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        out_tokens = gen.shape[1] - input_len
        raw = processor.batch_decode(gen[:, input_len:], skip_special_tokens=False)[0]
        return {'out_text': raw, 'out_tokens': out_tokens, 'raw': raw, 'input_tokens': input_len}

    def strip_thinking(self, text: str) -> str:
        # thinkOFF 模式默认无思考，但兼容老数据中残留的思考段
        if not text:
            return ''
        import re
        s = text.strip()
        # 去除明显的"思考段"标记
        if re.match(r'^(okay,|ok,|so,|got it,|alright,|let\'s)', s, re.IGNORECASE):
            # 提取"Final answer:" / "The answer is" / "**answer**"
            for marker in [r'\*\*answer:\*\*\s*([^\n]+)',
                           r'\*\*answer\*\*\s*([^\n]+)',
                           r'final answer[:\s]+([^\n]+)',
                           r'the answer is[:\s]+([^\n]+)',
                           r'the most direct answer is[:\s]+"?([^\n\."]+)']:
                m = re.search(marker, s, re.IGNORECASE)
                if m:
                    return m.group(1).strip().strip('".,;:!?')
        return s
