"""
Qwen3.5-2B 适配器（同 4B，thinkOFF）
"""
from .base import BaseAdapter
import re


class Qwen35_2BAdapter(BaseAdapter):
    MODEL_NAME = 'qwen3.5-2B'
    display_name = 'Qwen3.5-2B'
    family = 'Qwen3.5'
    has_thinking = False

    def load_model(self, model_path, device='cuda:0'):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer

    def infer(self, model, processor, image, prompt, max_new_tokens=128):
        import torch
        inputs = processor(text=prompt, return_tensors='pt').to(model.device)
        input_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        out_tokens = gen.shape[1] - input_len
        raw = processor.batch_decode(gen[:, input_len:], skip_special_tokens=False)[0]
        return {'out_text': raw, 'out_tokens': out_tokens, 'raw': raw, 'input_tokens': input_len}

    def strip_thinking(self, text: str) -> str:
        if not text:
            return ''
        s = text.strip()
        if re.match(r'^(okay,|ok,|so,|got it,|alright,|let\'s)', s, re.IGNORECASE):
            for marker in [r'\*\*answer:\*\*\s*([^\n]+)',
                           r'\*\*answer\*\*\s*([^\n]+)',
                           r'final answer[:\s]+([^\n]+)',
                           r'the answer is[:\s]+([^\n]+)',
                           r'the most direct answer is[:\s]+"?([^\n\."]+)']:
                m = re.search(marker, s, re.IGNORECASE)
                if m:
                    return m.group(1).strip().strip('".,;:!?')
        return s
