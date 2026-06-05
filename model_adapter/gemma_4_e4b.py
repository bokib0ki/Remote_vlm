"""
Gemma-4-4B 适配器（thinkOFF）
注意: Gemma 4 实际是闭源，本适配器占位
"""
from .base import BaseAdapter


class Gemma4E4BAdapter(BaseAdapter):
    MODEL_NAME = 'gemma-4-e4b'
    display_name = 'Gemma-4-4B'
    family = 'Gemma'
    has_thinking = False

    def load_model(self, model_path, device='cuda:0'):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # Gemma 4 实际加载方式取决于发布后的接口
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map=device
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception:
            # 兜底：如果 Gemma 4 还没正式发布
            raise NotImplementedError('Gemma 4 加载逻辑待官方发布后补充')
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
        return text.strip()
