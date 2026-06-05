"""
Qwen2.5-VL-3B 适配器（thinkOFF）
"""
from .base import BaseAdapter


class Qwen25VL3BAdapter(BaseAdapter):
    MODEL_NAME = 'qwen2.5-vl-3B'
    display_name = 'Qwen2.5-VL-3B'
    family = 'Qwen2.5-VL'
    has_thinking = False

    def load_model(self, model_path, device='cuda:0'):
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=device
        )
        processor = AutoProcessor.from_pretrained(model_path)
        return model, processor

    def infer(self, model, processor, image, prompt, max_new_tokens=128):
        import torch
        messages = [{
            'role': 'user',
            'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': prompt}],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors='pt').to(model.device)
        input_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        out_tokens = gen.shape[1] - input_len
        raw = processor.batch_decode(gen[:, input_len:], skip_special_tokens=False)[0]
        return {'out_text': raw, 'out_tokens': out_tokens, 'raw': raw, 'input_tokens': input_len}

    def strip_thinking(self, text: str) -> str:
        return text.strip()
