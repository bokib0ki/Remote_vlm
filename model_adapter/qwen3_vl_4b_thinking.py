"""
Qwen3-VL-4B-Thinking 适配器

输出格式特征（thinkON）：
  - 思考段以 "Got it, let's" / "So," / "Okay," 开头
  - 中间大量 "Wait, X. Wait, no. Wait, maybe Y." 循环
  - 80% 样本有 "the answer is X" 标记（但 X 之后常跟 "Wait, no"）
  - 3% 样本有 "Therefore, X" 标记
  - 17% 没有任何标记
"""
from .base import BaseAdapter
import re

# 模块级注册信息（register() 在 __init__.py 读这些）
MODEL_NAME = 'qwen3-vl-4B-thinking'
display_name = 'Qwen3-VL-4B-Thinking'
family = 'Qwen3-VL'
has_thinking = True
AdapterClass = None  # 下面赋值


class Qwen3VL4BThinkingAdapter(BaseAdapter):
    MODEL_NAME = MODEL_NAME
    display_name = display_name
    family = family
    has_thinking = True

    def load_model(self, model_path, device='cuda:0'):
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=device
        )
        processor = AutoProcessor.from_pretrained(model_path)
        return model, processor

    def infer(self, model, processor, image, prompt, max_new_tokens=4096):
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
        """
        Qwen3-VL-4B-Thinking 裁剪策略（按优先级）:

        P0. 取 "the answer is X" / "Therefore, X" / "Final answer: X" 标记
            - X 取到下个句子边界前的最短内容
            - 必须跳过 Wait, no / Wait, but / So, the answer is / But the answer is
        P1. 反复纠结模式："Wait, maybe X. Wait, no. Wait, X. Wait, no. Wait, Y."
            - 取**所有** "Wait, [non-Wait-non-no-non-but-non-so] X" 里的 X
            - 投票取最多次出现的 X
        P2. 完整句抽取（"The X is Y" / "X is in the Y" 模式）→ 抽最右边的名词短语
        P3. 取最后 1-2 个非 Wait/So/But 开头的短句
        P4. 兜底：取最后 50 字符
        """
        if not text:
            return ''

        # ─── P0: 找明确的答案标记 ───
        # 匹配 "the answer is X" 形式（X = 最短内容，到 .!?\n 之前）
        answer_markers = [
            # "the answer is X" / "the answer is: X"
            (r'(?<![\w-])(?:the\s+)?answer\s+is[:\s]+([^\n.!?]{1,50})', 'P0'),
            # "Therefore, X" / "Therefore the answer is X"
            (r'therefore[,\s]+(?:the\s+answer\s+is[:\s]+)?([^\n.!?]{1,50})', 'P0'),
            # "So, X" / "So the answer is X"
            (r'\bso[, ]+(?:the\s+answer\s+is[:\s]+)?([^\n.!?]{1,50})', 'P0'),
            # "Final answer: X"
            (r'final\s+answer[:\s]+([^\n.!?]{1,50})', 'P0'),
            # "In summary, X"
            (r'in\s+summary[,\s]+([^\n.!?]{1,50})', 'P0'),
        ]
        candidates = []
        for pat, src in answer_markers:
            for m in re.finditer(pat, text, re.IGNORECASE):
                cand = m.group(1).strip().strip('"\'`.,;:!?')
                # 过滤纯 Wait/So/But 过渡句
                if re.match(r'^(wait|so|but|however|maybe|let|then|hence|although|ok|ok,|ok\.|okay|i think)\b',
                            cand.lower()):
                    continue
                # 过滤明显是反悔的（"Wait, no" / "Wait, but" 之类）
                if cand.lower().startswith(('wait, no', 'wait, but', 'wait, actually')):
                    continue
                candidates.append(cand)

        if candidates:
            # 投票：出现最多的为最终答案
            from collections import Counter
            c = Counter(candidates)
            most_common = c.most_common(1)[0][0]
            return most_common

        # ─── P1: 反复纠结模式（"Wait, X. Wait, no. Wait, Y."）───
        # 找所有 "Wait, X" 后跟 ". Wait, no" / ". Wait, but" 的 X
        # 实际上更简单：取所有 "Wait, (the answer is )?X" 模式的 X
        wait_candidates = []
        for m in re.finditer(r'wait[, ]+(?:the\s+answer\s+is[:\s]+)?([^\n.!?]{1,50})', text, re.IGNORECASE):
            cand = m.group(1).strip().strip('"\'`.,;:!?')
            if cand.lower() in ('no', 'but', 'maybe', 'actually', 'let me think', 'let\'s think'):
                continue
            wait_candidates.append(cand)
        if wait_candidates:
            from collections import Counter
            c = Counter(wait_candidates)
            return c.most_common(1)[0][0]

        # ─── P2: 完整句抽取（"X is Y" 模式）───
        # 找最后几个 "X is/are Y" 短句，Y 是答案
        # 跳过 Wait/So/But 开头
        sents = re.split(r'[\.\!\?\n]+', text)
        sents = [s.strip() for s in sents if s.strip()]
        for sent in reversed(sents[-20:]):  # 只看最后 20 句
            if len(sent) > 80:  # 跳过超长句
                continue
            if re.match(r'^(wait|so|but|however|maybe|let me|let\'s|i think|hmm|ok|ok,|ok\.|okay|alright|got it)',
                        sent.lower()):
                continue
            # 尝试抽 "X is/are Y" 形式的 Y
            m = re.search(r'\b(?:is|are|was|were)\s+(?:a\s+|an\s+|the\s+)?([^\s,]+(?:\s+[^\s,]+){0,3})', sent, re.IGNORECASE)
            if m:
                return m.group(1).strip().strip('"\'`.,;:!?')
            # 没有 is/are → 直接返回整个短句（但限制长度）
            if len(sent) <= 50:
                return sent.strip().strip('"\'`.,;:!?')

        # ─── P3: 兜底取最后短句 ───
        for sent in reversed(sents):
            if 5 < len(sent) <= 50 and not re.match(r'^(wait|so|but|however|maybe)', sent.lower()):
                return sent.strip().strip('"\'`.,;:!?')

        # ─── P4: 兜底取最后 50 字符 ───
        return text.strip()[-50:].strip().strip('"\'`.,;:!?')


# 注册到模块（供 __init__.py 读取）
AdapterClass = Qwen3VL4BThinkingAdapter

