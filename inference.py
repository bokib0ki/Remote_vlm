"""
模型加载和推理。
支持 thinkOFF / thinkON 双模式，支持单图/双图输入。
"""
import re
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from config import DO_SAMPLE


def load_model(model_name: str, model_root: str):
    """加载 VLM 模型和 processor。"""
    path = f"{model_root}/{model_name}"
    model = AutoModelForImageTextToText.from_pretrained(
        path, torch_dtype=torch.bfloat16,
        device_map='auto', trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        path, trust_remote_code=True,
    )
    return model, processor


def infer(
    model, processor,
    img: Image.Image,
    prompt: str,
    max_new_tokens: int = 128,
    extra_imgs: list | None = None,
    enable_thinking: bool = False,
    do_sample: bool = DO_SAMPLE,
) -> dict:
    """
    单次推理。

    Args:
        img: 主图片（PIL RGB）
        prompt: 文本 prompt
        max_new_tokens: 最大生成 token 数
        extra_imgs: 额外图片（如 LEVIR-CC 的 after 图）
        enable_thinking: 是否开启 thinkON 模式

    Returns:
        dict with keys:
            out_text: str       # 最终答案文本（去 think 标签）
            out_tokens: int     # 总输出 token 数
            input_tokens: int   # 总输入 token 数（含图+文）
            img_tokens: int     # 图片 token 数（估算）
            prompt_tokens: int  # 纯文本 prompt token 数
            thinking_tokens: int # 思考 token 数（仅 thinkON）
            answer_tokens: int  # 答案 token 数（仅 thinkON）
            speed: float        # token/s（总 output token / 耗时）
            raw: str            # 原始输出（含 think 标签）
            gen_ids: tensor     # 原始生成果（用于精确 token 计数）
            input_ids: tensor   # 输入 token ids
    """
    def _build_conv(text_prompt: str):
        content = [{"type": "image", "image": img}]
        if extra_imgs:
            for ei in extra_imgs:
                content.append({"type": "image", "image": ei})
        content.append({"type": "text", "text": text_prompt})
        return [{"role": "user", "content": content}]

    conv = _build_conv(prompt)
    try:
        text = processor.apply_chat_template(
            conv,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        if enable_thinking:
            injected = (
                "Think step-by-step. Put your reasoning inside <think>...</think>.\n"
                "After </think>, output ONLY the final answer.\n\n"
                f"{prompt}"
            )
            conv = _build_conv(injected)
        text = processor.apply_chat_template(
            conv,
            add_generation_prompt=True,
            tokenize=False,
        )

    if extra_imgs:
        all_imgs = [img] + extra_imgs
        inputs = processor(images=all_imgs, text=text,
                           return_tensors="pt").to('cuda:0')
    else:
        inputs = processor(images=img, text=text,
                           return_tensors="pt").to('cuda:0')

    input_ids = inputs['input_ids']
    input_len = input_ids.shape[1]

    # 估算 prompt tokens（纯文本，不含图）
    try:
        prompt_only = processor(text=[prompt], return_tensors="pt", add_special_tokens=False)
        prompt_tokens = prompt_only['input_ids'].shape[1]
    except Exception:
        prompt_tokens = 0
    img_tokens = max(input_len - prompt_tokens, 0)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            output_scores=False,
            return_dict_in_generate=True,
        )

    gen = out.sequences[0, input_len:]
    gen_len = gen.shape[0]

    # 统计 thinking / answer token 数
    thinking_tokens = 0
    answer_tokens = 0
    raw_text = processor.decode(gen, skip_special_tokens=False)

    if enable_thinking:
        # 找 <think> 起始位置
        raw_lower = raw_text.lower()
        think_start = raw_lower.find('<think>')
        think_end_tags = ['</think>', '</thinking>']
        think_end = -1
        for tag in think_end_tags:
            idx = raw_lower.rfind(tag)
            if idx != -1:
                think_end = idx + len(tag)
                break

        if think_start != -1 and think_end > think_start:
            think_text = raw_text[think_start:think_end]
            think_ids = processor(thoughts=[think_text], add_special_tokens=False, return_tensors="pt")
            thinking_tokens = think_ids['input_ids'].shape[1]
            # answer 是 <think> 结束后到末尾
            answer_text = raw_text[think_end:].strip()
            if answer_text:
                answer_ids = processor(text=[answer_text], add_special_tokens=False, return_tensors="pt")
                answer_tokens = answer_ids['input_ids'].shape[1]
        else:
            # 没找到 tag，全算 answer
            answer_tokens = gen_len
    else:
        answer_tokens = gen_len

    out_text = strip_thinking(raw_text).strip()

    # speed: 总 output token / 耗时（耗时由 caller 算）
    # 这里先不填，由 caller 计算后填入
    result = {
        'out_text': out_text,
        'out_tokens': gen_len,
        'input_tokens': input_len,
        'img_tokens': img_tokens,
        'prompt_tokens': prompt_tokens,
        'thinking_tokens': thinking_tokens,
        'answer_tokens': answer_tokens,
        'speed': 0.0,  # placeholder，caller 负责填
        'raw': raw_text,
        '_gen_ids': gen,
        '_input_ids': input_ids,
    }
    return result


def strip_thinking(text: str) -> str:
    """清除 thinkON 模式产生的 thinking 块。"""
    if not text:
        return ''
    s = text
    lower = s.lower()
    for close_tag in ('</think>', '</thinking>'):
        idx = lower.rfind(close_tag)
        if idx != -1:
            s = s[idx + len(close_tag):]
            lower = s.lower()
    s = re.sub(r'<think>.*?</think>\s*\n?\s*', '', s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r'<thinking>.*?</thinking>\s*\n?\s*', '', s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r'</?think(?:ing)?>', '', s, flags=re.IGNORECASE)
    return s.strip()


def extract_letter(text: str, choices: str = 'ABCDE') -> str:
    """
    从模型输出中提取选项字母。
    支持多种常见格式：
      - "The best answer is: A"
      - "(B)"
      - "C."
      - "D" (行末独立字母)
    """
    s = text.strip().upper()
    # 精确匹配各种前缀模式
    for pat in [
        r'(?:ANSWER|OPTION)\s+(?:IS|:)\s*[(]?([' + choices + r'])[)]?',
        r'CORRECT\s+(?:ANSWER|OPTION)\s+IS\s+[(]?([' + choices + r'])[)]?',
        r'THE\s+BEST\s+ANSWER\s+IS\s+[(]?([' + choices + r'])[)]?',
        r'IS\s+CORRECT\s*[.:]?\s*[(]?([' + choices + r'])[)]?',
    ]:
        m = re.findall(pat, s)
        if m:
            return m[-1]
    # 括号内字母
    parenthesized = re.findall(r'\(' + choices + r'\)', s)
    if parenthesized:
        return parenthesized[-1].strip('()')
    # 行首/空白后的独立字母
    letters = re.findall(
        r'(?:^|[\s(])([' + choices + r'])(?:\)|\]|\}|\.|,|\s|$)',
        s,
    )
    if letters:
        return letters[-1]
    return ""


def safe_img(path, fallback_dir=None):
    """加载图片，不存在时用占位图代替。"""
    p = Path(path)
    if p.exists():
        return Image.open(p).convert('RGB')
    if fallback_dir:
        fallbacks = sorted(Path(fallback_dir).glob('*.*'))
        if fallbacks:
            return Image.open(fallbacks[0]).convert('RGB')
    return Image.new('RGB', (512, 512), 'gray')
