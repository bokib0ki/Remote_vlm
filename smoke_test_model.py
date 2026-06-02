#!/usr/bin/env python3
"""单模型冒烟测试：加载 + 推理 1 张图，验证 generate 流程。"""
import os, sys, time
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import torch
from PIL import Image
from pathlib import Path

from inference import load_model, infer, strip_thinking

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else 'qwen3-vl-4B-thinking'
ENABLE_THINKING = sys.argv[2].lower() == 'true' if len(sys.argv) > 2 else True

MODEL_ROOT = '/home/admin1/models'
TEST_IMG_DIR = Path('/home/admin1/models/vrsbench_images/Images_val')
TEST_IMG = sorted(TEST_IMG_DIR.glob('*.png'))[0]

print(f"[smoke] model={MODEL_NAME}  thinking={ENABLE_THINKING}")
print(f"[smoke] test image: {TEST_IMG.name}")

print("[smoke] loading model...")
t0 = time.time()
model, processor = load_model(MODEL_NAME, MODEL_ROOT)
print(f"[smoke] loaded in {time.time()-t0:.1f}s, VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB")

img = Image.open(TEST_IMG).convert('RGB')
prompt = "Describe this remote sensing image in detail."

print(f"[smoke] prompt: {prompt!r}")
t0 = time.time()
res = infer(
    model, processor, img, prompt,
    max_new_tokens=2048,
    enable_thinking=ENABLE_THINKING,
    do_sample=False,
)
dt = time.time() - t0

print(f"\n[smoke] === output ===")
print(f"[smoke] raw_text (full, with think):")
print(res['raw'][:1500])
print(f"\n[smoke] clean_text (think stripped):")
print(res['out_text'][:500])
print(f"\n[smoke] === stats ===")
print(f"  time:           {dt:.2f}s")
print(f"  out_tokens:     {res['out_tokens']}")
print(f"  input_tokens:   {res['input_tokens']}")
print(f"  thinking_tokens:{res['thinking_tokens']}")
print(f"  answer_tokens:  {res['answer_tokens']}")
print(f"  speed:          {res['out_tokens']/dt:.1f} tok/s")

has_think = '<think>' in res['raw'].lower()
print(f"\n[smoke] has <think> tag: {has_think}")
if ENABLE_THINKING:
    assert has_think, f"thinking=True but no <think> tag in output!"
    assert res['thinking_tokens'] > 0, "thinking_tokens should be > 0"
    print("[smoke] ✓ thinkON works (has <think>, thinking_tokens > 0)")
else:
    assert not has_think or res['thinking_tokens'] == 0, "thinking=False but think tag/tokens present"
    print("[smoke] ✓ thinkOFF works (no <think> or stripped)")

del model, processor
torch.cuda.empty_cache()
print(f"\n[smoke] ✓ {MODEL_NAME} passed")
