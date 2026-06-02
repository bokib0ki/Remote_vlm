#!/usr/bin/env python3
"""
Smoke test: 验证 Qwen2.5-VL-3B + inference.py 现有代码路径能跑通。
不经过 eval_select.py，直接调 inference.infer 跑 1 张图。
"""
import sys, os, time, traceback

os.chdir('/home/admin1/projects/remote_vlm_eval')
sys.path.insert(0, '/home/admin1/projects/remote_vlm_eval')

import torch
from PIL import Image

TEST_IMG = "/home/admin1/models/vrsbench_images/Images_val/P0027_0008.png"
PROMPT = "Does the image show any buildings?\nAnswer the question using a single word or phrase."

print("=" * 60)
print("Qwen2.5-VL-3B 冒烟测试")
print("=" * 60)
print(f"图片: {TEST_IMG} (存在: {os.path.exists(TEST_IMG)})")
print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM free: {torch.cuda.mem_get_info()[0]/1e9:.1f}GB")

# 加载
print("\n[1/2] 加载 Qwen2.5-VL-3B...")
t0 = time.time()
from inference import load_model, infer
model, processor = load_model('qwen2.5-vl-3B', '/home/admin1/models')
load_dt = time.time() - t0
print(f"  加载耗时: {load_dt:.1f}s")
print(f"  VRAM used: {torch.cuda.memory_allocated()/1e9:.2f}GB")
print(f"  model class: {type(model).__name__}")
print(f"  processor class: {type(processor).__name__}")

# 准备图片
img = Image.open(TEST_IMG).convert('RGB')
print(f"  img size: {img.size}")

# 推理
print("\n[2/2] 推理（现有 inference.py 代码: processor(images=img, text=text)）")
t0 = time.time()
try:
    res = infer(model, processor, img, PROMPT, max_new_tokens=64, enable_thinking=False)
    dt = time.time() - t0
    print(f"  ✓ 成功 {dt:.2f}s")
    print(f"  out_tokens: {res['out_tokens']}")
    print(f"  speed: {res['out_tokens']/dt:.1f} t/s")
    print(f"  input_tokens: {res['input_tokens']} (img: {res['img_tokens']}, prompt: {res['prompt_tokens']})")
    print(f"  out_text: {repr(res['out_text'])}")
    print(f"  raw: {repr(res['raw'][:200])}")
    print("\n" + "=" * 60)
    print("✅ 现有 inference.py 代码路径可用")
    print("=" * 60)
except Exception as e:
    dt = time.time() - t0
    print(f"  ✗ 失败 ({dt:.2f}s): {type(e).__name__}: {e}")
    print("\n--- 完整 traceback ---")
    traceback.print_exc()
    print("\n" + "=" * 60)
    print(f"❌ 现有路径不可用: {type(e).__name__}")
    print("=" * 60)
    sys.exit(1)
