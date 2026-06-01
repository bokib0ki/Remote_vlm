#!/usr/bin/env python3
"""
创建飞书云文档 — VLM 遥感小模型评测框架完整说明（含源码）
"""
import json, os, time, requests
from pathlib import Path

APP_ID = os.environ.get('FEISHU_APP_ID', 'cli_aa8b2fc430211cde')
APP_SECRET = os.environ.get('FEISHU_APP_SECRET', '')
DOC_TITLE = "VLM 遥感评测框架 — thinkOFF_eval 完整说明"

CODE_DIR = Path('/home/admin1/projects/thinkOFF_eval')


def get_token():
    r = requests.post(
        'https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal',
        json={"app_id": APP_ID, "app_secret": APP_SECRET},
    )
    return r.json().get('tenant_access_token', '')


def create_doc(token, title):
    r = requests.post(
        'https://open.feishu.cn/open-apis/docx/v1/documents',
        headers={"Authorization": f"Bearer {token}"},
        json={"title": title},
    )
    return r.json().get('data', {}).get('document', {}).get('document_id', '')


# ─── Block helpers ───────────────────────────────────────────

def add_children(token, doc_id, parent_id, children):
    """批量添加子 block。"""
    r = requests.post(
        f'https://open.feishu.cn/open-apis/docx/v1/documents/{doc_id}/blocks/{parent_id}/children',
        headers={"Authorization": f"Bearer {token}"},
        json={"children": children},
    )
    if r.json().get('code') != 0:
        print(f"  [WARN] block add failed: {r.json().get('msg', '')}")

def h1(token, doc_id, parent, text):
    add_children(token, doc_id, parent, [{
        "block_type": 4, "heading1": {
            "elements": [{"text_run": {"content": text, "text_element_style": {}}}]
        }
    }])

def h2(token, doc_id, parent, text):
    add_children(token, doc_id, parent, [{
        "block_type": 5, "heading2": {
            "elements": [{"text_run": {"content": text, "text_element_style": {}}}]
        }
    }])

def h3(token, doc_id, parent, text):
    add_children(token, doc_id, parent, [{
        "block_type": 6, "heading3": {
            "elements": [{"text_run": {"content": text, "text_element_style": {}}}]
        }
    }])

def p(token, doc_id, parent, text):
    add_children(token, doc_id, parent, [{
        "block_type": 2, "text": {
            "elements": [{"text_run": {"content": text, "text_element_style": {}}}]
        }
    }])

def code_block(token, doc_id, parent, code_text, language=1):
    """添加代码块。language: 1=Python, 3=Shell/Bash, 18=JSON, 21=Text"""
    add_children(token, doc_id, parent, [{
        "block_type": 15,
        "code": {
            "elements": [{"text_run": {"content": code_text, "text_element_style": {}}}],
            "style": {"language": language, "wrap": True},
        }
    }])

def divider(token, doc_id, parent):
    add_children(token, doc_id, parent, [{"block_type": 22}])

def bullet(token, doc_id, parent, text):
    add_children(token, doc_id, parent, [{
        "block_type": 13, "bullet": {
            "elements": [{"text_run": {"content": text, "text_element_style": {}}}]
        }
    }])


# ─── Read source files ──────────────────────────────────────

def read_src(name):
    fp = CODE_DIR / name
    if fp.exists():
        return fp.read_text()
    return f"[{name} not found]"


# ─── Build document ──────────────────────────────────────────

def build_doc():
    token = get_token()
    if not token:
        print("Failed to get token"); return
    print(f"Token: {token[:20]}...")

    doc_id = create_doc(token, DOC_TITLE)
    print(f"Doc ID: {doc_id}")

    root = doc_id  # page block id = doc id

    # ════════════════════════════════════════════════
    # 标题
    # ════════════════════════════════════════════════
    h1(token, doc_id, root, "VLM 遥感评测框架 — thinkOFF_eval")
    p(token, doc_id, root, f"生成时间：{time.strftime('%Y-%m-%d %H:%M')}")
    p(token, doc_id, root, "基于 transformers 的零样本遥感 VLM 评测框架。支持 4 个 benchmark (VRSBench / MME-RS / LEVIR-CC / XLRS) × 8 个模型的完整评测流程，含 thinkON/thinkOFF 双模式对比。")
    divider(token, doc_id, root)

    # ════════════════════════════════════════════════
    # 一、数据集说明
    # ════════════════════════════════════════════════
    h1(token, doc_id, root, "一、数据集说明")

    # 1.1 VRSBench Caption
    h2(token, doc_id, root, "1.1 VRSBench Caption")
    bullet(token, doc_id, root, "来源：xiang709/VRSBench (HuggingFace)")
    bullet(token, doc_id, root, "图片量：9,351 张（验证集 Images_val/）")
    bullet(token, doc_id, root, "标注量：877 条（VRSBench_EVAL_Cap.json）")
    bullet(token, doc_id, root, "采样：50 条 × 2 batch（sampled_eval/batch{1,2}/）")
    bullet(token, doc_id, root, "任务：遥感图像描述（Scene-level Caption）")
    bullet(token, doc_id, root, "指标：BLEU-1~4, ROUGE-L, CIDER")
    bullet(token, doc_id, root, 'Prompt："Describe this remote sensing image in detail."')

    # 1.2 VRSBench VQA
    h2(token, doc_id, root, "1.2 VRSBench VQA")
    bullet(token, doc_id, root, "图片量：9,351 张（同上）")
    bullet(token, doc_id, root, "标注量：1,958 条（VRSBench_EVAL_vqa.json）")
    bullet(token, doc_id, root, "采样：50 条 × 2 batch")
    bullet(token, doc_id, root, "任务：遥感 VQA（12 种类型：物体类别/存在/计数/颜色/形状/大小/方向/位置/场景/推理/城乡等）")
    bullet(token, doc_id, root, "指标：L1 substring 匹配 / L3 LLM 语义判定")
    bullet(token, doc_id, root, 'Prompt："{question}\nAnswer the question using a single word or phrase."（GeoChat 官方）')

    # 1.3 MME-RealWorld RS
    h2(token, doc_id, root, "1.3 MME-RealWorld Remote Sensing")
    bullet(token, doc_id, root, "来源：yifanzhang114/MME-RealWorld-Base64 (HuggingFace)")
    bullet(token, doc_id, root, "图片量：1,300 张（mme_images/remote_sensing_full/）")
    bullet(token, doc_id, root, "标注量：1,265 条 RS 子集（mme_rs_annotations.json）")
    bullet(token, doc_id, root, "任务：多选 VQA（遥感子集，含属性/存在/颜色等类型）")
    bullet(token, doc_id, root, "指标：准确率（E 选项标记为偏见）")
    bullet(token, doc_id, root, "Prompt：MME 官方多选格式，只回复字母")

    # 1.4 LEVIR-CC
    h2(token, doc_id, root, "1.4 LEVIR-CC")
    bullet(token, doc_id, root, "来源：Chen-Yang-Liu/LEVIR-CC-Dataset (GitHub)")
    bullet(token, doc_id, root, "图片量：2,135 对（test: A = before, B = after）")
    bullet(token, doc_id, root, "标注量：2,135 条，每样本 2-3 条参考描述")
    bullet(token, doc_id, root, "任务：变化描述（Change Captioning，双图输入）")
    bullet(token, doc_id, root, "指标：BLEU-1~4, ROUGE-L, CIDER（多参考）")
    bullet(token, doc_id, root, 'Prompt："Describe the changes between these two remote sensing images taken at different times."')

    # 1.5 XLRS-Bench
    h2(token, doc_id, root, "1.5 XLRS-Bench")
    bullet(token, doc_id, root, "来源：initiacms/XLRS-Bench-lite (HuggingFace)")
    bullet(token, doc_id, root, "图片量：42 张（超高分辨率 ~2500x2500）")
    bullet(token, doc_id, root, "标注量：42 条（xlrs_arrow/xlrs_samples_42.json，全量）")
    bullet(token, doc_id, root, "任务：超高分辨率多选 VQA")
    bullet(token, doc_id, root, "指标：准确率（多选题，答案可含多字母）")
    bullet(token, doc_id, root, "Prompt：XLRS 官方多选格式，只回复字母")

    divider(token, doc_id, root)

    # ════════════════════════════════════════════════
    # 二、目录结构
    # ════════════════════════════════════════════════
    h1(token, doc_id, root, "二、目录结构")
    code_block(token, doc_id, root, read_src('README.md').split('# 数据集说明')[0].split('## 使用方法')[0].split('## 数据集说明')[0], language=3)

    divider(token, doc_id, root)

    # ════════════════════════════════════════════════
    # 三、评测流程
    # ════════════════════════════════════════════════
    h1(token, doc_id, root, "三、评测流程")
    h2(token, doc_id, root, "环境准备")
    code_block(token, doc_id, root, "pip install -r requirements.txt", language=3)

    h2(token, doc_id, root, "推理评测（单模型）")
    p(token, doc_id, root, "thinkOFF 模式：")
    code_block(token, doc_id, root, "python3 eval.py --model qwen3.5-0.8B", language=3)
    p(token, doc_id, root, "thinkON 模式（跳 caption 以节省时间）：")
    code_block(token, doc_id, root, "python3 eval.py --model qwen3.5-0.8B --thinking --max_new 4096 --skip_caption", language=3)
    p(token, doc_id, root, "指定 batch：")
    code_block(token, doc_id, root, "python3 eval.py --model qwen3.5-0.8B --batch 2", language=3)

    h2(token, doc_id, root, "批量评测所有模型")
    code_block(token, doc_id, root, "python3 eval.py --all", language=3)

    h2(token, doc_id, root, "VQA 三级判定")
    code_block(token, doc_id, root, "python3 vqa_judge.py --model qwen3.5-0.8B --mode thinkOFF --batch 1 --gguf /path/to/judge.gguf", language=3)
    p(token, doc_id, root, "判分模型：Qwen3.6-35B-A3B Q4_K_M GGUF (~21GB)")

    h2(token, doc_id, root, "生成 Excel 报告")
    code_block(token, doc_id, root, "python3 gen_excel.py              # batch1\npython3 gen_excel.py --batch 2    # batch2", language=3)

    divider(token, doc_id, root)

    # ════════════════════════════════════════════════
    # 四、源码
    # ════════════════════════════════════════════════
    h1(token, doc_id, root, "四、源码详解")

    src_files = [
        ('config.py', '全局配置', 1),
        ('inference.py', '模型加载与推理', 1),
        ('metrics.py', 'BLEU/ROUGE/CIDER 计算', 1),
        ('datasets.py', '数据集加载', 1),
        ('eval.py', '主评测入口', 1),
        ('vqa_judge.py', 'VQA 三级判定器', 1),
        ('gen_excel.py', 'Excel 报告生成', 1),
    ]

    for fname, desc, lang in src_files:
        h2(token, doc_id, root, f"{fname} — {desc}")
        code_text = read_src(fname)
        # Feishu code blocks have ~50K char limit, split if too long
        max_chunk = 45000
        if len(code_text) <= max_chunk:
            code_block(token, doc_id, root, code_text, language=lang)
        else:
            parts = [code_text[i:i+max_chunk] for i in range(0, len(code_text), max_chunk)]
            for pi, part in enumerate(parts):
                p(token, doc_id, root, f"(continued {pi+1}/{len(parts)})")
                code_block(token, doc_id, root, part, language=lang)

    divider(token, doc_id, root)

    # ════════════════════════════════════════════════
    # 五、输出指标说明
    # ════════════════════════════════════════════════
    h1(token, doc_id, root, "五、输出指标说明")
    p(token, doc_id, root, "MME-RS — MME-RealWorld RS 准确率 (0-1)：多选 VQA")
    p(token, doc_id, root, "XLRS — XLRS-Bench 准确率 (0-1)：超高分辨率多选")
    p(token, doc_id, root, "VRS-Cap B4 — VRSBench Caption BLEU-4 (0-1)：n-gram 精确匹配")
    p(token, doc_id, root, "VRS-Cap R-L — VRSBench Caption ROUGE-L (0-1)：最长公共子序列")
    p(token, doc_id, root, "VRS-Cap CIDER — VRSBench Caption CIDER (0-∞)：TF-IDF 加权匹配")
    p(token, doc_id, root, "LEVIR B4 — LEVIR-CC BLEU-4 (0-1)：变化描述 n-gram")
    p(token, doc_id, root, "LEVIR R-L — LEVIR-CC ROUGE-L (0-1)：变化描述 LCS")
    p(token, doc_id, root, "LEVIR CIDER — LEVIR-CC CIDER (0-∞)：变化描述 TF-IDF")
    p(token, doc_id, root, "VRS-VQA — VRSBench VQA L1 准确率 (0-1)：substring 匹配")

    p(token, doc_id, root, "")
    p(token, doc_id, root, "⚠️ 注意：Caption 类指标（BLEU/CIDER）对输出风格极其敏感。模型预测为长段落格式时，即使语义正确，BLEU-4 和 CIDER 也可能接近 0。这是指标特性，不是 bug。")

    divider(token, doc_id, root)

    # ════════════════════════════════════════════════
    # 六、模型列表
    # ════════════════════════════════════════════════
    h1(token, doc_id, root, "六、模型列表")
    p(token, doc_id, root, "所有模型通过 AutoModelForImageTextToText 加载，存放在 models/{name}/ 目录下。")
    p(token, doc_id, root, "minicpm-v-4.6 — 4.6B — thinkON ✅")
    p(token, doc_id, root, "qwen3.5-0.8B — 0.8B — thinkON ✅")
    p(token, doc_id, root, "qwen3.5-2B — 2B — thinkON ✅")
    p(token, doc_id, root, "qwen3.5-4B — 4B — thinkON ✅")
    p(token, doc_id, root, "qwen3-vl-2B — 2B — thinkON ❌")
    p(token, doc_id, root, "qwen3-vl-4B — 4B — thinkON ❌")
    p(token, doc_id, root, "gemma-4-e2b — 2B — thinkON ❌")
    p(token, doc_id, root, "gemma-4-e4b — 4B — thinkON ❌")

    print(f"\n✅ Done! https://bytedance.feishu.cn/docx/{doc_id}")
    return doc_id


if __name__ == '__main__':
    build_doc()
