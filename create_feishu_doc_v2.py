#!/usr/bin/env python3
"""创建飞书云文档 — VLM 遥感评测框架完整说明（含源码）"""
import json, os, time, requests, sys
from pathlib import Path

APP_ID = os.environ.get('FEISHU_APP_ID', 'cli_aa8b2fc430211cde')
APP_SECRET = os.environ.get('FEISHU_APP_SECRET', '')
DOC_TITLE = "VLM 遥感评测框架 — thinkOFF_eval 完整说明"
CODE_DIR = Path('/home/admin1/projects/thinkOFF_eval')

assert APP_ID and APP_SECRET, "FEISHU_APP_ID / FEISHU_APP_SECRET not set"

def get_token():
    r = requests.post(
        'https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal',
        json={"app_id": APP_ID, "app_secret": APP_SECRET},
    )
    d = r.json()
    assert d.get('code') == 0, f"Token failed: {d}"
    return d['tenant_access_token']

def add_children(token, doc_id, parent_id, children):
    r = requests.post(
        f'https://open.feishu.cn/open-apis/docx/v1/documents/{doc_id}/blocks/{parent_id}/children',
        headers={"Authorization": f"Bearer {token}"},
        json={"children": children},
    )
    d = r.json()
    if d.get('code') != 0:
        print(f"  [WARN] {d.get('msg','?')}")

T = lambda lvl, text, bt: {
    "block_type": lvl, bt: {
        "elements": [{"text_run": {"content": text, "text_element_style": {}}}]
    }
}
H1 = lambda t: T(4, t, "heading1")
H2 = lambda t: T(5, t, "heading2")
H3 = lambda t: T(6, t, "heading3")
P = lambda t: {"block_type": 2, "text": {"elements": [{"text_run": {"content": t, "text_element_style": {}}}]}}
BL = lambda t: {"block_type": 13, "bullet": {"elements": [{"text_run": {"content": t, "text_element_style": {}}}]}}
DIV = {"block_type": 22}
CODE = lambda txt, lang=1: {
    "block_type": 15, "code": {
        "elements": [{"text_run": {"content": txt, "text_element_style": {}}}],
        "style": {"language": lang, "wrap": True},
    }
}

def read_src(name):
    fp = CODE_DIR / name
    return fp.read_text() if fp.exists() else f"[{name} not found]"

# ─── Build ──────────────────────────────────────────────────
token = get_token()
print(f"Token OK")

# Create doc
r = requests.post(
    'https://open.feishu.cn/open-apis/docx/v1/documents',
    headers={"Authorization": f"Bearer {token}"},
    json={"title": DOC_TITLE},
)
doc_id = r.json()['data']['document']['document_id']
print(f"Doc: {doc_id}")
root = doc_id

# ─── Add blocks in batches ──────────────────────────────────
batch = []
def flush():
    global batch
    if batch:
        add_children(token, doc_id, root, batch)
        batch = []
        sys.stdout.flush()

def add(*blocks):
    global batch
    batch.extend(blocks)
    if len(batch) >= 20:
        flush()

# │ 标题
add(
    H1("VLM 遥感评测框架 — thinkOFF_eval"),
    P(f"生成时间：{time.strftime('%Y-%m-%d %H:%M')}"),
    P("基于 transformers 的零样本遥感 VLM 评测框架。支持 4 个 benchmark (VRSBench / MME-RS / LEVIR-CC / XLRS) × 8 个模型的完整评测流程，含 thinkON/thinkOFF 双模式对比。"),
    DIV,
)
flush()

# ═══════ 一、数据集说明 ═══════
add(H1("一、数据集说明"))
add(
    H2("1.1 VRSBench Caption"),
    BL("来源：xiang709/VRSBench (HuggingFace)"),
    BL("图片量：9,351 张（验证集 Images_val/）"),
    BL("标注量：877 条（VRSBench_EVAL_Cap.json）"),
    BL("采样：50 条 × 2 batch（sampled_eval/batch{1,2}/）"),
    BL("任务：遥感图像描述（Scene-level Caption）"),
    BL("指标：BLEU-1~4, ROUGE-L, CIDER"),
    BL('Prompt："Describe this remote sensing image in detail."'),
)
add(
    H2("1.2 VRSBench VQA"),
    BL("图片量：9,351 张（同上）"),
    BL("标注量：1,958 条（VRSBench_EVAL_vqa.json）"),
    BL("任务：遥感 VQA（12 种类型）"),
    BL("指标：L1 substring / L3 LLM 语义判定"),
)
add(
    H2("1.3 MME-RealWorld RS"),
    BL("来源：yifanzhang114/MME-RealWorld-Base64"),
    BL("图片量：1,300 张"),
    BL("标注量：1,265 条 RS 子集"),
    BL("任务：多选 VQA"),
    BL("指标：准确率"),
)
add(
    H2("1.4 LEVIR-CC"),
    BL("来源：Chen-Yang-Liu/LEVIR-CC-Dataset"),
    BL("图片量：2,135 对（A=before, B=after）"),
    BL("标注量：2,135 条，每样本 2-3 参考"),
    BL("任务：变化描述（双图输入）"),
    BL("指标：BLEU-1~4, ROUGE-L, CIDER（多参考）"),
)
add(
    H2("1.5 XLRS-Bench"),
    BL("来源：initiacms/XLRS-Bench-lite"),
    BL("图片量：42 张（超高分辨率 ~2500x2500）"),
    BL("标注量：42 条（全量）"),
    BL("任务：超高分辨率多选 VQA"),
    BL("指标：准确率"),
    DIV,
)
flush()

# ═══════ 二、目录结构 ═══════
add(H1("二、目录结构"))
add(CODE(read_src('README.md').split('## 目录结构')[1].split('## 评测流程')[0].strip()
         if '## 目录结构' in read_src('README.md') 
         else "thinkOFF_eval/\n├── config.py\n├── inference.py\n├── metrics.py\n├── datasets.py\n├── eval.py\n├── vqa_judge.py\n├── gen_excel.py\n├── requirements.txt\n├── create_feishu_doc.py\n├── run_all.sh\n└── README.md", lang=3))
add(DIV)
flush()

# ═══════ 三、评测流程 ═══════
add(H1("三、评测流程"))
add(H2("环境准备"))
add(CODE("pip install -r requirements.txt", lang=3))
add(H2("推理评测"))
add(P("thinkOFF 模式："))
add(CODE("python3 eval.py --model qwen3.5-0.8B", lang=3))
add(P("thinkON 模式（跳 caption 加速）："))
add(CODE("python3 eval.py --model qwen3.5-0.8B --thinking --max_new 4096 --skip_caption", lang=3))
add(P("批量评测所有模型："))
add(CODE("python3 eval.py --all", lang=3))
add(H2("VQA 三级判定"))
add(CODE("python3 vqa_judge.py --model qwen3.5-0.8B --mode thinkOFF --batch 1 --gguf /path/to/judge.gguf", lang=3))
add(H2("Excel 报告"))
add(CODE("python3 gen_excel.py              # batch1\npython3 gen_excel.py --batch 2    # batch2", lang=3))
add(DIV)
flush()

# ═══════ 四、源码 ═══════
add(H1("四、源码详解"))

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
    add(H2(f"{fname} — {desc}"))
    code_text = read_src(fname)
    # 代码块最大 ~45K chars，超长则分段
    max_chunk = 40000
    if len(code_text) <= max_chunk:
        add(CODE(code_text, lang))
    else:
        parts = [code_text[i:i+max_chunk] for i in range(0, len(code_text), max_chunk)]
        for pi, part in enumerate(parts):
            add(P(f"(continued {pi+1}/{len(parts)})"))
            add(CODE(part, lang))
    flush()

add(DIV)
flush()

# ═══════ 五、输出指标 ═══════
add(H1("五、输出指标说明"))
add(
    P("MME-RS — MME-RealWorld RS 准确率 (0-1)：多选 VQA"),
    P("XLRS — XLRS-Bench 准确率 (0-1)：超高分辨率多选"),
    P("VRS-Cap B4 — VRSBench Caption BLEU-4 (0-1)：n-gram 精确匹配"),
    P("VRS-Cap R-L — VRSBench Caption ROUGE-L (0-1)：最长公共子序列"),
    P("VRS-Cap CIDER — VRSBench Caption CIDER (0-∞)：TF-IDF 加权匹配"),
    P("LEVIR B4 — LEVIR-CC BLEU-4 (0-1)：变化描述 n-gram"),
    P("LEVIR R-L — LEVIR-CC ROUGE-L (0-1)：变化描述 LCS"),
    P("LEVIR CIDER — LEVIR-CC CIDER (0-∞)：变化描述 TF-IDF"),
    P("VRS-VQA — VRSBench VQA L1 准确率 (0-1)：substring 匹配"),
    P(""),
    P("⚠️ 注意：Caption 类指标（BLEU/CIDER）对输出风格极其敏感。模型预测为长段落格式时，即使语义正确，BLEU-4 和 CIDER 也可能接近 0。这是指标特性，不是 bug。"),
    DIV,
)
flush()

# ═══════ 六、模型列表 ═══════
add(H1("六、模型列表"))
add(P("所有模型通过 AutoModelForImageTextToText 加载。"))
add(BL("minicpm-v-4.6 — 4.6B — thinkON ✅"))
add(BL("qwen3.5-0.8B — 0.8B — thinkON ✅"))
add(BL("qwen3.5-2B — 2B — thinkON ✅"))
add(BL("qwen3.5-4B — 4B — thinkON ✅"))
add(BL("qwen3-vl-2B — 2B — thinkON ❌"))
add(BL("qwen3-vl-4B — 4B — thinkON ❌"))
add(BL("gemma-4-e2b — 2B — thinkON ❌"))
add(BL("gemma-4-e4b — 4B — thinkON ❌"))
flush()

print(f"\n✅ Done! https://bytedance.feishu.cn/docx/{doc_id}")
