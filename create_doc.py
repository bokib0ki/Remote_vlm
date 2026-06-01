#!/usr/bin/env python3
"""
创建飞书云文档 — VLM 遥感评测框架完整说明（含源码）
逐个 block 添加，避免 batch 问题
"""
import json, os, time, requests, sys
from pathlib import Path

APP_ID = os.environ.get('FEISHU_APP_ID', 'cli_aa8b2fc430211cde')
APP_SECRET = os.environ.get('FEISHU_APP_SECRET', '')
DOC_TITLE = "VLM 遥感评测框架 — thinkOFF_eval 完整说明"
CODE_DIR = Path('/home/admin1/projects/thinkOFF_eval')

assert APP_ID and APP_SECRET
ROOT_ID = None  # set after doc creation

def token():
    r = requests.post('https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal',
        json={"app_id": APP_ID, "app_secret": APP_SECRET})
    return r.json()['tenant_access_token']

def add(block):
    """添加一个 block，失败重试一次。"""
    r = requests.post(
        f'https://open.feishu.cn/open-apis/docx/v1/documents/{DOC_ID}/blocks/{ROOT_ID}/children',
        headers={"Authorization": f"Bearer {TOK}"},
        json={"children": [block]},
        timeout=30,
    )
    attempt = 0
    while r.status_code != 200 and attempt < 2:
        time.sleep(1)
        r = requests.post(
            f'https://open.feishu.cn/open-apis/docx/v1/documents/{DOC_ID}/blocks/{ROOT_ID}/children',
            headers={"Authorization": f"Bearer {TOK}"},
            json={"children": [block]},
            timeout=30,
        )
        attempt += 1
    try:
        d = r.json()
        if d.get('code') != 0:
            print(f"  WARN: {d.get('msg','?')[:80]}")
    except:
        print(f"  WARN: status={r.status_code} body={r.text[:100]}")
    return r

# ─── Block builders ──────────────────────────────────────
H1 = lambda t: {"block_type": 3, "heading1": {"elements": [{"text_run": {"content": t, "text_element_style": {}}}]}}
H2 = lambda t: {"block_type": 4, "heading2": {"elements": [{"text_run": {"content": t, "text_element_style": {}}}]}}
H3 = lambda t: {"block_type": 5, "heading3": {"elements": [{"text_run": {"content": t, "text_element_style": {}}}]}}
P = lambda t: {"block_type": 2, "text": {"elements": [{"text_run": {"content": t, "text_element_style": {}}}]}}
BL = lambda t: {"block_type": 12, "bullet": {"elements": [{"text_run": {"content": t, "text_element_style": {}}}]}}
CODE = lambda txt, lang=1: {
    "block_type": 14, "code": {
        "elements": [{"text_run": {"content": txt, "text_element_style": {}}}],
        "style": {"language": lang, "wrap": True},
    }
}

def read_src(name):
    fp = CODE_DIR / name
    return fp.read_text() if fp.exists() else f"[{name} not found]"

# ─── Main ──────────────────────────────────────────────────
TOK = token()
print("Token OK")

r = requests.post('https://open.feishu.cn/open-apis/docx/v1/documents',
    headers={"Authorization": f"Bearer {TOK}"},
    json={"title": DOC_TITLE})
DOC_ID = r.json()['data']['document']['document_id']
ROOT_ID = DOC_ID
print(f"Doc: https://bytedance.feishu.cn/docx/{DOC_ID}")

n = 0

# ══════════════════════════════════
# 标题
# ══════════════════════════════════
add(H1("VLM 遥感评测框架 — thinkOFF_eval")); n+=1
add(P(f"生成时间：{time.strftime('%Y-%m-%d %H:%M')}")); n+=1
add(P("基于 transformers 的零样本遥感 VLM 评测框架。支持 4 个 benchmark (VRSBench / MME-RS / LEVIR-CC / XLRS) × 8 个模型，含 thinkON/thinkOFF 双模式对比。")); n+=1

# ══════════════════════════════════
# 一、数据集说明
# ══════════════════════════════════
add(H1("一、数据集说明")); n+=1

add(H2("1.1 VRSBench Caption")); n+=1
add(BL("来源：xiang709/VRSBench (HuggingFace)")); n+=1
add(BL("图片量：9,351 张（验证集 Images_val/）")); n+=1
add(BL("标注量：877 条（VRSBench_EVAL_Cap.json）")); n+=1
add(BL("采样：50 条 × 2 batch")); n+=1
add(BL("任务：遥感图像描述（Scene-level Caption）")); n+=1
add(BL("指标：BLEU-1~4, ROUGE-L, CIDER")); n+=1
add(BL('Prompt："Describe this remote sensing image in detail."')); n+=1

add(H2("1.2 VRSBench VQA")); n+=1
add(BL("图片量：9,351 张（同上）")); n+=1
add(BL("标注量：1,958 条（VRSBench_EVAL_vqa.json）")); n+=1
add(BL("任务：遥感 VQA（12 种类型）")); n+=1
add(BL("指标：L1 substring / L3 LLM 语义判定")); n+=1
add(BL('Prompt：GeoChat 官方 prompt')); n+=1

add(H2("1.3 MME-RealWorld RS")); n+=1
add(BL("来源：yifanzhang114/MME-RealWorld-Base64")); n+=1
add(BL("图片量：1,300 张")); n+=1
add(BL("标注量：1,265 条 RS 子集")); n+=1
add(BL("任务：多选 VQA")); n+=1
add(BL("指标：准确率")); n+=1

add(H2("1.4 LEVIR-CC")); n+=1
add(BL("来源：Chen-Yang-Liu/LEVIR-CC-Dataset")); n+=1
add(BL("图片量：2,135 对（A=before, B=after）")); n+=1
add(BL("标注量：2,135 条，每样本 2-3 参考")); n+=1
add(BL("任务：变化描述（双图输入）")); n+=1
add(BL("指标：BLEU-1~4, ROUGE-L, CIDER（多参考）")); n+=1

add(H2("1.5 XLRS-Bench")); n+=1
add(BL("来源：initiacms/XLRS-Bench-lite")); n+=1
add(BL("图片量：42 张（超高分辨率 ~2500x2500）")); n+=1
add(BL("标注量：42 条（全量）")); n+=1
add(BL("任务：超高分辨率多选 VQA")); n+=1
add(BL("指标：准确率")); n+=1

# ══════════════════════════════════
# 二、目录结构
# ══════════════════════════════════
add(H1("二、目录结构")); n+=1
add(CODE("""thinkOFF_eval/
├── config.py        全局配置（路径、模型、prompt模板）
├── inference.py     模型加载与推理（单/双图，thinkON/OFF）
├── metrics.py       BLEU/ROUGE/CIDER 评分接口
├── datasets.py      5个数据集的加载函数
├── eval.py          主评测入口（单模型/批量/all）
├── vqa_judge.py     VQA 三级判定器（L1/L2/L3）
├── gen_excel.py     Excel 报告生成（含SOTA对比）
├── requirements.txt
├── create_feishu_doc.py
├── run_all.sh
└── README.md""", lang=3)); n+=1

add(P("数据目录："))
add(CODE("""models/
├── {model_name}/                  模型权重
├── vrsbench_images/Images_val/    VRSBench 图片 (9,351)
├── VRSBench_EVAL_Cap.json         VRS-Caption 完整标注
├── levircc_data/extracted/images/ LEVIR-CC 图片 (A/B)
├── mme_images/remote_sensing_full/ MME-RS 图片
├── xlrs_arrow/images/             XLRS 图片 (42)
├── sampled_eval/batch{1,2}/       采样文件
└── test_images/                   占位图""", lang=3)); n+=1

# ══════════════════════════════════
# 三、评测流程
# ══════════════════════════════════
add(H1("三、评测流程")); n+=1
add(H2("环境准备")); n+=1
add(CODE("pip install -r requirements.txt", lang=3)); n+=1

add(H2("推理评测")); n+=1
add(P("单模型 thinkOFF（默认）：")); n+=1
add(CODE("python3 eval.py --model qwen3.5-0.8B", lang=3)); n+=1
add(P("thinkON 模式（跳 caption 加速）：")); n+=1
add(CODE("python3 eval.py --model qwen3.5-0.8B --thinking --max_new 4096 --skip_caption", lang=3)); n+=1
add(P("批量所有模型："))
add(CODE("python3 eval.py --all              # 跑所有模型", lang=3)); n+=2

add(H2("VQA 三级判定")); n+=1
add(CODE("python3 vqa_judge.py --model qwen3.5-0.8B --mode thinkOFF --gguf /path/to/judge.gguf", lang=3)); n+=1

add(H2("生成 Excel 报告")); n+=1
add(CODE("python3 gen_excel.py              # batch1\npython3 gen_excel.py --batch 2    # batch2", lang=3)); n+=1
add(P("输出 3 个 Sheet：指标对比表（含SOTA） + 原始输出 + 分析总结"))

# ══════════════════════════════════
# 四、源码
# ══════════════════════════════════
add(H1("四、源码详解")); n+=1

src_files = [
    ('config.py', '全局配置', 1), ('inference.py', '模型加载与推理', 1),
    ('metrics.py', 'BLEU/ROUGE/CIDER 计算', 1), ('datasets.py', '数据集加载', 1),
    ('eval.py', '主评测入口', 1), ('vqa_judge.py', 'VQA 三级判定器', 1),
    ('gen_excel.py', 'Excel 报告生成', 1),
]

for fname, desc, lang in src_files:
    add(H2(f"{fname} — {desc}")); n+=1
    code_text = read_src(fname)
    max_chunk = 35000
    if len(code_text) <= max_chunk:
        add(CODE(code_text, lang)); n+=1
    else:
        parts = [code_text[i:i+max_chunk] for i in range(0, len(code_text), max_chunk)]
        for pi, part in enumerate(parts):
            add(P(f"[{pi+1}/{len(parts)}]"))
            add(CODE(part, lang)); n+=2

# ══════════════════════════════════
# 五、输出指标
# ══════════════════════════════════
add(H1("五、输出指标说明")); n+=1
add(P("MME-RS — MME-RealWorld RS 准确率 (0-1)：多选 VQA")); n+=1
add(P("XLRS — XLRS-Bench 准确率 (0-1)：超高分辨率多选")); n+=1
add(P("VRS-Cap B4 — VRSBench Caption BLEU-4 (0-1)")); n+=1
add(P("VRS-Cap R-L — VRSBench Caption ROUGE-L (0-1)")); n+=1
add(P("VRS-Cap CIDER — VRSBench Caption CIDER (0-∞)")); n+=1
add(P("LEVIR B4 — LEVIR-CC BLEU-4 (0-1)：变化描述 n-gram")); n+=1
add(P("LEVIR R-L — LEVIR-CC ROUGE-L (0-1)")); n+=1
add(P("LEVIR CIDER — LEVIR-CC CIDER (0-∞)")); n+=1
add(P("VRS-VQA — VRSBench VQA L1 准确率 (0-1)")); n+=1
add(P(""))
add(P("⚠️ 注意：Caption 指标（BLEU/CIDER）对输出风格极其敏感。模型输出长段落时即使语义正确，BLEU-4 和 CIDER 也可能接近 0。"))

# ══════════════════════════════════
# 六、模型列表
# ══════════════════════════════════
add(H1("六、模型列表")); n+=1
add(P("所有模型通过 AutoModelForImageTextToText 加载。"))
add(BL("minicpm-v-4.6 — 4.6B — thinkON ✅")); n+=1
add(BL("qwen3.5-0.8B — 0.8B — thinkON ✅")); n+=1
add(BL("qwen3.5-2B — 2B — thinkON ✅")); n+=1
add(BL("qwen3.5-4B — 4B — thinkON ✅")); n+=1
add(BL("qwen3-vl-2B — 2B — thinkON ❌")); n+=1
add(BL("qwen3-vl-4B — 4B — thinkON ❌")); n+=1
add(BL("gemma-4-e2b — 2B — thinkON ❌")); n+=1
add(BL("gemma-4-e4b — 4B — thinkON ❌")); n+=1

print(f"\n✅ Done! {n} blocks")
print(f"https://bytedance.feishu.cn/docx/{DOC_ID}")
