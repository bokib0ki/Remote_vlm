#!/usr/bin/env python3
"""
创建飞书云文档 — VLM 遥感评测框架完整说明（含源码）
飞书 API block type: heading1=3, heading2=4, heading3=5, text=2, bullet=12, code=14（标准文档+1偏移）
"""
import json, os, time, requests, sys
from pathlib import Path

APP_ID = os.environ.get('FEISHU_APP_ID', 'cli_aa8b2fc430211cde')
APP_SECRET = os.environ.get('FEISHU_APP_SECRET', '')
DOC_TITLE = "VLM 遥感评测框架 — thinkOFF_eval 完整说明"
CODE_DIR = Path('/home/admin1/projects/thinkOFF_eval')

assert APP_ID and APP_SECRET, "FEISHU_APP_ID / FEISHU_APP_SECRET not set"

# ─── API ──────────────────────────────────────────────────
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
def build_doc():
    token = get_token()
    print("Token OK")

    # Create doc
    r = requests.post(
        'https://open.feishu.cn/open-apis/docx/v1/documents',
        headers={"Authorization": f"Bearer {token}"},
        json={"title": DOC_TITLE},
    )
    doc_id = r.json()['data']['document']['document_id']
    print(f"Doc: {doc_id}")
    root = doc_id

    # ─── Batched block addition ────────────────────────────
    batch = []
    def flush():
        nonlocal batch
        if batch:
            add_children(token, doc_id, root, batch)
            batch = []
            sys.stdout.flush()

    def add(*blocks):
        batch.extend(blocks)
        if len(batch) >= 20:
            flush()

    # ══════════════════════════════════
    # 标题
    # ══════════════════════════════════
    add(
        H1("VLM 遥感评测框架 — thinkOFF_eval"),
        P(f"生成时间：{time.strftime('%Y-%m-%d %H:%M')}"),
        P("基于 transformers 的零样本遥感 VLM 评测框架。支持 4 个 benchmark (VRSBench / MME-RS / LEVIR-CC / XLRS) × 8 个模型的完整评测流程，含 thinkON/thinkOFF 双模式对比。"),
    )
    flush()

    # ══════════════════════════════════
    # 一、数据集说明
    # ══════════════════════════════════
    add(H1("一、数据集说明"))

    add(H2("1.1 VRSBench Caption"))
    add(BL("来源：xiang709/VRSBench (HuggingFace)"))
    add(BL("图片量：9,351 张（验证集 Images_val/）"))
    add(BL("标注量：877 条（VRSBench_EVAL_Cap.json）"))
    add(BL("采样：50 条 × 2 batch（sampled_eval/batch{1,2}/）"))
    add(BL("任务：遥感图像描述（Scene-level Caption）"))
    add(BL("指标：BLEU-1~4, ROUGE-L, CIDER"))
    add(BL('Prompt："Describe this remote sensing image in detail."'))
    flush()

    add(H2("1.2 VRSBench VQA"))
    add(BL("图片量：9,351 张（同上）"))
    add(BL("标注量：1,958 条（VRSBench_EVAL_vqa.json）"))
    add(BL("任务：遥感 VQA（12 种类型：物体类别/存在/计数/颜色/形状/大小/方向/位置/场景/推理/城乡等）"))
    add(BL("指标：L1 substring 匹配 / L3 LLM 语义判定"))
    add(BL('Prompt："{question}\\nAnswer the question using a single word or phrase."（GeoChat 官方）'))
    flush()

    add(H2("1.3 MME-RealWorld Remote Sensing"))
    add(BL("来源：yifanzhang114/MME-RealWorld-Base64 (HuggingFace)"))
    add(BL("图片量：1,300 张（mme_images/remote_sensing_full/）"))
    add(BL("标注量：1,265 条 RS 子集（mme_rs_annotations.json）"))
    add(BL("任务：多选 VQA（遥感子集，含属性/存在/颜色等类型）"))
    add(BL("指标：准确率（E 选项标记为偏见）"))
    add(BL("Prompt：MME 官方多选格式，只回复字母"))
    flush()

    add(H2("1.4 LEVIR-CC"))
    add(BL("来源：Chen-Yang-Liu/LEVIR-CC-Dataset (GitHub)"))
    add(BL("图片量：2,135 对（test: A = before, B = after）"))
    add(BL("标注量：2,135 条，每样本 2-3 条参考描述"))
    add(BL("任务：变化描述（Change Captioning，双图输入）"))
    add(BL("指标：BLEU-1~4, ROUGE-L, CIDER（多参考）"))
    add(BL('Prompt："Describe the changes between these two remote sensing images..."'))
    flush()

    add(H2("1.5 XLRS-Bench"))
    add(BL("来源：initiacms/XLRS-Bench-lite (HuggingFace)"))
    add(BL("图片量：42 张（超高分辨率 ~2500x2500）"))
    add(BL("标注量：42 条（xlrs_samples_42.json，全量）"))
    add(BL("任务：超高分辨率多选 VQA"))
    add(BL("指标：准确率（多选题，答案可含多字母）"))
    add(BL("Prompt：XLRS 官方多选格式，只回复字母"))
    flush()

    # ══════════════════════════════════
    # 二、目录结构
    # ══════════════════════════════════
    add(H1("二、目录结构"))
    flush()

    dir_tree = """thinkOFF_eval/
├── config.py           全局路径、模型、prompt
├── inference.py        模型加载 + 推理
├── metrics.py          BLEU/ROUGE/CIDER 评分
├── datasets.py         5个数据集加载
├── eval.py             主评测入口（单/批量）
├── vqa_judge.py        VQA 三级判定器
├── gen_excel.py        Excel 报告生成
├── requirements.txt
├── create_feishu_doc.py
├── run_all.sh
└── README.md           完整文档"""

    add(CODE(dir_tree, lang=3))

    add(P(""))
    add(P("数据目录结构（需提前准备）："))
    flush()

    data_tree = """models/
├── {model_name}/                   模型权重（AutoModel 格式）
├── vrsbench_images/Images_val/     VRSBench 图片 (9,351)
├── VRSBench_EVAL_Cap.json          VRS-Caption 完整标注
├── VRSBench_EVAL_vqa.json          VRS-VQA 完整标注
├── levircc_data/extracted/images/  LEVIR-CC 图片 (A/B)
├── mme_images/remote_sensing_full/ MME-RS 图片
├── mme_rs_annotations.json         MME-RS 完整标注
├── xlrs_arrow/images/              XLRS 图片 (42)
├── sampled_eval/batch{1,2}/        采样文件
├── results/                        评测结果 (JSON)
├── raw_outputs/batch{1,2}/         原始预测
└── test_images/                    占位图"""
    add(CODE(data_tree, lang=3))
    flush()

    # ══════════════════════════════════
    # 三、评测流程
    # ══════════════════════════════════
    add(H1("三、评测流程"))
    add(H2("环境准备"))
    add(CODE("pip install -r requirements.txt", lang=3))
    flush()

    add(H2("推理评测"))
    add(P("thinkOFF 模式（单模型）："))
    add(CODE("python3 eval.py --model qwen3.5-0.8B", lang=3))
    add(P("thinkON 模式（跳 caption 加速）："))
    add(CODE("python3 eval.py --model qwen3.5-0.8B --thinking --max_new 4096 --skip_caption", lang=3))
    add(P("指定 batch 2："))
    add(CODE("python3 eval.py --model qwen3.5-0.8B --batch 2", lang=3))
    add(P("批量评测所有模型："))
    add(CODE("python3 eval.py --all", lang=3))
    add(P("或者用 shell 脚本："))
    add(CODE("bash run_all.sh            # batch1\nbash run_all.sh 2          # batch2", lang=3))
    flush()

    add(H2("VQA 三级判定"))
    add(CODE("python3 vqa_judge.py --model qwen3.5-0.8B --mode thinkOFF --batch 1 --gguf /path/to/judge.gguf", lang=3))
    add(P("判分模型：Qwen3.6-35B-A3B Q4_K_M GGUF (~21GB)"))
    flush()

    add(H2("Excel 报告生成"))
    add(CODE("python3 gen_excel.py              # batch1\npython3 gen_excel.py --batch 2    # batch2", lang=3))
    add(P("输出：projects/batch{1,2}_thinkOFF.xlsx 和 _thinkON.xlsx"))
    add(BL("Sheet 1: 指标对比表（含 SOTA 参考行）"))
    add(BL("Sheet 2: 原始输出表（每模型 × 每样本）"))
    add(BL("Sheet 3: 分析总结（最佳模型/得分）"))
    flush()

    # ══════════════════════════════════
    # 四、源码
    # ══════════════════════════════════
    add(H1("四、源码详解"))
    flush()

    src_files = [
        ('config.py', '全局配置'), ('inference.py', '模型加载与推理'),
        ('metrics.py', 'BLEU/ROUGE/CIDER 计算'), ('datasets.py', '数据集加载'),
        ('eval.py', '主评测入口'), ('vqa_judge.py', 'VQA 三级判定器'),
        ('gen_excel.py', 'Excel 报告生成'),
    ]

    for fname, desc in src_files:
        add(H2(f"{fname} — {desc}"))
        code_text = read_src(fname)
        max_chunk = 35000
        if len(code_text) <= max_chunk:
            add(CODE(code_text, lang=1))
        else:
            parts = [code_text[i:i+max_chunk] for i in range(0, len(code_text), max_chunk)]
            for pi, part in enumerate(parts):
                add(P(f"(continued {pi+1}/{len(parts)})"))
                add(CODE(part, lang=1))
        flush()

    # ══════════════════════════════════
    # 五、输出指标说明
    # ══════════════════════════════════
    add(H1("五、输出指标说明"))
    add(P("MME-RS — MME-RealWorld RS 准确率 (0-1)：多选 VQA"))
    add(P("XLRS — XLRS-Bench 准确率 (0-1)：超高分辨率多选"))
    add(P("VRS-Cap B4 — VRSBench Caption BLEU-4 (0-1)：n-gram 精确匹配"))
    add(P("VRS-Cap R-L — VRSBench Caption ROUGE-L (0-1)：最长公共子序列"))
    add(P("VRS-Cap CIDER — VRSBench Caption CIDER (0-∞)：TF-IDF 加权匹配"))
    add(P("LEVIR B4 — LEVIR-CC BLEU-4 (0-1)：变化描述 n-gram"))
    add(P("LEVIR R-L — LEVIR-CC ROUGE-L (0-1)：变化描述 LCS"))
    add(P("LEVIR CIDER — LEVIR-CC CIDER (0-∞)：变化描述 TF-IDF"))
    add(P("VRS-VQA — VRSBench VQA L1 准确率 (0-1)：substring 匹配"))
    add(P(""))
    add(P("⚠️ 注意：Caption 类指标（BLEU/CIDER）对输出风格极其敏感。模型预测为长段落格式时，即使语义正确，BLEU-4 和 CIDER 也可能接近 0。这是指标特性，不是 bug。"))
    flush()

    # ══════════════════════════════════
    # 六、模型列表
    # ══════════════════════════════════
    add(H1("六、模型列表"))
    add(P("所有模型通过 AutoModelForImageTextToText 加载，存放在 models/{name}/ 目录下。"))
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
    return doc_id


if __name__ == '__main__':
    build_doc()
