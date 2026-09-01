#!/usr/bin/env python3
"""
循环内重排（rerank_loop）A/B 对照评测 + naive 基线。

目的：量化 agentic 循环里「循环内重排」（Rerank ①：tools → reflect 之间的精排闸门）
是否必要。三列对照：

    A（循环重排开）  agent，agent_reranker_loop_enabled=True   （生产默认）
    B（循环重排关）  agent，agent_reranker_loop_enabled=False  （仅保留出口 rerank_exit）
    C（naive 基线）  RAGChain（history 串联，无 agentic 循环）

指标（与需求对齐）：
    - token 消耗：token_usage_total（prompt/completion/cache/total/llm_calls）
    - 耗时：avg_elapsed_ms
    - 准确率：generation_metrics_avg.accuracy（LLM-as-judge 对比金标准）
    - 置信度：generation_metrics_avg.confidence（Judge 对自身评分的把握）
    - 召回率：retrieval_metrics_avg.recall@k
    - 精确度：retrieval_metrics_avg.precision@k
    - 附带：MRR / NDCG / ROUGE-L / 语义相似度 / faithfulness / answer_relevance
    - agent 过程量：iterations_avg（循环轮数）/ judge_verdict_distribution

关键实现点（防串台）：
    - 同一进程串行跑三列，reranker 模型只加载一次（get_reranker lru_cache）。
    - 切换 agent_reranker_loop_enabled 后必须 build_graph.cache_clear() 让 StateGraph
      重新编译（tools→reflect 的边依赖该开关）；reset_cache() 清语义缓存。
    - 每列用独立 TokenMeter 绑定全局 handler，finally 恢复现场。

用法：
    python scripts/run_eval_rerank_ab.py
    python scripts/run_eval_rerank_ab.py --subset 5 --k 3 5 10
    python scripts/run_eval_rerank_ab.py --dataset my_eval.json --subset 8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

# 原生库载入顺序护栏：pyarrow 必须早于 torch 载入。
# 否则 sentence_transformers → sklearn → pandas → pyarrow 会在 torch(cu126) 的 DLL 已就位后
# 才首次加载 arrow.dll，触发 Windows access violation 直接终止进程——无 Python 异常可捕，
# 现场只在事件日志里留下 arrow.dll 0xc0000005。显式先 import 一次即可稳定绕开。
import pyarrow  # noqa: F401

# 确保 src 在 path 里
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 三列配置：变体名 / 标签 / 是否 agent 链路 / 循环重排开关
VARIANTS = [
    {
        "key": "A",
        "label": "A · 循环重排开 + 图扩展开（生产默认）",
        "target_kind": "agent",
        "loop_rerank": True,
        "graph_expand": True,
    },
    {
        "key": "B",
        "label": "B · 循环重排关 + 图扩展开（仅出口重排）",
        "target_kind": "agent",
        "loop_rerank": False,
        "graph_expand": True,
    },
    {
        "key": "C",
        "label": "C · naive 基线（RAGChain）",
        "target_kind": "naive",
        "loop_rerank": None,  # naive 不涉及该开关
        "graph_expand": None,
    },
    {
        "key": "D",
        "label": "D · 循环重排开 + 图扩展关（隔离图扩展价值）",
        "target_kind": "agent",
        "loop_rerank": True,
        "graph_expand": False,
    },
]

# 分环节耗时图配色（直方图按方式上色，饼图按环节上色）
STAGE_ORDER = ["规划决策", "检索", "图扩展", "重排", "判定", "生成"]
STAGE_COLORS = {
    "规划决策": "#6366f1",
    "检索": "#0ea5e9",
    "图扩展": "#14b8a6",
    "重排": "#f59e0b",
    "判定": "#ec4899",
    "生成": "#84cc16",
}
METHOD_COLORS = {"A": "#3b82f6", "B": "#a855f7", "C": "#f59e0b", "D": "#10b981"}


def build_rag_chain():
    """构建 RAGChain 实例（naive 链路用）。"""
    from note_assistant.pipeline.rag_chain import RAGChain
    from note_assistant.retrieval.hybrid import HybridRetriever
    from note_assistant.retrieval.reranker import LocalReranker
    from note_assistant.generation.generator import Generator

    retriever = HybridRetriever()
    reranker = LocalReranker()
    try:
        generator = Generator()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Generator 初始化失败（答案将为空）: {e}")
        generator = None
    return RAGChain(hybrid_retriever=retriever, reranker=reranker, generator=generator)


def run_variant(variant, dataset, k_values, llm, rag_chain):
    """跑单列评测，返回 (key, EvalReport) 或 (key, None)。"""
    from note_assistant.config import settings
    from note_assistant.agent.agent import build_graph
    from note_assistant.agent.runner import reset_cache
    from note_assistant.evaluation.evaluator import Evaluator

    key = variant["key"]
    logger.info("=" * 70)
    logger.info(f"开始变体 {variant['label']}")
    logger.info("=" * 70)

    if variant["target_kind"] == "agent":
        # 注入 A/B/D 变量并强制重建图（边依赖这些开关），清语义缓存防串台
        settings.agent_reranker_loop_enabled = variant["loop_rerank"]
        if variant.get("graph_expand") is not None:
            settings.agent_graph_expand_enabled = variant["graph_expand"]
        build_graph.cache_clear()
        reset_cache()
        evaluator = Evaluator(None, llm=llm, target_kind="agent")
    else:
        evaluator = Evaluator(rag_chain, llm=llm, target_kind="naive")

    report = evaluator.run(dataset, k_values=k_values)
    logger.info(
        f"变体 {key} 完成：avg_elapsed={report.avg_elapsed_ms:.0f}ms, "
        f"total_tokens={report.token_usage_total['total_tokens'] if report.token_usage_total else 0}, "
        f"iterations_avg={report.iterations_avg}"
    )
    return key, report


def run_warmup():
    """所有变体前跑一次完整 agentic rag 流程预热（reranker 加载 / GPU 首次推理 / 连接）。"""
    import asyncio
    from note_assistant.agent.agent import build_graph
    from note_assistant.agent.runner import ainvoke, reset_cache
    from note_assistant.config import settings
    # 用生产默认（循环重排开 + 图扩展开）预热，避免首个变体首题被冷启动拖慢
    settings.agent_reranker_loop_enabled = True
    settings.agent_graph_expand_enabled = True
    build_graph.cache_clear()
    reset_cache()
    q = "请介绍知识库中关于检索增强生成（RAG）的核心概念与典型流程。"
    try:
        asyncio.run(ainvoke(q, session_id="__warmup__"))
        logger.info("预热完成：reranker / GPU / 首次推理已热，正式评测不再被冷启动拖慢")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"预热失败（不影响正式评测）: {e}")


def _safe(v):
    """把 None / 不可序列化值规整为 JSON 友好形式。"""
    if v is None:
        return None
    try:
        float(v)
    except (TypeError, ValueError):
        return v
    return v


def build_comparison(reports: "OrderedDict[str, object]", k_values: list) -> dict:
    """把三列 EvalReport 收敛成并排对比结构。"""
    comparison: dict = {
        "variants": [],
        "metrics": {
            "avg_elapsed_ms": {},
            "token_total": {},
            "token_prompt": {},
            "token_completion": {},
            "token_cache_read": {},
            "llm_calls": {},
            "llm_cache_hit_rate": {},
            "accuracy": {},
            "confidence": {},
            "rouge_l": {},
            "semantic_similarity": {},
            "faithfulness": {},
            "answer_relevance": {},
            "mrr": {},
            "iterations_avg": {},
            "judge_verdict_distribution": {},
        },
        "recall_at_k": {f"recall@{k}": {} for k in k_values},
        "precision_at_k": {f"precision@{k}": {} for k in k_values},
        "ndcg_at_k": {f"ndcg@{k}": {} for k in k_values},
        # 分环节耗时（avg ms），环节口径由 evaluator._agent_canonical_stages 统一
        "stage_timings_avg": {},
    }
    # 渲染函数（markdown / html）统一从 metrics["recall@{k}"] 扁平取值，
    # 故同步扁平化 recall/precision/ndcg，避免 KeyError 且与 recall_at_k 保持一致。
    for k in k_values:
        comparison["metrics"][f"recall@{k}"] = {}
        comparison["metrics"][f"precision@{k}"] = {}
        comparison["metrics"][f"ndcg@{k}"] = {}

    for key, report in reports.items():
        if report is None:
            continue
        comparison["variants"].append(key)
        gm = report.generation_metrics_avg
        rm = report.retrieval_metrics_avg
        tu = report.token_usage_total or {}

        comparison["metrics"]["avg_elapsed_ms"][key] = round(report.avg_elapsed_ms, 1)
        comparison["metrics"]["token_total"][key] = tu.get("total_tokens")
        comparison["metrics"]["token_prompt"][key] = tu.get("prompt_tokens")
        comparison["metrics"]["token_completion"][key] = tu.get("completion_tokens")
        comparison["metrics"]["token_cache_read"][key] = tu.get("cache_read_tokens")
        comparison["metrics"]["llm_calls"][key] = tu.get("llm_calls")
        comparison["metrics"]["llm_cache_hit_rate"][key] = report.llm_cache_hit_rate
        comparison["metrics"]["accuracy"][key] = _safe(gm.get("accuracy"))
        comparison["metrics"]["confidence"][key] = _safe(gm.get("confidence"))
        comparison["metrics"]["rouge_l"][key] = _safe(gm.get("rouge_l"))
        comparison["metrics"]["semantic_similarity"][key] = _safe(gm.get("semantic_similarity"))
        comparison["metrics"]["faithfulness"][key] = _safe(gm.get("faithfulness"))
        comparison["metrics"]["answer_relevance"][key] = _safe(gm.get("answer_relevance"))
        comparison["metrics"]["mrr"][key] = _safe(rm.get("mrr"))
        comparison["metrics"]["iterations_avg"][key] = (
            round(report.iterations_avg, 2) if report.iterations_avg is not None else None
        )
        comparison["metrics"]["judge_verdict_distribution"][key] = report.judge_verdict_distribution

        # 分环节耗时聚合（per_question 已含 stage_timings，按规范化环节口径对齐 agent/naive）
        _sacc: "dict[str, list]" = defaultdict(list)
        for _pq in report.per_question:
            _st = _pq.get("stage_timings") or {}
            for _s, _v in _st.items():
                if isinstance(_v, (int, float)):
                    _sacc[_s].append(_v)
        comparison["stage_timings_avg"][key] = {
            _s: round(sum(_v) / len(_v), 1) for _s, _v in _sacc.items()
        }

        for k in k_values:
            rk = f"recall@{k}"
            pk = f"precision@{k}"
            nk = f"ndcg@{k}"
            comparison["recall_at_k"][rk][key] = _safe(rm.get(rk))
            comparison["precision_at_k"][pk][key] = _safe(rm.get(pk))
            comparison["ndcg_at_k"][nk][key] = _safe(rm.get(nk))
            # 同步到 metrics 扁平结构，供 markdown / html 渲染取值
            comparison["metrics"][rk][key] = _safe(rm.get(rk))
            comparison["metrics"][pk][key] = _safe(rm.get(pk))
            comparison["metrics"][nk][key] = _safe(rm.get(nk))

    return comparison


def _fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def render_markdown(comparison: dict, k_values: list) -> str:
    """生成并排对比 Markdown 表。"""
    variants = comparison["variants"]
    if not variants:
        return "# 循环内重排 A/B/naive 对照评测\n\n（无可用结果）\n"

    lines = ["# 循环内重排 A/B/naive 对照评测", ""]
    lines.append(f"对比列：{', '.join(variants)}")
    lines.append("")
    lines.append("## 对比列说明")
    lines.append("")
    lines.append("- **A · 循环重排开（生产默认）**：agentic RAG，每轮工具调用后循环内 `rerank_loop` 节点对累积证据做交叉编码精排并裁剪 top-10，再交给 Judge（reflect）决策（配置 `agent_reranker_loop_enabled=True`）。")
    lines.append("- **B · 循环重排关（仅出口重排）**：agentic RAG 结构不变，但跳过 `rerank_loop`（`agent_reranker_loop_enabled=False`），只在出口 `rerank_exit` 做一次精排取 top-5，用于隔离「循环内重排」这一环节的增量价值。")
    lines.append("- **C · naive 基线（RAGChain）**：传统非 agent 链路，history 串联检索 + 单次生成，无 agentic 循环、无 Judge、无语义缓存，作为对照基线。")
    lines.append("")

    def table(metric_label: str, metric_key: str, nd=4):
        head = "| 指标 | " + " | ".join(variants) + " |"
        sep = "| --- | " + " | ".join(["---"] * len(variants)) + " |"
        row = f"| {metric_label} | " + " | ".join(
            _fmt(comparison["metrics"][metric_key].get(v), nd) for v in variants
        ) + " |"
        return "\n".join([head, sep, row])

    lines.append("## 核心指标（用户需求）")
    lines.append("")
    lines.append(table("耗时 avg_elapsed_ms", "avg_elapsed_ms", 1))
    lines.append(table("Token 总量 total", "token_total", 0))
    lines.append(table("准确率 accuracy", "accuracy", 4))
    lines.append(table("置信度 confidence", "confidence", 4))
    for k in k_values:
        lines.append(table(f"召回率 recall@{k}", f"recall@{k}" if False else f"recall@{k}"))
    for k in k_values:
        lines.append(table(f"精确度 precision@{k}", f"precision@{k}"))
    lines.append("")

    lines.append("## 附加指标")
    lines.append("")
    lines.append(table("MRR", "mrr"))
    for k in k_values:
        lines.append(table(f"NDCG@{k}", f"ndcg@{k}"))
    lines.append(table("ROUGE-L", "rouge_l"))
    lines.append(table("语义相似度", "semantic_similarity"))
    lines.append(table("忠诚度 faithfulness", "faithfulness"))
    lines.append(table("答案相关性", "answer_relevance"))
    lines.append("")

    lines.append("## Token 维度明细")
    lines.append("")
    lines.append(table("prompt_tokens", "token_prompt", 0))
    lines.append(table("completion_tokens", "token_completion", 0))
    lines.append(table("cache_read_tokens", "token_cache_read", 0))
    lines.append(table("llm_calls", "llm_calls", 0))
    lines.append(table("token 缓存命中率", "llm_cache_hit_rate", 4))
    lines.append("")

    lines.append("## Agent 过程量（仅 A/B 有意义，C 为 —）")
    lines.append("")
    lines.append(table("平均循环轮数 iterations_avg", "iterations_avg", 2))
    # Judge 判定分布单独渲染
    head = "| verdict | " + " | ".join(variants) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(variants)) + " |"
    dist_rows = []
    all_verdicts = set()
    for v in variants:
        d = comparison["metrics"]["judge_verdict_distribution"].get(v) or {}
        all_verdicts.update(d.keys())
    if all_verdicts:
        for verdict in sorted(all_verdicts):
            row = f"| {verdict} | " + " | ".join(
                str((comparison["metrics"]["judge_verdict_distribution"].get(v) or {}).get(verdict, 0))
                for v in variants
            ) + " |"
            dist_rows.append(row)
        lines.append("\n".join([head, sep] + dist_rows))
    else:
        lines.append("（无 agent 过程量）")
    # ── 分环节耗时表 ──
    lines.append("## 分环节耗时（avg ms，与 naive 对齐口径）")
    lines.append("")
    _sd = comparison.get("stage_timings_avg", {})
    _shead = "| 环节 | " + " | ".join(variants) + " |"
    _ssep = "| --- | " + " | ".join(["---"] * len(variants)) + " |"
    _srows = [_shead, _ssep]
    for _s in STAGE_ORDER:
        _srows.append(
            f"| {_s} | " + " | ".join(
                str(_sd.get(v, {}).get(_s, 0) or 0) for v in variants
            ) + " |"
        )
    _srows.append(
        "| 合计 | " + " | ".join(
            str(round(sum(_sd.get(v, {}).values()), 1)) for v in variants
        ) + " |"
    )
    lines.append("\n".join(_srows))
    lines.append("")

    return "\n".join(lines)


def render_stage_histogram_svg(comparison: dict) -> str:
    """分组直方图：x=环节，每组 N 根柱（A/B/C/D），高度=平均耗时(ms)。"""
    import html as _html
    variants = comparison["variants"]
    sd = comparison.get("stage_timings_avg", {})
    stages = STAGE_ORDER
    maxv = 0.0
    for v in variants:
        for s in stages:
            maxv = max(maxv, float(sd.get(v, {}).get(s, 0) or 0))
    if maxv <= 0:
        return "<p>无分环节耗时数据</p>"
    W, H = 760, 380
    ml, mr, mt, mb = 58, 16, 16, 60
    plot_w = W - ml - mr
    plot_h = H - mt - mb
    group_w = plot_w / len(stages)
    bar_w = min(48.0, group_w * 0.82 / max(1, len(variants)))
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" preserveAspectRatio="xMinYMin meet" '
             f'font-family="-apple-system,Segoe UI,Microsoft YaHei,sans-serif">']
    for i in range(4):
        y = mt + plot_h * i / 3
        val = maxv * (1 - i / 3)
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{W-mr}" y2="{y:.1f}" stroke="#eef2f7"/>')
        parts.append(f'<text x="{ml-8}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="#6b7280">{val:.0f}</text>')
    for si, s in enumerate(stages):
        gx = ml + group_w * si
        for vi, v in enumerate(variants):
            val = float(sd.get(v, {}).get(s, 0) or 0)
            h = plot_h * (val / maxv)
            x = gx + (group_w - bar_w * len(variants)) / 2 + vi * bar_w
            y = mt + plot_h - h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-3:.1f}" height="{max(0.0,h):.1f}" '
                f'fill="{METHOD_COLORS.get(v, "#64748b")}" rx="2">'
                f'<title>{_html.escape(v)} · {_html.escape(s)}: {val:.0f} ms</title></rect>'
            )
        parts.append(f'<text x="{gx+group_w/2:.1f}" y="{H-mb+18:.1f}" text-anchor="middle" '
                     f'font-size="12" fill="#374151">{_html.escape(s)}</text>')
    parts.append(f'<line x1="{ml}" y1="{mt+plot_h:.1f}" x2="{W-mr}" y2="{mt+plot_h:.1f}" stroke="#9ca3af"/>')
    parts.append("</svg>")
    return "".join(parts)


def render_stage_pie_svg(comparison: dict) -> str:
    """每列一张饼图：该方式下各环节时间占比。"""
    import html as _html
    import math
    variants = comparison["variants"]
    sd = comparison.get("stage_timings_avg", {})
    size = 190
    cx = cy = size / 2
    r = size / 2 - 18
    pies = []
    for v in variants:
        data = sd.get(v, {})
        total = sum(float(x) for x in data.values() if isinstance(x, (int, float)))
        slices = []
        if total > 0:
            ang = -90.0
            for s in STAGE_ORDER:
                val = float(data.get(s, 0) or 0)
                if val <= 0:
                    continue
                frac = val / total
                a1 = ang + frac * 360
                large = 1 if (a1 - ang) > 180 else 0
                x0 = cx + r * math.cos(math.radians(ang)); y0 = cy + r * math.sin(math.radians(ang))
                x1 = cx + r * math.cos(math.radians(a1)); y1 = cy + r * math.sin(math.radians(a1))
                slices.append(
                    f'<path d="M {cx:.1f} {cy:.1f} L {x0:.1f} {y0:.1f} '
                    f'A {r:.1f} {r:.1f} 0 {large} 1 {x1:.1f} {y1:.1f} Z" '
                    f'fill="{STAGE_COLORS[s]}" stroke="#fff" stroke-width="1">'
                    f'<title>{_html.escape(s)}: {val:.0f} ms ({frac*100:.0f}%)</title></path>'
                )
                ang = a1
        pies.append(
            f'<div class="piecard"><svg viewBox="0 0 {size} {size}" width="170" height="170">'
            f'{"".join(slices)}'
            f'<text x="{cx:.1f}" y="{cy-2:.1f}" text-anchor="middle" font-size="14" '
            f'font-weight="600" fill="{METHOD_COLORS.get(v, "#374151")}">{_html.escape(v)}</text>'
            f'<text x="{cx:.1f}" y="{cy+15:.1f}" text-anchor="middle" font-size="10" fill="#6b7280">'
            f'{total:.0f} ms</text></svg></div>'
        )
    legend = "".join(
        f'<span class="lg"><i style="background:{STAGE_COLORS[s]}"></i>{_html.escape(s)}</span>'
        for s in STAGE_ORDER
    )
    method_legend = "".join(
        f'<span class="lg"><i style="background:{METHOD_COLORS.get(v, "#64748b")}"></i>{_html.escape(v)}</span>'
        for v in variants
    )
    return (f'<div class="pies">{"".join(pies)}</div>'
            f'<div class="legend"><b>环节</b> {legend}</div>'
            f'<div class="legend"><b>方式</b> {method_legend}</div>')


def render_html(comparison: dict, k_values: list) -> str:
    """生成自包含 HTML 对比图（CSS 横向条形图，无外部依赖）。"""
    import html as _html

    variants = comparison["variants"]
    if not variants:
        return "<h1>循环内重排 A/B/naive 对照评测</h1><p>无可用结果</p>"

    # 颜色（浅色块区分各列，呼应可视化约定）
    colors = {"A": "#3b82f6", "B": "#a855f7", "C": "#f59e0b", "D": "#10b981"}
    color_of = lambda v: colors.get(v, "#64748b")

    def bars(metric_key, fmt="{:.4f}", higher_better=True):
        vals = {v: comparison["metrics"][metric_key].get(v) for v in variants}
        nums = [x for x in vals.values() if isinstance(x, (int, float))]
        if not nums:
            return ""
        lo, hi = min(nums), max(nums)
        span = (hi - lo) or 1.0
        rows = []
        for v in variants:
            x = vals[v]
            if not isinstance(x, (int, float)):
                pct = 0
                label = "—"
            else:
                pct = 100.0 * (x - lo) / span if hi != lo else 100.0
                # 归一化到 60%~100% 视觉区间，避免 0 值条不可见
                pct = 60.0 + 0.4 * pct
                label = fmt.format(x)
            rows.append(
                f'<div class="barrow"><span class="blab">{_html.escape(str(v))}</span>'
                f'<div class="track"><div class="fill" style="width:{pct:.1f}%;'
                f'background:{color_of(v)}"></div></div>'
                f'<span class="bval">{label}</span></div>'
            )
        return "\n".join(rows)

    sections = []
    # 核心指标卡
    core = [
        ("耗时 (ms)", "avg_elapsed_ms", "{:.0f}"),
        ("Token 总量", "token_total", "{:.0f}"),
        ("准确率 accuracy", "accuracy", "{:.4f}"),
        ("置信度 confidence", "confidence", "{:.4f}"),
    ]
    for k in k_values:
        core.append((f"召回率 recall@{k}", f"recall@{k}", "{:.4f}"))
    for k in k_values:
        core.append((f"精确度 precision@{k}", f"precision@{k}", "{:.4f}"))

    blocks = []
    for title, key, fmt in core:
        blocks.append(f'<div class="card"><h3>{_html.escape(title)}</h3>{bars(key, fmt)}</div>')
    sections.append(('<h2>核心指标（用户需求）</h2>', '<div class="grid">' + "".join(blocks) + "</div>"))

    # Token 明细
    tok_blocks = []
    for title, key, fmt in [
        ("prompt tokens", "token_prompt", "{:.0f}"),
        ("completion tokens", "token_completion", "{:.0f}"),
        ("cache_read tokens", "token_cache_read", "{:.0f}"),
        ("llm_calls", "llm_calls", "{:.0f}"),
        ("token 缓存命中率", "llm_cache_hit_rate", "{:.4f}"),
    ]:
        tok_blocks.append(f'<div class="card"><h3>{_html.escape(title)}</h3>{bars(key, fmt)}</div>')
    sections.append(('<h2>Token 维度明细</h2>', '<div class="grid">' + "".join(tok_blocks) + "</div>"))

    # Agent 过程量
    proc_blocks = [
        f'<div class="card"><h3>平均循环轮数</h3>{bars("iterations_avg", "{:.2f}")}</div>'
    ]
    # Judge 分布表
    all_verdicts = set()
    for v in variants:
        d = comparison["metrics"]["judge_verdict_distribution"].get(v) or {}
        all_verdicts.update(d.keys())
    if all_verdicts:
        th = "<th>verdict</th>" + "".join(f"<th>{_html.escape(v)}</th>" for v in variants)
        trs = []
        for verdict in sorted(all_verdicts):
            tds = f"<td>{_html.escape(verdict)}</td>" + "".join(
                f"<td>{(comparison['metrics']['judge_verdict_distribution'].get(v) or {}).get(verdict, 0)}</td>"
                for v in variants
            )
            trs.append(f"<tr>{tds}</tr>")
        proc_blocks.append(
            f'<div class="card"><h3>Judge 判定分布</h3>'
            f'<table class="vtbl"><tr>{th}</tr>{"".join(trs)}</table></div>'
        )
    sections.append(('<h2>Agent 过程量（仅 A/B）</h2>', '<div class="grid">' + "".join(proc_blocks) + "</div>"))

    # ── A/B/C 定义说明卡片（避免报告读不懂各列含义）──
    variant_meta = {v["key"]: v["label"] for v in VARIANTS}
    variant_desc = {
        "A": "循环内重排 <b>开启</b>（生产默认）。每轮工具调用后，循环内 <code>rerank_loop</code> 节点对累积证据做交叉编码精排并裁剪 top-10，再交给 Judge（reflect）决策。配置 <code>agent_reranker_loop_enabled=True</code>。",
        "B": "循环内重排 <b>关闭</b>。agentic RAG 结构不变，但跳过 <code>rerank_loop</code>（<code>agent_reranker_loop_enabled=False</code>），只在出口 <code>rerank_exit</code> 做一次精排取 top-5。用于隔离「循环内重排」这一环节的增量价值。",
        "C": "naive 基线（RAGChain）。传统非 agent 链路：history 串联检索 + 单次生成，无 agentic 循环、无 Judge、无语义缓存。作为对照基线。",
        "D": "循环内重排 <b>开启</b> + 图扩展 <b>关闭</b>。与 A 唯一差异是关闭 <code>agent_graph_expand_enabled</code>，"
             "用于隔离「沿 wikilinks 自动扩展关联笔记」这一环节的增量价值（耗时 / 召回 / 质量）。",
    }
    def_cards = []
    for v in variants:
        c = color_of(v)
        lab = _html.escape(variant_meta.get(v, v))
        dsc = variant_desc.get(v, "")
        def_cards.append(
            f'<div class="defcard" style="border-left:4px solid {c}">'
            f'<div class="defkey"><span class="dot" style="background:{c}"></span>'
            f'{_html.escape(v)} · {lab}</div>'
            f'<div class="defbody">{dsc}</div></div>'
        )
    defs_html = '<div class="defs">' + "".join(def_cards) + "</div>"

    # ── 分环节耗时 直方图 + 饼图 ──
    stage_html = render_stage_histogram_svg(comparison)
    pie_html = render_stage_pie_svg(comparison)
    sections.append((
        '<h2>分环节耗时占比（avg ms）</h2>',
        f'<div class="chartcard"><h3>直方图：各环节平均耗时（ms）</h3>{stage_html}</div>'
        f'<div class="chartcard"><h3>饼图：各环节时间占比（每列一种方式）</h3>{pie_html}</div>',
    ))

    sections_html = "\n".join(h + body for h, body in sections)

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>循环内重排 A/B/naive 对照评测</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
         background:#f7f8fa; color:#1f2937; margin:0; padding:24px; }}
  h1 {{ color:#111827; }}
  h2 {{ color:#374151; border-left:4px solid #3b82f6; padding-left:10px; margin-top:28px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:14px; }}
  .card {{ background:#fff; border:1px solid #e5e7eb; border-radius:10px; padding:14px 16px;
          box-shadow:0 1px 2px rgba(0,0,0,.04); }}
  .card h3 {{ margin:0 0 10px; font-size:14px; color:#4b5563; }}
  .barrow {{ display:flex; align-items:center; gap:8px; margin:6px 0; font-size:13px; }}
  .blab {{ width:18px; font-weight:600; }}
  .track {{ flex:1; background:#eef2f7; border-radius:6px; height:16px; overflow:hidden; }}
  .fill {{ height:100%; border-radius:6px; transition:width .3s; }}
  .bval {{ width:64px; text-align:right; color:#374151; font-variant-numeric:tabular-nums; }}
  table.vtbl {{ border-collapse:collapse; width:100%; font-size:13px; }}
  table.vtbl th, table.vtbl td {{ border:1px solid #e5e7eb; padding:6px 8px; text-align:center; }}
  table.vtbl th {{ background:#f3f4f6; }}
  .defs {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:14px; margin:18px 0 6px; }}
  .defcard {{ background:#fff; border:1px solid #e5e7eb; border-radius:10px; padding:14px 16px; box-shadow:0 1px 2px rgba(0,0,0,.04); }}
  .defkey {{ font-weight:600; font-size:14px; color:#111827; margin-bottom:6px; display:flex; align-items:center; gap:8px; }}
  .dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
  .defbody {{ font-size:13px; color:#4b5563; line-height:1.65; }}
  .defbody code {{ background:#f3f4f6; padding:1px 5px; border-radius:4px; font-size:12px; }}
  .chartcard {{ background:#fff; border:1px solid #e5e7eb; border-radius:10px; padding:14px 16px; margin:10px 0; box-shadow:0 1px 2px rgba(0,0,0,.04); }}
  .chartcard h3 {{ margin:0 0 10px; font-size:14px; color:#4b5563; }}
  .pies {{ display:flex; flex-wrap:wrap; gap:14px; }}
  .piecard {{ text-align:center; }}
  .piecard svg {{ display:block; }}
  .legend {{ margin-top:10px; font-size:12px; color:#4b5563; display:flex; flex-wrap:wrap; gap:10px; align-items:center; }}
  .lg {{ display:inline-flex; align-items:center; gap:4px; }}
  .lg i {{ width:12px; height:12px; border-radius:3px; display:inline-block; }}
</style>
</head>
<body>
<h1>循环内重排 A/B/naive 对照评测</h1>
<p>对比列：{' / '.join(_html.escape(v) for v in variants)} — 条形按各指标内相对高低归一化（越长越好）。</p>
{defs_html}
{sections_html}
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="循环内重排 A/B/naive 对照评测")
    parser.add_argument("--dataset", "-d", default="builtin", help="数据集（builtin 或 JSON 路径）")
    parser.add_argument("--subset", "-n", type=int, default=5, help="取前 N 条（pilot 默认 5）")
    parser.add_argument("--k", nargs="+", type=int, default=[3, 5, 10], help="检索指标截断点")
    parser.add_argument("--output-dir", "-o", default="./eval/report", help="报告输出目录")
    parser.add_argument("--variants", nargs="+", choices=[v["key"] for v in VARIANTS], default=None,
                        help="只跑指定列（如 --variants A 单跑生产默认列）；缺省=全跑")
    args = parser.parse_args()

    from note_assistant.evaluation.eval_dataset import get_builtin_dataset, load_eval_dataset
    from note_assistant.llm.client import get_llm

    if args.dataset == "builtin":
        full = get_builtin_dataset()
    else:
        full = load_eval_dataset(args.dataset)
    dataset = full.subset(args.subset)
    logger.info(f"评测集: {dataset.name}, {dataset.size} 条问题")

    # 评测用 LLM（准确率/置信度/faithfulness/answer_relevance 复用主通道）
    llm = get_llm(temperature=0.0)
    rag_chain = build_rag_chain()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 预热：所有变体前跑一次完整 agentic rag 流程，避免首题被冷启动（reranker/GPU/连接）拖慢
    run_warmup()
    selected = [v for v in VARIANTS if not args.variants or v["key"] in args.variants]
    logger.info(f"本次跑列：{', '.join(v['key'] for v in selected)}")
    reports: "OrderedDict[str, object]" = OrderedDict()
    for variant in selected:
        try:
            key, report = run_variant(variant, dataset, args.k, llm, rag_chain)
            reports[key] = report
            # 变体间退避，缓解 agnes 免费额度 429 速率限制
            time.sleep(30)
            # 每变体立即落盘中间结果，避免最终合并阶段崩导致整轮数据丢失
            try:
                (out_dir / f"rerank_ab_{key}.json").write_text(
                    json.dumps(report.to_dict() if report else None, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                logger.info(f"变体 {key} 中间结果已落盘")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"变体 {key} 中间落盘失败（不影响后续）: {e}")
        except Exception as e:  # noqa: BLE001
            logger.exception(f"变体 {variant['key']} 评测失败，跳过: {e}")
            reports[variant["key"]] = None

    # 汇总 + 输出
    comparison = build_comparison(reports, args.k)

    def _write(path, content):
        try:
            path.write_text(content, encoding="utf-8")
            return path
        except PermissionError:
            # 可能被预览面板/IDE 占用，fallback 到带 _10 后缀的文件，绝不丢数据
            alt = path.with_name(path.stem + "_10" + path.suffix)
            logger.warning(f"写入 {path} 被拒（可能被预览面板占用），已改写到 {alt}")
            alt.write_text(content, encoding="utf-8")
            return alt

    # JSON：合并三列原始报告 + 对比摘要
    raw = {k: (r.to_dict() if r else None) for k, r in reports.items()}
    json_path = _write(out_dir / "rerank_ab_compare.json",
        json.dumps({"comparison": comparison, "reports": raw}, ensure_ascii=False, indent=2))
    md_path = _write(out_dir / "rerank_ab_compare.md", render_markdown(comparison, args.k))
    html_path = _write(out_dir / "rerank_ab_compare.html", render_html(comparison, args.k))

    logger.info("")
    logger.info("=" * 70)
    logger.info("对比摘要（核心指标）")
    logger.info("=" * 70)
    for label, key in [
        ("耗时ms", "avg_elapsed_ms"), ("token总", "token_total"),
        ("准确率", "accuracy"), ("置信度", "confidence"),
    ]:
        cells = " | ".join(
            f"{v}:{_fmt(comparison['metrics'][key].get(v))}" for v in comparison["variants"]
        )
        logger.info(f"  {label:>6} | {cells}")
    for k in args.k:
        key = f"recall@{k}"
        cells = " | ".join(
            f"{v}:{_fmt(comparison['metrics'][key].get(v))}" for v in comparison["variants"]
        )
        logger.info(f"  recall@{k:<2} | {cells}")
    logger.info("")
    logger.info(f"JSON : {json_path}")
    logger.info(f"MD   : {md_path}")
    logger.info(f"HTML : {html_path}")


if __name__ == "__main__":
    main()
