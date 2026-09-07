#!/usr/bin/env python3
"""
多列「同一链路（方式 A）、不同索引参数」评测报告并排对比。

列与列之间只允许差索引参数（chunk_size / chunk_overlap / chunking_strategy），
因此检索、生成、token、耗时上的差异可直接归因到切分。

用法：
    python scripts/_compare_chunk_ab.py \
        --arm "v2b·500/100=eval/report/chunk_v2b_500_100/rerank_ab_A.json" \
        --arm "v2b·800/150=eval/report/chunk_v2b_800_150/rerank_ab_A.json" \
        --out eval/report/chunk_ab_compare

若某列 json 同目录下有 _meta.json（索引侧统计：chunks / 重建耗时等），自动并入「索引侧差异」表。
产出 <out>.md / <out>.html / <out>.json。
"""
from __future__ import annotations

import argparse
import html as _html
import json
import sys
from collections import defaultdict
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

PALETTE = ["#3b82f6", "#a855f7", "#f59e0b", "#10b981", "#ef4444", "#0ea5e9", "#84cc16", "#ec4899"]

METRIC_GROUPS = [
    ("索引侧差异", [
        ("子块 chunks 数", "chunks", "int"),
        ("全量重建耗时 s", "reindex_seconds", "int"),
        ("chunk_size", "chunk_size", "int"),
        ("chunk_overlap", "chunk_overlap", "int"),
        ("chunking_strategy", "chunking_strategy", "str"),
    ]),
    ("检索指标", [
        ("recall@3", "recall@3", "f4"), ("recall@5", "recall@5", "f4"), ("recall@10", "recall@10", "f4"),
        ("precision@3", "precision@3", "f4"), ("precision@5", "precision@5", "f4"), ("precision@10", "precision@10", "f4"),
        ("ndcg@3", "ndcg@3", "f4"), ("ndcg@5", "ndcg@5", "f4"), ("ndcg@10", "ndcg@10", "f4"),
        ("MRR", "mrr", "f4"),
    ]),
    ("生成质量", [
        ("准确率 accuracy", "accuracy", "f4"), ("置信度 confidence", "confidence", "f4"),
        ("忠诚度 faithfulness", "faithfulness", "f4"), ("答案相关性 answer_relevance", "answer_relevance", "f4"),
        ("ROUGE-L", "rouge_l", "f4"), ("语义相似度 semantic_similarity", "semantic_similarity", "f4"),
    ]),
    ("成本与耗时", [
        ("平均耗时 ms", "_avg_elapsed_ms", "int"),
        ("Token 总量", "_total_tokens", "int"),
        ("prompt tokens", "_prompt_tokens", "int"),
        ("completion tokens", "_completion_tokens", "int"),
        ("LLM 调用次数", "_llm_calls", "int"),
        ("平均循环轮数", "_iterations_avg", "f2"),
        ("LLM token 缓存命中率", "_llm_cache_hit_rate", "f4"),
    ]),
]

HIGHER_BETTER = {
    "recall@3", "recall@5", "recall@10", "precision@3", "precision@5", "precision@10",
    "ndcg@3", "ndcg@5", "ndcg@10", "mrr", "accuracy", "confidence", "faithfulness",
    "answer_relevance", "rouge_l", "semantic_similarity", "_llm_cache_hit_rate",
}
LOWER_BETTER = {"_avg_elapsed_ms", "_total_tokens", "_prompt_tokens", "_completion_tokens", "_llm_calls"}


def gold_map() -> dict:
    """问题原文 → 应命中文件（EvalReport.per_question 不带 relevant_files，需回查数据集）。"""
    from note_assistant.evaluation.eval_dataset import get_builtin_dataset
    return {q.question: list(q.relevant_files) for q in get_builtin_dataset().questions}


def flatten(report: dict) -> dict:
    """把一份 EvalReport dict 收敛成「指标名 → 值」的扁平表。"""
    out: dict = {}
    out.update(report.get("retrieval_metrics_avg") or {})
    out.update(report.get("generation_metrics_avg") or {})
    out["_avg_elapsed_ms"] = report.get("avg_elapsed_ms")
    out["_iterations_avg"] = report.get("iterations_avg")
    out["_llm_cache_hit_rate"] = report.get("llm_cache_hit_rate")
    tu = report.get("token_usage_total") or {}
    out["_total_tokens"] = tu.get("total_tokens")
    out["_prompt_tokens"] = tu.get("prompt_tokens")
    out["_completion_tokens"] = tu.get("completion_tokens")
    out["_llm_calls"] = tu.get("llm_calls")
    return out


def stage_averages(report: dict) -> dict:
    acc: dict = defaultdict(list)
    for pq in report.get("per_question") or []:
        for s, v in (pq.get("stage_timings") or {}).items():
            if isinstance(v, (int, float)):
                acc[s].append(v)
    return {s: round(sum(v) / len(v), 1) for s, v in acc.items()}


def parse_arm(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(f"--arm 需为 LABEL=PATH 形式，收到：{spec}")
    label, path = spec.split("=", 1)
    return label.strip(), Path(path.strip())


def fmt(v, kind="f4"):
    if v is None:
        return "—"
    if kind == "str":
        return str(v)
    if not isinstance(v, (int, float)):
        return str(v)
    if kind == "int":
        return f"{round(v):,}"
    if kind == "f2":
        return f"{v:.2f}"
    return f"{v:.4f}"


def best_arm(labels: list, vals: dict, key: str):
    """按指标方向挑出该行的最优列；描述性参数列（切分大小/策略）不比优劣。"""
    if key in ("chunks", "reindex_seconds", "chunk_size", "chunk_overlap", "chunking_strategy"):
        return None
    nums = {l: v for l, v in vals.items() if isinstance(v, (int, float))}
    if len(nums) < 2 or len(set(nums.values())) < 2:
        return None
    if key in LOWER_BETTER:
        return min(nums, key=lambda l: nums[l])
    # 默认越大越好；未登记的指标（如 iterations）不比优劣
    if key not in HIGHER_BETTER:
        return None
    return max(nums, key=lambda l: nums[l])


def main():
    ap = argparse.ArgumentParser(description="多列切分参数评测对比")
    ap.add_argument("--arm", action="append", required=True, dest="arms",
                    help="LABEL=PATH，可重复；顺序即列顺序")
    ap.add_argument("--out", default="eval/report/chunk_ab_compare")
    ap.add_argument("--baseline", default=None, help="Δ 基准列标签（默认第一列）")
    args = ap.parse_args()

    parsed = [parse_arm(s) for s in args.arms]
    missing = [str(p) for _, p in parsed if not p.exists()]
    if missing:
        raise SystemExit("以下评测报告不存在：\n  " + "\n  ".join(missing))

    labels = [l for l, _ in parsed]
    baseline = args.baseline or labels[0]
    if baseline not in labels:
        raise SystemExit(f"--baseline {baseline} 不在列中：{labels}")

    reports, flats, metas, stages = {}, {}, {}, {}
    for label, path in parsed:
        rep = json.loads(path.read_text(encoding="utf-8"))
        reports[label] = rep
        flats[label] = flatten(rep)
        stages[label] = stage_averages(rep)
        mp = path.parent / "_meta.json"
        metas[label] = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}

    def value_of(label, key, kind):
        if key in ("chunks", "reindex_seconds", "chunk_size", "chunk_overlap", "chunking_strategy"):
            v = metas[label].get(key)
        else:
            v = flats[label].get(key)
        if kind == "str" and v is None:
            v = metas[label].get("chunking_strategy", "?")
        return v

    n_q = max(len(r.get("per_question") or []) for r in reports.values())
    k_values = sorted({k.replace("recall@", "") for l in labels
                       for k in flats[l] if k.startswith("recall@")}, key=int)

    L: list[str] = []  # markdown 行
    L.append("# 切分参数对照评测（各列均为方式 A：循环重排开 + 图扩展开，生产默认链路）")
    L.append("")
    L.append("| 列 | 报告 | 题数 |")
    L.append("| --- | --- | --- |")
    for label, path in parsed:
        L.append(f"| {label} | `{path}` | {reports[label].get('total_questions')} |")
    L.append("")
    L.append(f"🏆 = 该指标行里的最优列（方向已按「越大越好 / 越小越好」判定）；"
             f"评测集 `{next(iter(reports.values())).get('dataset_name')}`，{n_q} 题。")
    L.append("")

    summary = {"arms": [], "baseline": baseline, "metrics": {}, "per_question": [], "notes": []}
    for label, path in parsed:
        summary["arms"].append({"label": label, "report": str(path), "meta": metas[label],
                                "total_questions": reports[label].get("total_questions")})

    # ── 各指标组：一行一指标，一列一方式 ──
    for group, items in METRIC_GROUPS:
        if group == "索引侧差异" and not any(metas.values()):
            continue
        rows = []
        for label_, key, kind in items:
            vals = {l: value_of(l, key, kind) for l in labels}
            if all(v is None for v in vals.values()):
                continue
            b = best_arm(labels, vals, key)
            cells = []
            for l in labels:
                cell = fmt(vals[l], kind)
                if l == b and kind != "str":
                    cell += " 🏆"
                cells.append(cell)
            rows.append((label_, cells))
            summary["metrics"][key] = {"label": label_, "group": group,
                                       "values": {l: vals[l] for l in labels}}
        if not rows:
            continue
        L.append(f"## {group}")
        L.append("")
        L.append("| 指标 | " + " | ".join(labels) + " |")
        L.append("| --- | " + " | ".join(["---"] * len(labels)) + " |")
        for label_, cells in rows:
            L.append(f"| {label_} | " + " | ".join(cells) + " |")
        L.append("")

    # ── 分环节耗时 ──
    all_stages = sorted({s for d in stages.values() for s in d})
    if all_stages:
        L.append("## 分环节平均耗时（ms）")
        L.append("")
        L.append("| 环节 | " + " | ".join(labels) + " |")
        L.append("| --- | " + " | ".join(["---"] * len(labels)) + " |")
        for s in all_stages:
            L.append(f"| {s} | " + " | ".join(fmt(stages[l].get(s), "int") for l in labels) + " |")
        L.append(f"| 合计 | " + " | ".join(
            fmt(sum(stages[l].values()), "int") for l in labels) + " |")
        L.append("")
        summary["stage_timings_avg"] = stages

    # ── Judge 判定分布 ──
    verdicts = sorted({v for l in labels for v in (reports[l].get("judge_verdict_distribution") or {})})
    if verdicts:
        L.append("## Judge 判定分布")
        L.append("")
        L.append("| verdict | " + " | ".join(labels) + " |")
        L.append("| --- | " + " | ".join(["---"] * len(labels)) + " |")
        for v in verdicts:
            L.append(f"| {v} | " + " | ".join(
                str((reports[l].get("judge_verdict_distribution") or {}).get(v, 0)) for l in labels) + " |")
        L.append("")
        summary["judge_verdict_distribution"] = {
            l: reports[l].get("judge_verdict_distribution") or {} for l in labels}

    # ── 逐题明细 ──
    gm = gold_map()
    L.append("## 逐题检索命中（分子=命中金标文件数，recall@5）")
    L.append("")
    L.append("| # | 问题 | " + " | ".join(labels) + " | 应命中 |")
    L.append("| --- | --- | " + " | ".join(["---"] * len(labels)) + " | --- |")
    pq_by_arm = {l: (reports[l].get("per_question") or []) for l in labels}
    for i in range(n_q):
        rows_q = {l: (pq_by_arm[l][i] if i < len(pq_by_arm[l]) else {}) for l in labels}
        question = next((rows_q[l].get("question") for l in labels if rows_q[l].get("question")), "")
        rel = sorted({Path(f).name for f in gm.get(question, [])})
        rel_set = set(rel)
        hits = {l: {Path(f).name for f in (rows_q[l].get("retrieved_files") or [])} & rel_set
                for l in labels}
        cells = []
        for l in labels:
            r5 = (rows_q[l].get("retrieval_metrics") or {}).get("recall@5")
            acc = (rows_q[l].get("generation_metrics") or {}).get("accuracy")
            cells.append(f"{fmt(r5)} / {len(hits[l])} (acc {fmt(acc, 'f2')})")
        L.append(f"| {i + 1} | {question[:22]} | " + " | ".join(cells) + f" | {', '.join(rel) or '—'} |")
        # 命中集合不一致时，明确列出"只有谁命中/只有谁漏了"
        hit_sets = {l: hits[l] for l in labels if hits[l]}
        union = set().union(*hit_sets.values()) if hit_sets else set()
        missing_by = {l: sorted(union - hits[l]) for l in labels if union - hits[l]}
        if missing_by:
            L.append(f"|  | ↳ 各列漏检 | " + " | ".join(
                f"{', '.join(missing_by.get(l, [])) or '—'}" for l in labels) + " |  |")
        summary["per_question"].append({
            "index": i, "question": question, "relevant": rel,
            "per_arm": {l: {
                "recall@5": (rows_q[l].get("retrieval_metrics") or {}).get("recall@5"),
                "accuracy": (rows_q[l].get("generation_metrics") or {}).get("accuracy"),
                "iterations": rows_q[l].get("iterations"),
                "hits": sorted(hits[l]), "missed": sorted(rel_set - hits[l]),
                "n_retrieved_files": len(set(rows_q[l].get("retrieved_files") or [])),
            } for l in labels},
        })
    L.append("")

    # ── 结论提示：样本量 ──
    L.append("## 读表须知")
    L.append("")
    L.append(f"- 每列 {n_q} 题，单题召回波动即改变整体 recall 达 {100.0 / max(1, n_q) * (1 / max(1, sum(len(v) for v in gm.values()) / len(gm))):.0f}% 量级；"
             f"两列差异小于 2 个金标文件命中时，属噪声区间，不能据此下结论。")
    L.append("- 各列均为方式 A 单次运行，agentic 循环与 LLM-as-judge 存在非确定性；accuracy/confidence 的列间小差异不可当作稳定结论。")
    L.append("- `v2` 无父块，喂给生成端的就是子块本身；`v2b` 命中子块后回退整节（≤ parent_chunk_size=2000）。"
             "因此 chunk_size 在 v2 上同时改变「检索粒度 + 生成上下文长度」，在 v2b 上只改变「检索粒度」。")
    summary["notes"] = L[L.index("## 读表须知") + 1:]

    out_base = Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    out_base.with_suffix(".md").write_text("\n".join(L), encoding="utf-8")
    out_base.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out_base.with_suffix(".html").write_text(
        render_html(labels, summary, stages, reports, n_q, palette=len(labels)), encoding="utf-8")

    print("对比产出：")
    for suf in (".md", ".html", ".json"):
        print(f"  {out_base.with_suffix(suf)}")


def render_html(labels, summary, stages, reports, n_q, palette) -> str:
    """自包含 HTML：每张指标一组横向条，条长按该指标内相对高低归一化。"""
    colors = {l: PALETTE[i % len(PALETTE)] for i, l in enumerate(labels)}
    metrics = summary["metrics"]

    def bars(key, kind):
        vals = metrics[key]["values"]
        nums = [v for v in vals.values() if isinstance(v, (int, float))]
        if len(nums) < 2:
            return "".join(
                f'<div class="barrow"><span class="blab" style="color:{colors[l]}">{_html.escape(l)}</span>'
                f'<div class="track"></div><span class="bval">{_html.escape(fmt(vals[l], kind))}</span></div>'
                for l in labels)
        lo, hi = min(nums), max(nums)
        span = (hi - lo) or 1.0
        rows = []
        for l in labels:
            x = vals[l]
            if isinstance(x, (int, float)):
                pct = 60.0 + 0.4 * (100.0 * (x - lo) / span if hi != lo else 100.0)
                lab = fmt(x, kind)
            else:
                pct, lab = 0, "—"
            rows.append(
                f'<div class="barrow"><span class="blab" style="color:{colors[l]}">{_html.escape(l)}</span>'
                f'<div class="track"><div class="fill" style="width:{pct:.1f}%;background:{colors[l]}"></div></div>'
                f'<span class="bval">{_html.escape(lab)}</span></div>')
        return "\n".join(rows)

    sections = []
    for group, items in METRIC_GROUPS:
        blocks = []
        for label_, key, kind in items:
            if key not in metrics:
                continue
            blocks.append(f'<div class="card"><h3>{_html.escape(label_)}</h3>{bars(key, kind)}</div>')
        if blocks:
            sections.append((group, '<div class="grid">' + "".join(blocks) + "</div>"))

    # 分环节耗时分组柱状图
    all_stages = sorted({s for d in stages.values() for s in d})
    if all_stages:
        maxv = max((v for d in stages.values() for v in d.values()), default=1.0) or 1.0
        W, H = 820, 380
        ml, mr, mt, mb = 58, 16, 16, 74
        pw, ph = W - ml - mr, H - mt - mb
        gw = pw / len(all_stages)
        bw = min(48.0, gw * 0.82 / max(1, len(labels)))
        parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" preserveAspectRatio="xMinYMin meet" '
                 f'font-family="-apple-system,Segoe UI,Microsoft YaHei,sans-serif">']
        for i in range(4):
            y = mt + ph * i / 3
            parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{W - mr}" y2="{y:.1f}" stroke="#eef2f7"/>')
            parts.append(f'<text x="{ml - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" '
                         f'fill="#6b7280">{maxv * (1 - i / 3):.0f}</text>')
        for si, s in enumerate(all_stages):
            gx = ml + gw * si
            for vi, l in enumerate(labels):
                v = float(stages[l].get(s, 0) or 0)
                h = ph * (v / maxv)
                x = gx + (gw - bw * len(labels)) / 2 + vi * bw
                parts.append(
                    f'<rect x="{x:.1f}" y="{mt + ph - h:.1f}" width="{bw - 3:.1f}" height="{max(0.0, h):.1f}" '
                    f'fill="{colors[l]}" rx="2"><title>{_html.escape(l)} · {_html.escape(s)}: {v:.0f} ms</title></rect>')
            parts.append(f'<text x="{gx + gw / 2:.1f}" y="{H - mb + 18:.1f}" text-anchor="middle" '
                         f'font-size="12" fill="#374151">{_html.escape(s)}</text>')
        parts.append(f'<line x1="{ml}" y1="{mt + ph:.1f}" x2="{W - mr}" y2="{mt + ph:.1f}" stroke="#9ca3af"/>')
        parts.append("</svg>")
        legend = "".join(f'<span class="lg"><i style="background:{colors[l]}"></i>{_html.escape(l)}</span>'
                         for l in labels)
        sections.append(("分环节平均耗时（ms）",
                         f'<div class="chartcard"><h3>各环节平均耗时，每环节一组（跨列对比）</h3>'
                         f'{"".join(parts)}<div class="legend">{legend}</div></div>'))

    # 逐题命中表
    pq = summary["per_question"]
    if pq:
        th = "<th>#</th><th>问题</th>" + "".join(
            f'<th style="color:{colors[l]}">{_html.escape(l)}</th>' for l in labels) + "<th>应命中</th>"
        trs = []
        for row in pq:
            tds = [f'<td>{row["index"] + 1}</td>', f'<td class="q">{_html.escape(row["question"][:26])}</td>']
            for l in labels:
                d = row["per_arm"][l]
                missed = d["missed"]
                cls = "good" if not missed else "bad"
                tds.append(
                    f'<td class="{cls}">{fmt(d.get("recall@5"))} / {len(d["hits"])}'
                    f'<div class="sub">acc {fmt(d.get("accuracy"), "f2")} · iter {d.get("iterations")}</div>'
                    f'{f"<div class=miss>漏: {_html.escape(chr(44).join(m))}</div>" if (m := missed) else ""}</td>')
            tds.append(f'<td class="sub">{_html.escape(", ".join(row["relevant"]))}</td>')
            trs.append("<tr>" + "".join(tds) + "</tr>")
        sections.append((f"逐题检索命中（共 {n_q} 题）",
                         '<div class="chartcard"><table class="vtbl"><tr>' + th + "</tr>"
                         + "".join(trs) + "</table><div class='legend'>单元格 = recall@5 / 命中金标数；"
                         "绿=该题金标全命中，红=有漏检。</div></div>"))

    # 索引侧信息卡
    armcards = "".join(
        f'<div class="defcard" style="border-left:4px solid {colors[l]}">'
        f'<div class="defkey"><span class="dot" style="background:{colors[l]}"></span>{_html.escape(l)}</div>'
        f'<div class="defbody">{_html.escape(json.dumps(summary["arms"][i]["meta"] or {"题数": summary["arms"][i]["total_questions"]}, ensure_ascii=False))}</div></div>'
        for i, l in enumerate(labels))

    notes = "".join(f"<li>{_html.escape(s.lstrip('- '))}</li>" for s in summary.get("notes", []))
    body = "\n".join(
        f'<h2>{title}</h2>{content}' if not title.startswith("分环节") and not title.startswith("逐题")
        else f'<h2>{title}</h2>{content}'
        for title, content in sections)

    return f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>切分参数对照评测（方式 A）</title>
<style>
  body {{ font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif; background:#f7f8fa; color:#1f2937; margin:0; padding:24px; }}
  h1 {{ color:#111827; }} h2 {{ color:#374151; border-left:4px solid #3b82f6; padding-left:10px; margin-top:28px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:14px; }}
  .card,.chartcard {{ background:#fff; border:1px solid #e5e7eb; border-radius:10px; padding:14px 16px; box-shadow:0 1px 2px rgba(0,0,0,.04); margin:10px 0; }}
  .card h3,.chartcard h3 {{ margin:0 0 10px; font-size:14px; color:#4b5563; }}
  .barrow {{ display:flex; align-items:center; gap:8px; margin:6px 0; font-size:12px; }}
  .blab {{ width:88px; font-weight:600; text-align:right; }}
  .track {{ flex:1; background:#eef2f7; border-radius:6px; height:16px; overflow:hidden; }}
  .fill {{ height:100%; border-radius:6px; }}
  .bval {{ width:64px; text-align:right; font-variant-numeric:tabular-nums; color:#374151; }}
  .defs {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(240px,1fr)); gap:12px; margin:16px 0 6px; }}
  .defcard {{ background:#fff; border:1px solid #e5e7eb; border-radius:10px; padding:12px 14px; }}
  .defkey {{ font-weight:600; font-size:13px; margin-bottom:6px; display:flex; align-items:center; gap:8px; }}
  .dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
  .defbody {{ font-size:12px; color:#4b5563; word-break:break-all; }}
  table.vtbl {{ border-collapse:collapse; width:100%; font-size:12px; }}
  table.vtbl th,table.vtbl td {{ border:1px solid #e5e7eb; padding:5px 7px; text-align:center; vertical-align:top; }}
  table.vtbl th {{ background:#f3f4f6; }} table.vtbl td.q {{ text-align:left; }}
  td.good {{ background:#f0fdf4; }} td.bad {{ background:#fef2f2; }}
  .sub {{ color:#6b7280; font-size:11px; }} .miss {{ color:#b91c1c; font-size:11px; }}
  .legend {{ margin-top:10px; font-size:12px; color:#4b5563; display:flex; flex-wrap:wrap; gap:10px; }}
  .lg {{ display:inline-flex; align-items:center; gap:4px; }} .lg i {{ width:12px; height:12px; border-radius:3px; display:inline-block; }}
  ul {{ font-size:13px; color:#4b5563; line-height:1.7; }}
</style></head><body>
<h1>切分参数对照评测 · 方式 A（循环重排开 + 图扩展开，生产默认链路）</h1>
<p>列间唯一差异为索引参数（chunk_size / chunk_overlap / chunking_strategy）。条形按各指标内相对高低归一化。</p>
<div class="defs">{armcards}</div>
{body}
<h2>读表须知</h2><ul>{notes}</ul>
</body></html>"""


if __name__ == "__main__":
    main()
