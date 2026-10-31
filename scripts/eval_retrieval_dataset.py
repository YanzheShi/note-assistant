#!/usr/bin/env python3
"""纯检索评测：数据集 → HybridRetriever → MRR / Recall@k / nDCG@k。

不调用 LLM、不依赖 ragas，适合快速验证「索引换了/检索改动了」之后
文件级召回有没有退化。数据集格式与 EvalDataset 一致：

    {"name": "...", "questions": [{"question": "...", "relevant_files": [...]}, ...]}

用法：
    uv run python scripts/eval_retrieval_dataset.py --dataset eval/html_corpus_22.json
    uv run python scripts/eval_retrieval_dataset.py --dataset ... --rerank
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _norm_files(files: list[str], retrieved: list[str]) -> tuple[list[str], set[str]]:
    """短文件名数据集 → 只比 basename；全路径数据集 → 原样比较。"""
    if not files:
        return retrieved, set()
    sample = files[0]
    if not any(sep in sample for sep in ("/", "\\")):
        return [Path(f).name for f in retrieved], {Path(f).name for f in files}
    return retrieved, set(files)


def compute_metrics(retrieved: list[str], relevant: list[str], k_values: list[int]) -> dict:
    retrieved, relevant_set = _norm_files(relevant, retrieved)
    if not relevant_set:
        return {}

    first = next((i + 1 for i, f in enumerate(retrieved) if f in relevant_set), None)
    metrics: dict[str, float] = {"mrr": 1.0 / first if first else 0.0}
    for k in k_values:
        top = retrieved[:k]
        hits = sum(1 for f in top if f in relevant_set)
        metrics[f"recall@{k}"] = hits / len(relevant_set)
        metrics[f"precision@{k}"] = hits / max(1, len(top))
        dcg = sum(1 / math.log2(i + 2) for i, f in enumerate(top) if f in relevant_set)
        ideal = min(k, len(relevant_set))
        idcg = sum(1 / math.log2(i + 1) for i in range(1, ideal + 1))
        metrics[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="纯检索评测（无 LLM）")
    ap.add_argument("--dataset", "-d", required=True)
    ap.add_argument("--output", "-o", default="eval/report/html_corpus_retrieval.json")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    ap.add_argument("--rerank", action="store_true", help="混合检索后再过本地 reranker")
    args = ap.parse_args()

    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    questions = data["questions"]

    from note_assistant.config import settings
    from note_assistant.retrieval.hybrid import HybridRetriever

    print("=" * 72)
    print(f"  数据集: {data.get('name')} ({len(questions)} 题)")
    print(f"  Chroma: {settings.chroma_persist_dir} | collection={settings.collection_name}")
    print(f"  BM25:   {settings.bm25_index_path}")
    print(f"  top_k={args.top_k} rerank={args.rerank}")
    print("=" * 72)

    retriever = HybridRetriever(top_k=args.top_k)
    reranker = None
    if args.rerank:
        from note_assistant.retrieval.reranker import LocalReranker

        reranker = LocalReranker()

    rows = []
    totals: dict[str, list[float]] = {}
    t_start = time.time()
    for i, q in enumerate(questions, 1):
        question = q["question"]
        relevant = q.get("relevant_files") or []
        t0 = time.time()
        results = retriever.search(question, top_k=args.top_k)
        if reranker is not None:
            results = reranker.rerank(question, results, top_k=args.top_k)

        ranked_files: list[str] = []
        seen = set()
        for r in results:
            fp = (r.metadata or {}).get("filepath", "")
            if fp and fp not in seen:
                seen.add(fp)
                ranked_files.append(fp)

        m = compute_metrics(ranked_files, relevant, args.k)
        for key, value in m.items():
            totals.setdefault(key, []).append(value)

        row = {
            "question": question,
            "relevant_files": relevant,
            "retrieved_files": ranked_files[:args.top_k],
            "top_chunks": [
                {
                    "filepath": (r.metadata or {}).get("filepath", ""),
                    "heading_path": (r.metadata or {}).get("heading_path", ""),
                    "score": round(float(r.score), 4),
                }
                for r in results[:5]
            ],
            "metrics": {key: round(value, 4) for key, value in m.items()},
            "elapsed_ms": round((time.time() - t0) * 1000),
        }
        rows.append(row)
        hit = "✓" if m.get("mrr", 0) > 0 else "✗"
        print(
            f"[{i:02d}/{len(questions)}] {hit} MRR={m.get('mrr', 0):.3f} "
            f"R@5={m.get('recall@5', 0):.2f} | {question[:46]}"
        )

    summary = {
        key: round(sum(values) / len(values), 4) for key, values in totals.items()
    }
    report = {
        "dataset": data.get("name"),
        "questions": len(questions),
        "config": {
            "top_k": args.top_k,
            "rerank": args.rerank,
            "chroma_persist_dir": str(settings.chroma_persist_dir),
            "collection_name": settings.collection_name,
            "bm25_index_path": str(settings.bm25_index_path),
        },
        "summary": summary,
        "elapsed_ms": round((time.time() - t_start) * 1000),
        "per_question": rows,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    for key in ("mrr", *[f"recall@{k}" for k in args.k], *[f"ndcg@{k}" for k in args.k]):
        if key in summary:
            print(f"  {key:12} {summary[key]:.4f}")
    print(f"  报告: {out}")
    print("=" * 72)


if __name__ == "__main__":
    main()
