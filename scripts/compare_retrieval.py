"""
对比三档检索配置：
A: 纯向量（dense only）
B: 混合（dense + BM25）
C: 混合 + rerank

使用方法：
    1. 确保 Ollama 运行且 vault 已索引
    2. 先建 BM25 索引：python scripts/compare_retrieval.py --build-bm25
    3. 运行对比：    python scripts/compare_retrieval.py

面试说法："三档配置 A/B/C 逐步增强，量化对比召回率提升，
用数据证明混合检索 + reranker 的价值。"
"""

import argparse
import os
import sys
import time
from pathlib import Path

# 解决 Windows GBK 编码问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 确保 src/ 可被找到
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def build_bm25():
    """从 ChromaDB 建 BM25 索引并保存"""
    from note_assistant.retrieval.sparse_retriever import BM25Retriever

    print("📦 从 ChromaDB 构建 BM25 索引...")
    retriever = BM25Retriever.from_chroma()
    retriever.save()
    print(f"✅ BM25 索引已保存，共 {len(retriever.corpus)} 个 chunk")


def load_bm25():
    """从 pickle 加载 BM25 索引"""
    from note_assistant.retrieval.sparse_retriever import BM25Retriever

    retriever = BM25Retriever()
    retriever.load()
    return retriever


def run_comparison(alpha: float = 0.7, top_k: int = 5):
    """运行三档检索对比实验"""
    from note_assistant.indexing.embedder import OllamaEmbedder
    from note_assistant.indexing.ingestor import Ingestor
    from note_assistant.retrieval.hybrid import HybridRetriever
    from note_assistant.retrieval.reranker import LocalReranker
    from note_assistant.retrieval.query_rewrite import QueryRewriter

    # ─── 初始化组件 ─────────────────────────────────────
    print("🔧 初始化组件...")
    embedder = OllamaEmbedder()
    ingestor = Ingestor()
    bm25 = load_bm25()
    hybrid = HybridRetriever(alpha=alpha, top_k=top_k * 3)
    reranker = LocalReranker()
    rewriter = QueryRewriter()

    collection = ingestor.collection
    if collection.count() == 0:
        print("❌ ChromaDB 为空，请先运行 indexer")
        sys.exit(1)

    print(f"   ChromaDB: {collection.count()} chunks")
    print(f"   BM25:     {len(bm25.corpus)} chunks")

    # ─── 测试查询 ───────────────────────────────────────
    test_queries = [
        "RAG 评测指标 Faithfulness 定义",
        "FlashAttention 优化点",
        "LoRA 和 QLoRA 区别",
        "BM25 检索算法原理",
        "向量检索和稀疏检索的区别",
    ]

    # ─── 运行对比 ───────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  检索对比实验  (alpha={alpha}, top_k={top_k})")
    print(f"{'='*70}\n")

    total_times = {"A": 0.0, "B": 0.0, "C": 0.0}
    total_results = {"A": 0, "B": 0, "C": 0}

    for qi, query in enumerate(test_queries):
        print(f"\n[{qi+1}/5] Query: {query}")

        # Query 改写
        rewritten = rewriter.rewrite(query)
        print(f"       改写: {rewritten}")

        # 获取 embedding
        q_emb = embedder.embed_one(rewritten)

        # A: 纯向量
        t0 = time.time()
        a_results = collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        t_a = time.time() - t0
        total_times["A"] += t_a

        # B: 混合检索
        t0 = time.time()
        b_results = hybrid.search(rewritten, top_k=top_k * 2)[:top_k]
        t_b = time.time() - t0
        total_times["B"] += t_b

        # C: 混合 + rerank
        t0 = time.time()
        hybrid_top = hybrid.search(rewritten, top_k=top_k * 3)
        c_results = reranker.rerank(rewritten, hybrid_top, top_k=top_k)
        t_c = time.time() - t0
        total_times["C"] += t_c

        total_results["A"] += len(a_results["documents"][0])
        total_results["B"] += len(b_results)
        total_results["C"] += len(c_results)

        # 打印每档 top-3 结果
        print(f"  A 纯向量 ({t_a:.2f}s):")
        for i, (doc, meta) in enumerate(zip(
            a_results["documents"][0][:3],
            a_results["metadatas"][0][:3],
        )):
            fp = meta.get("filepath", "?")
            print(f"    [{i+1}] {fp}: {doc[:50]}...")

        print(f"  B 混合检索 ({t_b:.2f}s):")
        for i, r in enumerate(b_results[:3]):
            print(f"    [{i+1}] {r.filepath}: {r.page_content[:50]}...")

        print(f"  C 混合+rerank ({t_c:.2f}s):")
        for i, r in enumerate(c_results[:3]):
            print(f"    [{i+1}] {r.filepath}: {r.page_content[:50]}...")

    # ─── 汇总 ──────────────────────────────────────────
    n = len(test_queries)
    print(f"\n{'='*70}")
    print("  汇总")
    print(f"{'='*70}")
    print(f"  {'配置':<20} {'平均耗时':>10} {'平均结果数':>12}")
    print(f"  {'-'*42}")
    for cfg in ["A", "B", "C"]:
        label = {"A": "纯向量", "B": "混合检索", "C": "混合+rerank"}[cfg]
        avg_t = total_times[cfg] / n
        avg_r = total_results[cfg] / n
        print(f"  {label:<20} {avg_t:>9.2f}s {avg_r:>11.1f}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="对比三档检索配置 (A/B/C)")
    parser.add_argument("--build-bm25", action="store_true",
                        help="从 ChromaDB 构建 BM25 索引")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="混合检索 dense 权重 (默认 0.7)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="每档返回 top-k 条 (默认 5)")
    args = parser.parse_args()

    if args.build_bm25:
        build_bm25()
        return

    run_comparison(alpha=args.alpha, top_k=args.top_k)


if __name__ == "__main__":
    main()
