"""
端到端测试脚本：验证完整 RAG 管线（混合检索 → Rerank → 图扩展 → 生成）。

用法:
    # 先建索引（首次）
    uv run python -m note_assistant.indexing.ingestor

    # 再跑端到端测试
    uv run python scripts/demo_e2e.py "你的问题"

    # 指定 vault 路径
    uv run python scripts/demo_e2e.py "你的问题" --vault /path/to/vault
"""
import asyncio
import sys
import time
import argparse
from pathlib import Path

# Windows GBK 编码兼容
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 确保 src 在 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


async def main():
    parser = argparse.ArgumentParser(description="RAG 端到端测试")
    parser.add_argument("question", help="要问的问题")
    parser.add_argument("--vault", default=None, help="vault 路径（默认读 config）")
    parser.add_argument("--top-k", type=int, default=5, help="最终返回几个结果")
    parser.add_argument("--no-graph", action="store_true", help="跳过图扩展")
    parser.add_argument("--no-rerank", action="store_true", help="跳过 rerank")
    parser.add_argument("--no-stream", action="store_true", help="非流式输出")
    args = parser.parse_args()

    print("=" * 60)
    print("  RAG 端到端测试")
    print("=" * 60)
    print(f"\n📝 问题：{args.question}")
    print(f"   top_k={args.top_k}  graph={not args.no_graph}  rerank={not args.no_rerank}")

    # ─── 1. 初始化组件 ─────────────────────────────
    print("\n[1/5] 初始化组件...")
    t0 = time.time()

    from note_assistant.indexing.vault_loader import VaultLoader
    from note_assistant.retrieval.hybrid import HybridRetriever
    from note_assistant.retrieval.reranker import LocalReranker
    from note_assistant.retrieval.graph import WikiGraph
    from note_assistant.generation.generator import Generator
    from note_assistant.pipeline.rag_chain import RAGChain

    # 检索器
    retriever = HybridRetriever()

    # Reranker（可选）
    reranker = None
    if not args.no_rerank:
        try:
            reranker = LocalReranker()
            print("   ✅ Reranker 已加载")
        except Exception as e:
            print(f"   ⚠️  Reranker 加载失败: {e}")
            print("   → 跳过 rerank")

    # 图（可选）
    graph = None
    if not args.no_graph:
        graph = WikiGraph()
        try:
            graph.load()
            print(f"   ✅ WikiGraph 已加载: {graph.node_count} 节点, {graph.edge_count} 边")
        except FileNotFoundError:
            print("   ⚠️  未找到 graph.gpickle，尝试从 vault 建图...")
            vault_path = args.vault or None
            loader = VaultLoader(vault_path)
            docs = loader.load_all()
            graph.build_from_docs(docs)
            graph.save()
            print(f"   ✅ WikiGraph 已重建: {graph.node_count} 节点, {graph.edge_count} 边")
        except Exception as e:
            print(f"   ⚠️  图加载失败: {e}")
            graph = None

    # Generator
    generator = Generator()
    print("   ✅ Generator 已加载")

    # 组装管线
    chain = RAGChain(
        hybrid_retriever=retriever,
        reranker=reranker or LocalReranker.__new__(LocalReranker),  # dummy if None
        graph=graph,
        generator=generator,
    )
    # 如果没 reranker，给个 passthrough
    if reranker is None:
        class _Passthrough:
            def rerank(self, q, results, top_k=None):
                return results[:top_k] if top_k else results
        chain.reranker = _Passthrough()

    t1 = time.time()
    print(f"   ⏱  初始化耗时: {t1 - t0:.2f}s")

    # ─── 2. 混合检索 ─────────────────────────────
    print("\n[2/5] 混合检索...")
    t0 = time.time()

    from note_assistant.indexing.embedder import OllamaEmbedder
    embedder = OllamaEmbedder()
    query_embedding = embedder.embed_one(args.question)
    hybrid_results = retriever.search(args.question, top_k=20)

    t1 = time.time()
    print(f"   ✅ 检索到 {len(hybrid_results)} 个候选")
    print(f"   ⏱  耗时: {t1 - t0:.2f}s")

    # 打印 top-3 候选
    for i, r in enumerate(hybrid_results[:3]):
        fp = r.metadata.get("filepath", "?")
        print(f"   [{i+1}] {fp}  score={r.score:.4f}")

    # ─── 3. Rerank ─────────────────────────────
    if reranker:
        print("\n[3/5] Rerank 精排...")
        t0 = time.time()
        reranked = reranker.rerank(args.question, hybrid_results, top_k=args.top_k)
        t1 = time.time()
        print(f"   ✅ 精排后 {len(reranked)} 个结果")
        print(f"   ⏱  耗时: {t1 - t0:.2f}s")
        for i, r in enumerate(reranked[:3]):
            fp = r.metadata.get("filepath", "?")
            print(f"   [{i+1}] {fp}  score={r.score:.4f}")
    else:
        print("\n[3/5] Rerank 跳过")
        reranked = hybrid_results[:args.top_k]

    # ─── 4. 图扩展 ─────────────────────────────
    if graph:
        print("\n[4/5] 图扩展...")
        t0 = time.time()
        hit_files = set()
        for r in reranked:
            fp = r.metadata.get("filepath", "")
            if fp:
                hit_files.add(fp)

        if hit_files:
            neighbors = graph.expand(hit_files, hop=1, max_neighbors=5)
            print(f"   ✅ 命中 {len(hit_files)} 个文件 → 扩展出 {len(neighbors)} 个邻居")
            for fp, decay in neighbors[:3]:
                print(f"      {fp}  decay={decay}")
        else:
            print("   ⚠️  无命中文件，跳过扩展")
    else:
        print("\n[4/5] 图扩展跳过")

    # ─── 5. 生成回答 ─────────────────────────────
    print("\n[5/5] 生成回答...")
    t0 = time.time()

    # 组装 context：rerank 结果（RetrievalResult）+ 图扩展 chunks（RetrievalResult）
    all_chunks = list(reranked)  # RetrievalResult 列表
    if graph and hit_files:
        expand_chunks = chain._fetch_neighbor_chunks(
            graph.expand(hit_files, hop=1, max_neighbors=5)
        )
        all_chunks.extend(expand_chunks)

    if args.no_stream:
        answer = generator.generate(args.question, all_chunks)
        t1 = time.time()
        print(f"   ⏱  生成耗时: {t1 - t0:.2f}s")
        print(f"\n{'=' * 60}")
        print("  回答")
        print(f"{'=' * 60}\n")
        print(answer)
    else:
        print("  回答: ", end="", flush=True)
        t0 = time.time()
        tokens = 0
        async for chunk in generator.generate_stream(args.question, all_chunks):
            print(chunk, end="", flush=True)
            tokens += len(chunk)
        t1 = time.time()
        print(f"\n   ⏱  流式生成耗时: {t1 - t0:.2f}s ({tokens} chars)")

    # ─── 6. 来源汇总 ─────────────────────────────
    print(f"\n{'=' * 60}")
    print("  📚 来源")
    print(f"{'=' * 60}")
    for i, r in enumerate(reranked):
        fp = r.metadata.get("filepath", "?")
        title = r.metadata.get("title", "?")
        score = r.score
        preview = r.page_content[:100].replace("\n", " ")
        print(f"\n  [{i+1}] {title}")
        print(f"       📁 {fp}  score={score:.4f}")
        print(f"       {preview}...")

    print(f"\n{'=' * 60}")
    print("  ✅ 端到端测试完成")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
