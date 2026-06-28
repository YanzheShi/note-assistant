"""
Indexing 统一测试 —— 跑完整流程，打印每一步执行结果
测试一篇文档：加载 → 预处理 → 切分 → restore → summary → 入库 → 查询验证
"""

from pathlib import Path
from note_assistant.indexing.vault_loader import VaultLoader
from note_assistant.indexing.preprocessor import RichPreprocessor
from note_assistant.indexing.splitter import make_splitters, split_v2
from note_assistant.indexing.embedder import OllamaEmbedder
from note_assistant.indexing.ingestor import Ingestor


def main():
    # ─── Step 0: 选一篇测试文档 ─────────────────────────────────
    print("=" * 70)
    print("Step 0: 加载 vault，选一篇测试文档")
    print("=" * 70)

    loader = VaultLoader()
    docs = loader.load_all()
    if not docs:
        print("❌ vault 里没有 .md 文件")
        return

    # 选第一篇有足够内容的（这里简单取第一个）
    node = docs[0]
    print(f"  选中: {node.filepath}")
    print(f"  标题: {node.title}")
    print(f"  标签: {node.tags}")
    print(f"  双链: {node.wikilinks[:5]}")
    print(f"  标题树: {[(h['level'], h['text']) for h in node.headings[:5]]}")
    print(f"  原始长度: {len(node.raw_md)} chars")
    print()

    # ─── Step 1: 预处理 ─────────────────────────────────────────
    print("=" * 70)
    print("Step 1: 预处理 —— 抽取 code/table/mermaid/image")
    print("=" * 70)

    preprocessor = RichPreprocessor()
    cleaned, fm_chunks = preprocessor.process_with_meta(node.raw_md, node.front_matter)

    print(f"  清洗后长度: {len(cleaned)} chars")
    print(f"  抽取的富结构:")
    for ext in preprocessor.extracted:
        print(f"    [{ext.kind}] {ext.placeholder} → {ext.raw[:60]!r}...")
    print(f"  front_matter chunks: {len(fm_chunks)}")
    for fc in fm_chunks:
        print(f"    {fc['metadata']['kind']}: {fc['page_content'][:80]}")
    print()

    # ─── Step 2: 切分（v2 Header + Recursive）───────────────────
    print("=" * 70)
    print("Step 2: 切分（Header + Recursive）")
    print("=" * 70)

    hs, cs = make_splitters()
    chunks = split_v2(node, hs, cs)

    print(f"  切出 {len(chunks)} 个 chunk")
    for i, c in enumerate(chunks[:5]):  # 只打印前5个
        hp = c["metadata"].get("heading_path", "N/A")
        print(f"  [{i}] hp={hp}")
        print(f"      content: {c['page_content'][:80]!r}...")
    if len(chunks) > 5:
        print(f"  ... 还有 {len(chunks) - 5} 个 chunk")
    print()

    # ─── Step 3: Restore 还原占位符 ─────────────────────────────
    print("=" * 70)
    print("Step 3: Restore —— 还原占位符为原始内容")
    print("=" * 70)

    chunks = preprocessor.restore(chunks)

    for i, c in enumerate(chunks[:3]):
        has_code = c["metadata"].get("has_code", False)
        has_table = c["metadata"].get("has_table", False)
        has_mermaid = c["metadata"].get("has_mermaid", False)
        has_image = c["metadata"].get("has_image", False)
        marks = []
        if has_code: marks.append("code")
        if has_table: marks.append("table")
        if has_mermaid: marks.append("mermaid")
        if has_image: marks.append("image")
        print(f"  [{i}] 含: {marks or '纯文本'}")
        print(f"      content: {c['page_content'][:100]!r}...")
    print()

    # ─── Step 4: 生成辅路 summary chunks ────────────────────────
    print("=" * 70)
    print("Step 4: 生成辅路 summary chunks")
    print("=" * 70)

    summary_chunks = preprocessor.generate_summaries()

    print(f"  生成 {len(summary_chunks)} 个 summary chunk")
    for sc in summary_chunks:
        print(f"    [{sc['metadata']['kind']}] {sc['page_content'][:80]}")
    print()

    # ─── Step 5: 入库 ChromaDB ──────────────────────────────────
    print("=" * 70)
    print("Step 5: 入库 ChromaDB")
    print("=" * 70)

    # 补 metadata
    for c in chunks:
        c["metadata"]["wikilinks"] = node.wikilinks
        c["metadata"]["filepath"] = node.filepath
        c["metadata"]["title"] = node.title
        c["metadata"]["tags"] = node.tags
        c["metadata"]["kind"] = "text"

    for sc in summary_chunks:
        sc["metadata"]["filepath"] = node.filepath
        sc["metadata"]["title"] = node.title

    ingestor = Ingestor()
    all_chunks = chunks + summary_chunks
    n = ingestor.upsert(all_chunks)

    print(f"  入库 {n} 个 chunk（{len(chunks)} 文本 + {len(summary_chunks)} 摘要）")
    print(f"  collection 总数: {ingestor.collection.count()}")
    print()

    # ─── Step 6: 查询验证 ───────────────────────────────────────
    print("=" * 70)
    print("Step 6: 查询验证 —— 用测试文档的第一个 chunk 内容查相似")
    print("=" * 70)

    query_text = chunks[0]["page_content"][:100] if chunks else node.title
    print(f"  查询词: {query_text[:60]}...")

    # 必须用 query_embeddings 传自己算好的向量，不能用 query_texts
    # 否则 ChromaDB 会用内置默认模型（384维）嵌入，跟我们的 1024 维不匹配
    embedder = OllamaEmbedder()
    query_emb = embedder.embed_one(query_text)
    results = ingestor.collection.query(
        query_embeddings=[query_emb],
        n_results=5,
        include=["documents", "metadatas", "distances"],
    )

    print(f"  返回 {len(results['ids'][0])} 条结果:")
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        fp = meta.get("filepath", "?")
        kind = meta.get("kind", "?")
        print(f"  [{i}] dist={dist:.4f} | {fp} ({kind})")
        print(f"      {doc[:80]!r}...")
    print()

    print("=" * 70)
    print("✅ 测试完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
