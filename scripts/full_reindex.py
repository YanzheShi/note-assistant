"""
全量重建索引脚本：清空 ChromaDB → 重新加载 → 重新切分 → 重新向量化 → 重新入库。

用法:
    # 默认从 config.vault_path 读取
    uv run python scripts/full_reindex.py

    # 指定 vault 路径
    uv run python scripts/full_reindex.py --vault /path/to/vault

    # 跳过确认（自动化用）
    uv run python scripts/full_reindex.py --yes

    # 只加载不写入（dry-run，验证流程）
    uv run python scripts/full_reindex.py --dry-run

前置条件:
    - Ollama 服务已启动（bge-m3:latest 模型已 pull）
    - .env 配置了正确的 vault_path
"""

import sys
import time
import argparse
from pathlib import Path

# Windows GBK 编码兼容
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 确保 src 在 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="全量重建索引")
    parser.add_argument("--vault", default=None, help="vault 路径（默认读 config）")
    parser.add_argument("--yes", "-y", action="store_true", help="跳过确认")
    parser.add_argument("--dry-run", action="store_true", help="只加载不写入")
    args = parser.parse_args()

    print("=" * 60)
    print("  全量重建索引")
    print("=" * 60)

    # ─── 0. 初始化 ─────────────────────────────
    from note_assistant.config import settings
    from note_assistant.indexing.vault_loader import VaultLoader
    from note_assistant.indexing.preprocessor import RichPreprocessor
    from note_assistant.indexing.splitter import make_splitters, split_v2
    from note_assistant.indexing.ingestor import Ingestor

    vault_path = args.vault or str(settings.vault_path.resolve())
    print(f"\n  Vault: {vault_path}")
    print(f"  ChromaDB: {settings.chroma_persist_dir.resolve()}")
    print(f"  Embed Model: {settings.embed_model}")
    print(f"  Chunk Size: {settings.chunk_size} / Overlap: {settings.chunk_overlap}")

    if args.dry_run:
        print("\n  [DRY RUN 模式] 只加载不写入")

    # 确认
    if not args.yes and not args.dry_run:
        print("\n  ⚠️  此操作会清空 ChromaDB 并重建所有索引！")
        ans = input("  确认继续? [y/N]: ").strip().lower()
        if ans != "y":
            print("  已取消")
            return

    total_start = time.time()

    # ─── 1. 清空 ChromaDB ──────────────────────
    print("\n[1/6] 清空 ChromaDB...")
    t0 = time.time()
    ingestor = Ingestor()
    if not args.dry_run:
        ingestor.delete_all()
    t1 = time.time()
    print(f"  ✅ 已清空  ({t1 - t0:.2f}s)")

    # ─── 2. 加载笔记 ──────────────────────────
    print("\n[2/6] 加载笔记...")
    t0 = time.time()
    loader = VaultLoader(vault_path)
    docs = loader.load_all()
    t1 = time.time()
    print(f"  ✅ 加载 {len(docs)} 篇笔记  ({t1 - t0:.2f}s)")

    if not docs:
        print("  ⚠️  没有笔记，退出")
        return

    # 打印前几篇
    for d in docs[:3]:
        print(f"     📄 {d.filepath}  ({len(d.raw_md)} chars, {len(d.wikilinks)} links)")
    if len(docs) > 3:
        print(f"     ... 还有 {len(docs) - 3} 篇")

    # ─── 3. 初始化切分器 ──────────────────────
    print("\n[3/6] 初始化切分器...")
    hs, cs = make_splitters()
    preprocessor = RichPreprocessor()
    print(f"  ✅ HeaderSplitter + RecursiveCharacterTextSplitter (chunk_size={settings.chunk_size})")

    # ─── 4. 逐篇处理 ──────────────────────────
    print("\n[4/6] 逐篇处理（预处理 → 切分 → 还原 → 生成摘要）...")
    t0 = time.time()
    all_chunks = []
    stats = {"files": 0, "chunks": 0, "code_blocks": 0, "tables": 0, "mermaid": 0, "images": 0}

    for i, node in enumerate(docs):
        # 4a. 预处理（保护富结构）
        preprocessor.__init__()  # 重置 extracted
        cleaned, fm_chunks = preprocessor.process_with_meta(node)

        # 4b. 切分
        chunks = split_v2(node, hs, cs)

        # 4c. 还原占位符
        chunks = preprocessor.restore(chunks)

        # 4d. 补 metadata
        for c in chunks:
            c.metadata["filepath"] = node.filepath
            c.metadata["title"] = node.title
            if node.tags:
                c.metadata["tags"] = node.tags
            c.metadata["wikilinks"] = node.wikilinks

        # 4e. 富结构摘要 chunks
        summary_chunks = preprocessor.generate_summaries()
        for sc in summary_chunks:
            sc.metadata["filepath"] = node.filepath
            sc.metadata["title"] = node.title

        # 4f. 合并
        file_chunks = chunks + summary_chunks + fm_chunks
        all_chunks.extend(file_chunks)

        # 统计
        stats["files"] += 1
        stats["chunks"] += len(file_chunks)
        stats["code_blocks"] += sum(1 for e in preprocessor.extracted if e.kind == "code")
        stats["tables"] += sum(1 for e in preprocessor.extracted if e.kind == "table")
        stats["mermaid"] += sum(1 for e in preprocessor.extracted if e.kind == "mermaid")
        stats["images"] += sum(1 for e in preprocessor.extracted if e.kind == "image")

        # 进度
        if (i + 1) % 10 == 0 or i == len(docs) - 1:
            print(f"     [{i+1}/{len(docs)}] {node.filepath} → {len(file_chunks)} chunks")

    t1 = time.time()
    print(f"  ✅ 处理完成  ({t1 - t0:.2f}s)")
    print(f"     文件: {stats['files']}")
    print(f"     Chunks: {stats['chunks']}")
    print(f"     富结构: code={stats['code_blocks']}  table={stats['tables']}  mermaid={stats['mermaid']}  image={stats['images']}")

    if args.dry_run:
        print("\n  [DRY RUN] 跳过写入和向量化")
        print(f"\n  总计: {stats['chunks']} chunks 待写入")
        return

    # ─── 5. 向量化 ────────────────────────────
    print("\n[5/6] 向量化（Embedding）...")
    t0 = time.time()
    embeddings = ingestor.embedder.embed([c.page_content for c in all_chunks])
    t1 = time.time()
    print(f"  ✅ 向量化完成: {len(embeddings)} 个向量  ({t1 - t0:.2f}s)")
    print(f"     模型: {settings.embed_model}  维度: {len(embeddings[0]) if embeddings else 0}")

    # ─── 6. 写入 ChromaDB ─────────────────────
    print("\n[6/6] 写入 ChromaDB...")
    t0 = time.time()

    # 分批写入（避免一次性太大）
    batch_size = 100
    total_written = 0
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]

        # 构造 IDs
        ids = []
        for j, c in enumerate(batch_chunks):
            fp = c.metadata.get("filepath", "unknown")
            ids.append(ingestor._make_id(fp, i + j, c.kind))

        # 清理 metadata（ChromaDB 不接受空列表）
        metadatas = []
        for c in batch_chunks:
            clean = {k: v for k, v in c.metadata.items()
                     if not (isinstance(v, list) and len(v) == 0)}
            metadatas.append(clean)

        ingestor.collection.upsert(
            ids=ids,
            documents=[c.page_content for c in batch_chunks],
            embeddings=batch_embeddings,
            metadatas=metadatas,
        )
        total_written += len(batch_chunks)
        print(f"     写入 {total_written}/{len(all_chunks)}")

    t1 = time.time()
    print(f"  ✅ 写入完成  ({t1 - t0:.2f}s)")

    # ─── 汇总 ─────────────────────────────────
    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  全量重建完成")
    print(f"{'=' * 60}")
    print(f"  文件数:   {stats['files']}")
    print(f"  Chunks:   {stats['chunks']}")
    print(f"  写入:     {total_written}")
    print(f"  总耗时:   {total_time:.2f}s")
    print(f"  ChromaDB: {settings.chroma_persist_dir.resolve()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
