"""
增量索引脚本：只处理新增/变更/删除的文件，秒级完成。

用法:
    python scripts/reindex.py [vault_path]

    vault_path 默认从 config.vault_path 读取
"""

import sys
from pathlib import Path

# 确保 src 在 path 里
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def incremental_reindex(vault_path: str | None = None):
    """
    【核心逻辑待实现】增量索引主流程。

    步骤：
    1. 初始化 SyncDB + VaultLoader + Ingestor
    2. 扫描 vault，找出所有 .md 文件
    3. 对比 sync.db：
       a. 需要重新索引的文件（新增 or mtime/sha256 变了）
       b. 需要删除的文件（在 sync.db 里但不在 vault 里了）
    4. 删除已删除文件的 chunks
    5. 逐篇重建变更文件（先删旧的，再重新处理 + 写入）
    6. 更新 sync.db
    7. 打印统计

    Args:
        vault_path: vault 路径（可选，默认从 config 读）
    """
    from note_assistant.config import settings
    from note_assistant.indexing.vault_loader import VaultLoader
    from note_assistant.indexing.splitter import make_splitters, split_v2
    from note_assistant.indexing.preprocessor import RichPreprocessor
    from note_assistant.indexing.sync import SyncDB
    from note_assistant.indexing.ingestor import Ingestor

    vault = vault_path or str(settings.vault_path.resolve())

    sync = SyncDB()
    loader = VaultLoader(vault)
    ingestor = Ingestor()
    head_spliter, cs = make_splitters()

    all_docs = loader.load_all()
    to_reindex = []
    to_remove = []

    # 找出需要重新索引的文件
    for doc in all_docs:
        abs_path = doc.abs_path if hasattr(doc, 'abs_path') else None
        if sync.need_reindex(doc.filepath, abs_path):
            to_reindex.append(doc)

    # 找出已删除的文件（在 sync.db 里但不在 vault 里）
    vault_files = {d.filepath for d in all_docs}
    for indexed_file in sync.get_all_indexed():
        if indexed_file not in vault_files:
            to_remove.append(indexed_file)

    print(f"📊 全库 {len(all_docs)} 篇")
    print(f"  → 需更新: {len(to_reindex)} 篇")
    print(f"  → 需删除: {len(to_remove)} 篇")

    # 删除已删除文件的 chunks
    if to_remove:
        for filepath in to_remove:
            ingestor.collection.delete(where={"filepath": filepath})
            sync.remove_state(filepath)
        print(f"  ✅ 已删除 {len(to_remove)} 篇")

    # 增量索引
    preprocessor = RichPreprocessor()
    for i, doc in enumerate(to_reindex):
        # 先删旧的（如果存在）
        try:
            ingestor.collection.delete(where={"filepath": doc.filepath})
        except Exception:
            pass

        # 重新处理
        from note_assistant.indexing.embedder import OllamaEmbedder
        embedder = OllamaEmbedder()

        cleaned, fm_chunks = preprocessor.process_with_meta(doc)
        chunks = split_v2(doc, head_spliter, cs)
        chunks = preprocessor.restore(chunks)
        summary_chunks = preprocessor.generate_summaries()

        # 补 metadata
        for c in chunks:
            c.metadata["filepath"] = doc.filepath
            c.metadata["title"] = doc.title

        all_chunks = chunks + summary_chunks + fm_chunks
        if all_chunks:
            texts = [c.page_content for c in all_chunks]
            embeddings = embedder.embed(texts)
            ingestor.upsert(all_chunks)

        # 更新 sync.db
        abs_path = doc.abs_path if hasattr(doc, 'abs_path') else None
        sync.update_state(doc.filepath, abs_path)
        print(f"  ✅ [{i+1}/{len(to_reindex)}] {doc.filepath}")

    print(f"\n🎉 增量索引完成")


if __name__ == "__main__":
    vault = sys.argv[1] if len(sys.argv) > 1 else None
    incremental_reindex(vault)