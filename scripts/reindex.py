"""
增量索引脚本：只处理新增/变更/删除的文件，秒级完成。

与全量 index_vault 共用同一套「最新流程」：
- 注入 make_image_enricher（图片 VLM 理解 / SVG 原生解析）
- 遵守 chunking_strategy（v1 / v2 / v2b 分流）
- 补 dir 元数据 + build_structural_prefix 结构前缀
- v2b 父块写入 ParentDocstore
- 统一经 Ingestor.upsert 入库（内部负责 embedding）

用法:
    python scripts/reindex.py [vault_path]

    vault_path 默认从 config.vault_path 读取
"""

import sys
from pathlib import Path
from dataclasses import replace

# 确保 src 在 path 里
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def incremental_reindex(vault_path: str | None = None):
    from note_assistant.config import settings
    from note_assistant.indexing.vault_loader import VaultLoader
    from note_assistant.indexing.splitter import make_splitters, split_v1, split_v2, split_v2b
    from note_assistant.indexing.preprocessor import RichPreprocessor
    from note_assistant.indexing.understanding import make_image_enricher
    from note_assistant.indexing.ingestor import build_structural_prefix
    from note_assistant.retrieval.docstore import ParentDocstore
    from note_assistant.indexing.sync import SyncDB
    from note_assistant.indexing.ingestor import Ingestor

    vault = vault_path or str(settings.vault_path.resolve())

    sync = SyncDB()
    loader = VaultLoader(vault)
    ingestor = Ingestor()
    hs, cs = make_splitters()
    # 注入 enricher（与 index_vault 一致）
    preprocessor = RichPreprocessor(
        image_enricher=make_image_enricher(vault_path or settings.vault_path)
    )
    is_v2b = settings.chunking_strategy == "v2b"
    # v2b：父块存 docstore（与 index_vault 一致）
    docstore = ParentDocstore(settings.parent_docstore_path) if is_v2b else None

    all_docs = loader.load_all()
    to_reindex = []
    to_remove = []

    # 找出需要重新索引的文件
    for doc in all_docs:
        abs_path = doc.abs_path if hasattr(doc, "abs_path") else None
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
    print(f"  chunking_strategy={settings.chunking_strategy}  image_understand_enabled={settings.image_understand_enabled}")

    # 删除已删除文件的 chunks
    if to_remove:
        for filepath in to_remove:
            ingestor.collection.delete(where={"filepath": filepath})
            sync.remove_state(filepath)
        print(f"  ✅ 已删除 {len(to_remove)} 篇")

    # 增量索引（逐篇，与 index_vault 的 per-doc 流程对齐）
    for i, doc in enumerate(to_reindex):
        # 先删旧的（如果存在）
        try:
            ingestor.collection.delete(where={"filepath": doc.filepath})
        except Exception:
            pass

        abs_path = doc.abs_path if hasattr(doc, "abs_path") else None
        dir_ = str(Path(doc.filepath).parent)
        if dir_ == ".":
            dir_ = ""

        # 1. 预处理（保护富结构 + 图片 enricher）
        cleaned, fm_chunks = preprocessor.process_with_meta(doc)

        # 2. 切分（按 chunking_strategy 分流；切分吃 cleaned 副本）
        node_for_split = replace(doc, raw_md=cleaned)
        if settings.chunking_strategy == "v1":
            chunks = split_v1(node_for_split, cs)
            parents = []
        elif settings.chunking_strategy == "v2b":
            split_res = split_v2b(node_for_split, hs, cs)
            chunks = split_res["children"]
            parents = split_res["parents"]
        else:  # v2
            chunks = split_v2(node_for_split, hs, cs)
            parents = []

        # 3. 还原占位符
        chunks = preprocessor.restore(chunks)
        parents = preprocessor.restore(parents)

        # 4. 补 metadata + 结构前缀（机制 A：语义层感知层级）
        for c in chunks:
            c.metadata["wikilinks"] = doc.wikilinks
            c.metadata["filepath"] = doc.filepath
            c.metadata["title"] = doc.title
            if dir_:
                c.metadata["dir"] = dir_
            if doc.tags:
                c.metadata["tags"] = doc.tags
            prefix = build_structural_prefix(doc, c.metadata, dir_)
            c.page_content = f"{prefix}\n\n{c.page_content}"

        summary_chunks = preprocessor.generate_summaries()
        for sc in summary_chunks:
            sc.metadata["filepath"] = doc.filepath
            sc.metadata["title"] = doc.title
            if dir_:
                sc.metadata["dir"] = dir_
            prefix = build_structural_prefix(doc, sc.metadata, dir_)
            sc.page_content = f"{prefix}\n\n{sc.page_content}"

        for fc in fm_chunks:
            fc.metadata["filepath"] = doc.filepath
            fc.metadata["title"] = doc.title
            if dir_:
                fc.metadata["dir"] = dir_
            prefix = build_structural_prefix(doc, fc.metadata, dir_)
            fc.page_content = f"{prefix}\n\n{fc.page_content}"

        # 5. v2b 父块：写 docstore（不进 ChromaDB）
        if docstore is not None:
            for p in parents:
                p.metadata["filepath"] = doc.filepath
                p.metadata["title"] = doc.title
                if dir_:
                    p.metadata["dir"] = dir_
                p.metadata["wikilinks"] = doc.wikilinks
                if doc.tags:
                    p.metadata["tags"] = doc.tags
                docstore.add(p.metadata["parent_id"], p.page_content, p.metadata)

        # 6. 入库（children + summary + fm 进 ChromaDB；upsert 内部负责 embedding）
        all_chunks = chunks + summary_chunks + fm_chunks
        if all_chunks:
            ingestor.upsert(all_chunks)

        # 更新 sync.db
        sync.update_state(doc.filepath, abs_path)
        print(f"  ✅ [{i + 1}/{len(to_reindex)}] {doc.filepath}")

    # v2b：落盘 docstore（非 v2b 不碰，避免误加载）
    if docstore is not None:
        docstore.save()

    print(f"\n🎉 增量索引完成")


if __name__ == "__main__":
    vault = sys.argv[1] if len(sys.argv) > 1 else None
    incremental_reindex(vault)
