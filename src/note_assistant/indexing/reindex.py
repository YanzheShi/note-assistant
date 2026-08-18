# src/note_assistant/indexing/reindex.py
"""
增量索引执行模块：只处理新增/变更/删除的文件，秒级完成。

与全量 index_vault 共用同一套「最新流程」：
- 注入 make_image_enricher（图片 VLM 理解 / SVG 原生解析）
- 遵守 chunking_strategy（v1 / v2 / v2b 分流）
- 补 dir 元数据 + build_structural_prefix 结构前缀
- v2b 父块写入 ParentDocstore
- 统一经 Ingestor.upsert 入库（内部负责 embedding）

调用方：
- scripts/reindex.py（CLI 薄壳）
- api/main.py::POST /reindex（手动触发）
- indexing/autoindex.py::AutoIndexService（watcher 自动触发）

用法:
    from note_assistant.indexing.reindex import incremental_reindex
    incremental_reindex()                                    # 全库变更比对（现状语义）
    incremental_reindex(filepaths=["a.md"])                  # 只处理指定文件（watcher 单文件路径）
    incremental_reindex(deleted_filepaths=["a.md"])          # 只删除指定文件

已知缺口:
    增量索引不重建 BM25（data/bm25.pkl）与 WikiGraph（data/bm25.graph）这两个
    全量派生产物——长期增量后二者会滞后于 ChromaDB。定期跑一次
    scripts/full_reindex.py 即可全量刷新（该脚本已内置两步重建）。
    自动触发场景由 AutoIndexService 按 autoindex_full_sync_every 计数校准（设计方案 4.5）。
"""

import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

from note_assistant.config import settings
from note_assistant.indexing.splitter import split_v1, split_v2, split_v2b

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 内部辅助：per-doc 索引（全库 / 单文件共用同一份最新流程）
# ──────────────────────────────────────────────


def _make_ingredients(vault_path: str):
    """构建一次增量会话的全部组件（sync/loader/ingestor/splitter/preprocessor/docstore）。"""
    from note_assistant.indexing.vault_loader import VaultLoader
    from note_assistant.indexing.splitter import make_splitters
    from note_assistant.indexing.preprocessor import RichPreprocessor
    from note_assistant.indexing.understanding import make_image_enricher
    from note_assistant.indexing.ingestor import Ingestor
    from note_assistant.indexing.sync import SyncDB
    from note_assistant.retrieval.docstore import ParentDocstore

    sync = SyncDB()
    loader = VaultLoader(vault_path)
    ingestor = Ingestor()
    hs, cs = make_splitters()
    preprocessor = RichPreprocessor(
        image_enricher=make_image_enricher(Path(vault_path) if vault_path else settings.vault_path)
    )
    is_v2b = settings.chunking_strategy == "v2b"
    # v2b：docstore 必须 load 既有持久化实例再增量改（新建空实例 save 会覆盖全部既有父块）
    docstore = None
    if is_v2b:
        if settings.parent_docstore_path.exists():
            try:
                docstore = ParentDocstore.load(settings.parent_docstore_path)
            except Exception as e:  # noqa: BLE001
                logger.warning("docstore 加载失败，按空实例重建: %s", e)
                docstore = ParentDocstore(settings.parent_docstore_path)
        else:
            docstore = ParentDocstore(settings.parent_docstore_path)
    return SimpleNamespace(
        sync=sync, loader=loader, ingestor=ingestor,
        hs=hs, cs=cs, preprocessor=preprocessor, docstore=docstore, is_v2b=is_v2b,
    )


def _index_one(doc, ing) -> None:
    """索引单篇笔记（删旧 → 预处理 → 切分 → 补元数据 → 入库 → 更新 sync.db）。

    ing: _make_ingredients 的返回（含 sync/loader/ingestor/splitters/preprocessor/docstore）。
    """
    # 先删旧的（如果存在）
    try:
        ing.ingestor.collection.delete(where={"filepath": doc.filepath})
    except Exception:  # noqa: BLE001
        pass

    abs_path = doc.abs_path if hasattr(doc, "abs_path") else None
    dir_ = str(Path(doc.filepath).parent)
    if dir_ == ".":
        dir_ = ""

    # 1. 预处理（保护富结构 + 图片 enricher）
    cleaned, fm_chunks = ing.preprocessor.process_with_meta(doc)

    # 2. 切分（按 chunking_strategy 分流；切分吃 cleaned 副本）
    node_for_split = replace(doc, raw_md=cleaned)
    if settings.chunking_strategy == "v1":
        chunks = split_v1(node_for_split, ing.cs)
        parents: List = []
    elif settings.chunking_strategy == "v2b":
        split_res = split_v2b(node_for_split, ing.hs, ing.cs)
        chunks = split_res["children"]
        parents = split_res["parents"]
    else:  # v2
        chunks = split_v2(node_for_split, ing.hs, ing.cs)
        parents = []

    # 3. 还原占位符
    chunks = ing.preprocessor.restore(chunks)
    parents = ing.preprocessor.restore(parents)

    # 4. 补 metadata + 结构前缀（机制 A：语义层感知层级）
    from note_assistant.indexing.ingestor import build_structural_prefix
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

    summary_chunks = ing.preprocessor.generate_summaries()
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
    if ing.docstore is not None:
        for p in parents:
            p.metadata["filepath"] = doc.filepath
            p.metadata["title"] = doc.title
            if dir_:
                p.metadata["dir"] = dir_
            p.metadata["wikilinks"] = doc.wikilinks
            if doc.tags:
                p.metadata["tags"] = doc.tags
            ing.docstore.add(p.metadata["parent_id"], p.page_content, p.metadata)

    # 6. 入库（children + summary + fm 进 ChromaDB；upsert 内部负责 embedding）
    all_chunks = chunks + summary_chunks + fm_chunks
    if all_chunks:
        ing.ingestor.upsert(all_chunks)

    # 更新 sync.db
    ing.sync.update_state(doc.filepath, abs_path)


def _delete_files(filepaths: List[str], ing, vault_root: Path) -> None:
    """删除指定文件的所有索引痕迹（ChromaDB chunks + sync.db 状态 + v2b docstore 父块）。"""
    for filepath in filepaths:
        ing.ingestor.collection.delete(where={"filepath": filepath})
        ing.sync.remove_state(filepath)
        logger.info("  ✅ 已删除: %s", filepath)

    # v2b：清理该文件的父块（docstore 无 per-file API，直接对内存实例过滤）
    if ing.docstore is not None:
        targets = set(filepaths)
        for pid in list(ing.docstore._store.keys()):  # noqa: SLF001
            if ing.docstore._store[pid]["metadata"].get("filepath") in targets:  # noqa: SLF001
                del ing.docstore._store[pid]  # noqa: SLF001


# ═══════════════════════════════════════════════════════════════
# 增量索引入口
# ═══════════════════════════════════════════════════════════════

def incremental_reindex(
    vault_path: str | None = None,
    *,
    filepaths: Optional[List[str]] = None,
    deleted_filepaths: Optional[List[str]] = None,
) -> dict:
    """
    增量索引：只处理变更集，秒级完成。

    Args:
        vault_path: vault 根目录；None → settings.vault_path
        filepaths: 相对 vault 根的路径列表；None → 全库变更比对（现状语义）。
                   指定 → 只重索引这些文件（watcher 单文件路径，跳过全库扫描）。
        deleted_filepaths: 相对 vault 根的路径列表；只删除这些文件的索引痕迹（不扫描）。

    Returns:
        {"status": str, "reindexed": int, "removed": int}
    """
    vault = vault_path or str(settings.vault_path.resolve())
    ing = _make_ingredients(vault)
    reindexed = 0
    removed = 0
    status = "ok"

    try:
        # ── 纯删除模式：不扫描 vault，直接删指定文件 ──
        if deleted_filepaths:
            _delete_files(list(deleted_filepaths), ing, Path(vault))
            removed += len(deleted_filepaths)

        # ── 单文件模式：watcher 已给出精确路径，跳过全库扫描 ──
        if filepaths is not None:
            for fp in filepaths:
                abs_path = Path(vault) / fp
                if not abs_path.exists():
                    logger.warning("autoindex: 文件已不存在，跳过: %s", fp)
                    continue
                # need_reindex 复核（mtime/sha256）：防重复事件空转
                if not ing.sync.need_reindex(fp, abs_path):
                    logger.info("autoindex: 内容未变，跳过: %s", fp)
                    continue
                try:
                    doc = ing.loader.load_file(abs_path)
                    _index_one(doc, ing)
                    reindexed += 1
                except Exception as e:  # noqa: BLE001
                    status = "partial"
                    logger.error("autoindex: 单文件索引失败（不阻塞后续）: %s: %s", fp, e)

        # ── 全库模式：扫描 + 比对（现状语义；纯删除/纯单文件模式跳过扫描）──
        if filepaths is None and not deleted_filepaths:
            all_docs = ing.loader.load_all()
            to_reindex = []
            to_remove = []

            for doc in all_docs:
                abs_path = doc.abs_path if hasattr(doc, "abs_path") else None
                if ing.sync.need_reindex(doc.filepath, abs_path):
                    to_reindex.append(doc)

            vault_files = {d.filepath for d in all_docs}
            for indexed_file in ing.sync.get_all_indexed():
                if indexed_file not in vault_files:
                    to_remove.append(indexed_file)

            logger.info(
                "📊 全库 %d 篇 → 需更新 %d / 需删除 %d（chunking=%s image=%s）",
                len(all_docs), len(to_reindex), len(to_remove),
                settings.chunking_strategy, settings.image_understand_enabled,
            )

            if to_remove:
                _delete_files(to_remove, ing, Path(vault))
                removed += len(to_remove)

            for i, doc in enumerate(to_reindex):
                _index_one(doc, ing)
                reindexed += 1
                logger.info("  ✅ [%d/%d] %s", i + 1, len(to_reindex), doc.filepath)

        # v2b：落盘 docstore（非 v2b 不碰，避免误加载）
        if ing.docstore is not None:
            ing.docstore.save()
    finally:
        ing.sync.close()

    logger.info("🎉 增量索引完成: reindexed=%d removed=%d", reindexed, removed)
    return {"status": status, "reindexed": reindexed, "removed": removed}