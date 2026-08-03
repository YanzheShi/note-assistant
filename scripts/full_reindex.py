"""
全量重建索引脚本：清空 ChromaDB → 重新加载 → 重新切分 → 重新向量化 → 重新入库，
并重建两个派生产物：BM25 稀疏索引（data/bm25.pkl）与 WikiGraph（data/bm25.graph）。

注意：本脚本直接委托 Ingestor.index_vault(wipe=True)，即生产索引的「最新流程」，
确保与 API / 增量索引共用同一套逻辑（图片 enricher 注入、chunking_strategy 分流、
dir + 结构前缀、v2b 父块 docstore、ingestor.upsert 入库）。

派生产物为何要在此重建：index_vault 只写 ChromaDB + v2b docstore；
BM25/WikiGraph 是它们的下游派生，wipe 重建后若不刷新，稀疏检索与图扩展
会一直跑在旧语料/旧图上（静默降级，不报错）。

用法:
    # 默认从 config.vault_path 读取
    uv run python scripts/full_reindex.py

    # 指定 vault 路径
    uv run python scripts/full_reindex.py --vault /path/to/vault

    # 跳过确认（自动化用）
    uv run python scripts/full_reindex.py --yes

    # 只打印配置、不执行（dry-run）
    uv run python scripts/full_reindex.py --dry-run

前置条件:
    - Ollama 服务已启动（bge-m3:latest 模型已 pull）
    - .env 配置了正确的 vault_path / chunking_strategy 等
    - 图片 VLM 理解开启时（image_understand_enabled）：VLM_* 通道可用
"""

import sys
import time
from pathlib import Path

# Windows GBK 编码兼容
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 确保 src 在 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _rebuild_derived(vault_path: str) -> None:
    """重建派生产物：① BM25 稀疏索引 ② WikiGraph（index_vault 不覆盖这两项）。

    各步独立 try/except：主索引此时已成功，派生重建失败只打印明确告警，
    不让整个全量重建白跑——但失败项必须手动补，否则稀疏检索/图扩展跑旧数据。
    """
    from note_assistant.config import settings

    # ① BM25：从 ChromaDB 全量拉 chunks 重建（与主索引同源，天然一致）
    t0 = time.time()
    try:
        from note_assistant.retrieval.sparse_retriever import BM25Retriever

        bm25 = BM25Retriever.from_chroma()
        bm25.save()
        print(f"  ✅ BM25 索引重建完成（{time.time() - t0:.1f}s）→ {settings.bm25_index_path.resolve()}")
    except Exception as e:  # noqa: BLE001
        print(f"  ❌ BM25 索引重建失败（稀疏检索仍在旧索引上，请手动重跑本步）: {e}")

    # ② WikiGraph：从全库 wikilinks 重建（agent/rag_chain 的图扩展依赖）
    t0 = time.time()
    try:
        from note_assistant.indexing.vault_loader import VaultLoader
        from note_assistant.retrieval.graph import WikiGraph

        docs = VaultLoader(vault_path).load_all()
        g = WikiGraph()
        g.build_from_docs(docs)
        g.save()
        print(f"  ✅ WikiGraph 重建完成（{time.time() - t0:.1f}s）：{g.summary()}")
    except Exception as e:  # noqa: BLE001
        print(f"  ❌ WikiGraph 重建失败（图扩展仍在旧图上，请手动重跑本步）: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="全量重建索引（委托 Ingestor.index_vault）")
    parser.add_argument("--vault", default=None, help="vault 路径（默认读 config）")
    parser.add_argument("--yes", "-y", action="store_true", help="跳过确认")
    parser.add_argument("--dry-run", action="store_true", help="只打印配置、不执行")
    args = parser.parse_args()

    from note_assistant.config import settings
    from note_assistant.indexing.ingestor import Ingestor

    vault_path = args.vault or str(settings.vault_path.resolve())

    print("=" * 60)
    print("  全量重建索引（委托 Ingestor.index_vault, wipe=True）")
    print("=" * 60)
    print(f"\n  Vault: {vault_path}")
    print(f"  ChromaDB: {settings.chroma_persist_dir.resolve()}")
    print(f"  Embed Model: {settings.embed_model}")
    print(f"  Chunking Strategy: {settings.chunking_strategy}")
    print(f"  Image Understand Enabled: {settings.image_understand_enabled}")
    print(f"  Parent Docstore: {settings.parent_docstore_path.resolve()}")
    print(f"  BM25 Index: {settings.bm25_index_path.resolve()}（主索引后全量重建）")
    print(f"  WikiGraph: {settings.bm25_index_path.with_suffix('.graph').resolve()}（主索引后全量重建）")

    if args.dry_run:
        print("\n  [DRY RUN] 只打印配置，未执行重建")
        return

    if not args.yes:
        print("\n  ⚠️  此操作会清空 ChromaDB 并重建所有索引！")
        ans = input("  确认继续? [y/N]: ").strip().lower()
        if ans != "y":
            print("  已取消")
            return

    total_start = time.time()
    ingestor = Ingestor()
    # index_vault(wipe=True) 内部会 delete_all 后按最新流程重建，
    # 自动注入 image_enricher、遵守 chunking_strategy、补 dir+结构前缀、处理 v2b docstore。
    stats = ingestor.index_vault(vault_path=vault_path, wipe=True)
    total_time = time.time() - total_start

    print(f"\n{'=' * 60}")
    print(f"  主索引重建完成")
    print(f"{'=' * 60}")
    print(f"  文件数:   {stats.get('files')}")
    print(f"  Chunks:   {stats.get('chunks')}")
    print(f"  索引耗时: {total_time:.2f}s")
    print(f"  ChromaDB: {settings.chroma_persist_dir.resolve()}")

    # 派生产物重建（BM25 + WikiGraph）：wipe 重建后必须刷新，否则跑旧数据
    print(f"\n{'=' * 60}")
    print(f"  重建派生产物（BM25 / WikiGraph）")
    print(f"{'=' * 60}")
    _rebuild_derived(vault_path)

    print(f"\n  全流程总耗时: {time.time() - total_start:.2f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
