"""
全量重建索引脚本：清空 ChromaDB → 重新加载 → 重新切分 → 重新向量化 → 重新入库。

注意：本脚本直接委托 Ingestor.index_vault(wipe=True)，即生产索引的「最新流程」，
确保与 API / 增量索引共用同一套逻辑（图片 enricher 注入、chunking_strategy 分流、
dir + 结构前缀、v2b 父块 docstore、ingestor.upsert 入库）。

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
"""

import sys
import time
from pathlib import Path

# Windows GBK 编码兼容
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 确保 src 在 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


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
    print(f"  全量重建完成")
    print(f"{'=' * 60}")
    print(f"  文件数:   {stats.get('files')}")
    print(f"  Chunks:   {stats.get('chunks')}")
    print(f"  总耗时:   {total_time:.2f}s")
    print(f"  ChromaDB: {settings.chroma_persist_dir.resolve()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
