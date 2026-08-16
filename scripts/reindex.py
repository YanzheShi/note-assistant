"""
增量索引 CLI 薄壳 —— 实现已迁入包内 `note_assistant.indexing.reindex`。

与全量 index_vault 共用同一套「最新流程」：
- 注入 make_image_enricher（图片 VLM 理解 / SVG 原生解析）
- 遵守 chunking_strategy（v1 / v2 / v2b 分流）
- 补 dir 元数据 + build_structural_prefix 结构前缀
- v2b 父块写入 ParentDocstore
- 统一经 Ingestor.upsert 入库（内部负责 embedding）

用法:
    python scripts/reindex.py [vault_path]

    vault_path 默认从 config.vault_path 读取

已知缺口:
    增量索引不重建 BM25（data/bm25.pkl）与 WikiGraph（data/bm25.graph）这两个
    全量派生产物——长期增量后二者会滞后于 ChromaDB。定期跑一次
    scripts/full_reindex.py 即可全量刷新（该脚本已内置两步重建）。
"""

import sys
from pathlib import Path

# 确保 src 在 path 里
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from note_assistant.indexing.reindex import incremental_reindex  # noqa: E402


if __name__ == "__main__":
    vault = sys.argv[1] if len(sys.argv) > 1 else None
    incremental_reindex(vault)