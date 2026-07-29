"""
父块存储（docstore）—— v2b 父子双存的核心组件。

设计见 docs/父子双存切分（v2b）设计方案.md。

背景：v2b 把笔记切成「子块（800 字，进 ChromaDB 检索）」+「父块（整节，只返回给 LLM）」。
父块不参与 embedding / BM25，只在检索命中子块后，按 parent_id 从本存储取回整节正文。

为何独立文件：
- ingestor 在索引期写入父块；hybrid 在检索期读取父块。
- 两方都依赖它，但谁也不该直接 import 对方（避免循环依赖），故独立成模块。
"""

from pathlib import Path
from typing import Any, Dict, Optional


class ParentDocstore:
    """
    父块存储：内存 dict + pickle 持久化。

    结构：{ parent_id: {"page_content": str, "metadata": dict} }
    page_content 为整节正文（已 restore），metadata 含 title/filepath/dir/heading_path/
    wikilinks/tags/parent_id/kind="parent"，供检索命中后回填到 RetrievalResult。
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._store: Dict[str, Dict[str, Any]] = {}

    # ──────────────────────────────────────────────
    # 写入期（ingestor 调用）
    # ──────────────────────────────────────────────
    def add(self, parent_id: str, page_content: str, metadata: Dict[str, Any]) -> None:
        """存入一个父块。"""
        self._store[parent_id] = {
            "page_content": page_content,
            "metadata": metadata,
        }

    def __len__(self) -> int:
        return len(self._store)

    # ──────────────────────────────────────────────
    # 读取期（hybrid 调用）
    # ──────────────────────────────────────────────
    def get(self, parent_id: str) -> Optional[Dict[str, Any]]:
        """按 id 取父块；不存在返回 None。"""
        return self._store.get(parent_id)

    # ──────────────────────────────────────────────
    # 持久化
    # ──────────────────────────────────────────────
    def save(self, path: str | Path | None = None) -> None:
        """保存到 pickle（覆盖写）。"""
        save_path = Path(path) if path else self.path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            import pickle
            pickle.dump(self._store, f)

    @classmethod
    def load(cls, path: str | Path) -> "ParentDocstore":
        """从 pickle 加载；文件不存在/损坏由调用方捕获。"""
        import pickle
        store = cls(path)
        with open(path, "rb") as f:
            store._store = pickle.load(f)
        return store
