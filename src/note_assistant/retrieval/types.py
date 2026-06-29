from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class RetrievalResult:
    """检索结果的业务层统一结构"""
    score: float                          # 最终检索分数（越高越相关）
    page_content: str                     # 匹配的文本内容
    metadata: Dict[str, Any]              # 来源 metadata（filepath/title/heading_path 等）
    index: Optional[int] = None           # 在 corpus 中的位置（sparse 用）
    dense_score: Optional[float] = None   # dense 通路分数（调试/优化用）
    sparse_score: Optional[float] = None  # sparse 通路分数（调试/优化用）

    @property
    def filepath(self) -> str:
        return self.metadata.get("filepath", "")

    @property
    def title(self) -> str:
        return self.metadata.get("title", "")

    def __repr__(self) -> str:
        parts = [f"score={self.score:.4f}"]
        if self.dense_score is not None:
            parts.append(f"dense={self.dense_score:.4f}")
        if self.sparse_score is not None:
            parts.append(f"sparse={self.sparse_score:.4f}")
        return f"RetrievalResult({', '.join(parts)}, filepath={self.filepath!r})"
