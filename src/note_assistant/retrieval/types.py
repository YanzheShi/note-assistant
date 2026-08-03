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
    structural_score: Optional[float] = None  # 结构分（层级标题 boost，调试/排序观察用）

    @property
    def filepath(self) -> str:
        return self.metadata.get("filepath", "")

    @property
    def title(self) -> str:
        return self.metadata.get("title", "")

    def identity_key(self) -> tuple:
        """chunk 身份键：跨轮/跨工具累积去重用。

        只用 ``(filepath, heading_path)`` 不够——图片/表格/mermaid 的 summary chunk
        与同章节正文 chunk（v2b 下是整节父块）共享同一 heading，按章节去重会让
        二者互斥：谁分数高谁先进 accumulated，另一个被静默丢弃。这解释了
        「有的 query 图片进 sources、有的不进」的随机性（取决于分数竞速）。

        加上 ``kind`` + ``placeholder`` 后：富结构 chunk 各自成为独立身份
        （placeholder 是抽取期 uuid，天然唯一，同节多张图也不互相挤掉）；
        普通正文 chunk 这两项均为空串，正文 vs 正文的去重行为与原先逐字节一致。
        """
        meta = self.metadata if isinstance(self.metadata, dict) else {}
        return (
            self.filepath,
            meta.get("heading_path", ""),
            str(meta.get("kind") or ""),
            str(meta.get("placeholder") or meta.get("asset_id") or ""),
        )

    def __repr__(self) -> str:
        parts = [f"score={self.score:.4f}"]
        if self.dense_score is not None:
            parts.append(f"dense={self.dense_score:.4f}")
        if self.sparse_score is not None:
            parts.append(f"sparse={self.sparse_score:.4f}")
        return f"RetrievalResult({', '.join(parts)}, filepath={self.filepath!r})"
