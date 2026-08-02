# src/note_assistant/indexing/types.py （新建这个文件，专门放业务类型）
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any

@dataclass
class DocNode:
    """Obsidian笔记的业务层数据结构，零框架依赖"""
    # === 标识类 ===
    filepath: str           # 相对vault根的路径（比如"01-大模型/01-基础.md"）
    abs_path: Path          # 绝对路径（调试/溯源用）
    # === 内容类 ===
    raw_md: str             # 原始markdown内容（已剥离front matter）
    front_matter: Dict[str, Any]  # 解析后的front matter（嵌套结构保留）
    # === 特征类 ===
    title: str              # 笔记标题（优先取fm里的title，否则取文件名）
    tags: List[str]         # fm里的tags，默认空列表
    wikilinks: List[str]    # 全文提取的[[link]]目标名（去重保序）
    headings: List[Dict[str, Any]] = field(default_factory=list)  # 标题树：{"level":2,"text":"xxx","line":42}


@dataclass
class ExtractedChunk:
    uid: str
    kind: str          # "table" | "mermaid" | "image" | "code"
    placeholder: str   # 在原文中的占位符
    raw: str           # 原始内容（图片为完整 markdown 语法，如 "![alt](path)"，供 restore 还原渲染）
    context: str       # 前后文（用于生成描述/summary），已剔除占位符噪声
    # 结构化附加信息，避免继续往 context 字符串里拼接（如 image 的 src/alt/dims）
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """切分后的 chunk，业务层统一结构"""
    page_content: str
    metadata: Dict[str, Any]
    kind: str = "text"  # "text" | "extracted_summary" | "front_matter"