"""
结构分（structural score）：query 与 chunk 结构元数据的重叠度。

背景：Obsidian 笔记是「文件夹 → 文档 → 多级标题」的层级结构，用户提问常以
"哪篇 / 哪章"为锚点（如"Code Agent 架构的关键设计点"）。chunk 的元数据里
已经带有这些结构信号（dir / title / heading_path，见 ingestor.index_vault），
本模块把它们单独抽成一路分数，叠在 dense+sparse 融合分之上，让结构命中段
稳定靠前（"层级优先"）。

设计要点（详见 docs/层级检索与结构优先设计方案.md）：
- 复用 jieba 分词（BM25 已在用，无新依赖）。
- score ∈ [0, 1]：title 权重最高(0.6)、heading_path 次之(0.3)、dir 最低(0.1)。
- title_hit：query 归一化后**包含**文档标题归一化片段 → 强信号（硬兜底）。
- 与"前缀拼进 page_content"互补：本模块只吃元数据、不污染正文语义。
"""

import re

import jieba

# 归一化时清除的标点/括号/空白（《》「」【】[]() 等），保留中文/英文/数字
_PUNCT_RE = re.compile(r"[《》「」『』【】\[\]\(\)（）\s]")


def _normalize(text: str) -> str:
    """轻量归一：去书名号/括号/空白，便于精确包含判定与分词稳定。"""
    return _PUNCT_RE.sub("", text or "")


def structural_score(query: str, meta: dict) -> tuple[float, bool]:
    """
    计算 query 与 chunk 结构元数据的重叠度。

    Args:
        query: 用户查询（可能已被 query_rewrite 改写）。
        meta:  chunk 的 metadata（需含 dir / title / heading_path 之一）。

    Returns:
        (score, title_hit)
        - score ∈ [0, 1]：结构重叠度；query 为空或非空 token 时返回 0.0。
        - title_hit：query 包含文档标题（归一化后）→ True，供混合融合追加硬兜底 bonus。
    """
    if not query:
        return 0.0, False

    q_norm = _normalize(query)
    q_tokens = {t for t in jieba.lcut(q_norm) if t.strip()}
    if not q_tokens:
        return 0.0, False

    title = (meta.get("title") or "")
    heading_path = (meta.get("heading_path") or "")
    dir_ = (meta.get("dir") or "")

    # title 精确命中（归一化包含判定）：query 里出现完整文档名
    title_hit = False
    if title and len(title) > 1:
        t_norm = _normalize(title)
        if t_norm and t_norm in q_norm:
            title_hit = True

    # 各字段 token 池（heading_path 用空格替换 " > " 分隔符再分词）
    title_tokens = {t for t in jieba.lcut(_normalize(title)) if t.strip()} if title else set()
    if heading_path:
        heading_text = _normalize(heading_path.replace(" > ", " "))
        heading_tokens = {t for t in jieba.lcut(heading_text) if t.strip()}
    else:
        heading_tokens = set()
    if dir_:
        dir_text = _normalize(dir_.replace("/", " ").replace("\\", " "))
        dir_tokens = {t for t in jieba.lcut(dir_text) if t.strip()}
    else:
        dir_tokens = set()

    t_overlap = len(q_tokens & title_tokens) / len(q_tokens)
    h_overlap = len(q_tokens & heading_tokens) / len(q_tokens)
    d_overlap = len(q_tokens & dir_tokens) / len(q_tokens)

    # title 权重最高，heading 次之，dir 最低；天然归一化到 [0, 1]
    score = 0.6 * t_overlap + 0.3 * h_overlap + 0.1 * d_overlap
    return min(1.0, score), title_hit
