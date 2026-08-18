"""
来源内容类型判定 —— 从命中的 chunk 里识别「这是文本 / 表格 / 流程图 / 图片」，
并抽出前端渲染所需的原始载荷。

为什么单独成模块：

`SourceInfo.origin`（来源渠道：direct / graph）与 `SourceSchema.type`（内容类型：
text / table / mermaid / image）是两套语义。历史实现把前者直接透传给后者，
导致前端收到的永远是 "direct"，四个渲染分支全成死代码。契约拆开后，
「内容类型怎么判定」就需要一个共用的、可被 /ask 与 /agent 两条链路复用的落点。

判定原则：
- metadata 显式声明优先。summary chunk 的 `metadata["kind"]` 就是富结构类型，最权威。
- 正文 chunk 靠内容嗅探。前提是 preprocessor 的 restore() 已生效（P0 修复），
  还原后的正文里才会重新出现 ```mermaid / | 表格 | / ![](img) 这些原始语法。
- 载荷字段（raw_table / raw_mermaid / img_path）**不受 kind 约束，能抽就抽**。
  一个 chunk 可能同时含表格和图片，kind 只决定主图标，前端按字段有无决定渲染什么，
  这样不会因为主类型判错而丢掉可渲染的内容。
"""

import re
from typing import Any, Dict, Optional

# restore 之后正文里重新出现的原始语法
_MERMAID_RE = re.compile(r"```mermaid\s*\n.*?```", re.DOTALL)
_TABLE_RE = re.compile(r"(?:^|\n)((?:\|.*\n?){2,})")
_IMAGE_RE = re.compile(r"!\[\[([^\]]+)\]\]|!\[([^\]]*)\]\(([^)]+)\)")

VALID_KINDS = frozenset({"text", "table", "mermaid", "image"})


def _first_image_target(page_content: str) -> str:
    """取正文里第一张图的地址（去掉 Obsidian 的 |300 尺寸后缀）。"""
    m = _IMAGE_RE.search(page_content)
    if not m:
        return ""
    target = m.group(1) or m.group(3) or ""
    return target.partition("|")[0].strip()


def classify_source(page_content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """
    判定来源的内容类型并抽取渲染载荷。

    Returns:
        {"kind": "text|table|mermaid|image",
         "img_path": str, "raw_table": str, "raw_mermaid": str}
        —— 载荷字段缺省为空串，调用方自行决定是否转成 None。
    """
    meta = metadata or {}
    content = page_content or ""

    m_mermaid = _MERMAID_RE.search(content)
    # 正文嗅探不到（mermaid summary chunk 的 page_content 是结构化文本，无 ```fence）
    # 时，回退到 metadata 里的原始源码（preprocessor 写入的 mermaid_src）。
    raw_mermaid = m_mermaid.group(0) if m_mermaid else str(meta.get("mermaid_src") or "")

    m_table = _TABLE_RE.search(content)
    raw_table = m_table.group(1).strip() if m_table else ""

    # 图片地址：正文语法优先，其次 summary chunk 的 metadata（正文可能只有占位摘要）
    img_path = _first_image_target(content) or str(meta.get("img_src") or "")

    # ── kind 判定 ──
    declared = str(meta.get("kind") or "")
    if declared in VALID_KINDS and declared != "text":
        # summary chunk：kind 直接由 preprocessor 标注，最权威
        kind = declared
    elif img_path and (meta.get("has_image") or _IMAGE_RE.search(content)):
        # 图片优先于表格/流程图：表格和 mermaid 的 markdown 本身就能被 preview 渲染，
        # 只有图片必须走专门的 st.image 分支，判错代价最大。
        kind = "image"
    elif raw_mermaid:
        kind = "mermaid"
    elif raw_table:
        kind = "table"
    else:
        kind = "text"

    return {
        "kind": kind,
        "img_path": img_path,
        "raw_table": raw_table,
        "raw_mermaid": raw_mermaid,
        "render_hint": str(meta.get("render_hint") or ""),
        "diagram_type": str(meta.get("diagram_type") or ""),
        # P2：资产定位信息，供 /assets 端点与 [[IMG:]] 渲染
        "asset_id": str(meta.get("asset_id") or ""),
        "img_url": str(meta.get("img_url") or ""),
    }
