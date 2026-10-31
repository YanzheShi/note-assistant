"""HTML 文档加载：把 HTML 归一化为 Markdown，产出与笔记一致的 DocNode。

设计取舍：
- 只在加载层做归一化，preprocessor / splitter / ingestor 全部复用 Markdown 链路。
  富结构（表格、代码、图片）仍能被占位符保护与摘要 chunk 覆盖。
- 静态 DOM 解析，不用无头浏览器：本项目的 HTML 都是静态正文，且大量答案在
  ``<details>`` 折叠区里——渲染后取 innerText 会丢掉折叠内容，DOM 解析不会。
- 主动内容（script/style/iframe 等）一律剥离；正文转换后仍走既有注入检测。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List

from bs4 import BeautifulSoup, Comment
from markdownify import MarkdownConverter

from note_assistant.indexing.markdown_utils import extract_headings, extract_wikilinks
from note_assistant.indexing.types import DocNode

_HTML_SUFFIXES = {".html", ".htm"}

# 页面噪声：导航、目录、工具栏、统计条、题目标签。它们不承载答案，
# 且目录/统计往往是标题与计数的重复，留着只会制造伪命中。
_NOISE_SELECTORS = (
    "script, style, noscript, iframe, object, embed, svg, canvas, "
    "form, input, button, select, textarea, "
    "nav, #toc, .toc, .toolbar, .stats, .q .tag, .q .freq"
)

_CHARSET_RE = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?\s*([\w-]+)""", re.I)


def decode_html_bytes(raw: bytes) -> str:
    """按 BOM / meta charset / UTF-8 / GB18030 的顺序解码 HTML 字节。

    中文导出文档里 GBK/GB2312 很常见；meta charset 在文件头是纯 ASCII，
    因此即使正文不是 UTF-8 也能先拿到正确编码再解码。
    """
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig", errors="replace")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass

    match = _CHARSET_RE.search(raw[:4096])
    if match:
        encoding = match.group(1).decode("ascii", "ignore").lower()
        encoding = {"utf8": "utf-8", "gb2312": "gb18030", "gbk": "gb18030"}.get(
            encoding, encoding
        )
        try:
            return raw.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            pass

    try:
        return raw.decode("gb18030")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


def _meta_keywords(soup: BeautifulSoup) -> List[str]:
    for meta in soup.find_all("meta"):
        if (meta.get("name") or "").strip().lower() != "keywords":
            continue
        raw = (meta.get("content") or "").strip()
        if raw:
            return [x.strip() for x in re.split(r"[,，;；]", raw) if x.strip()]
    return []


def _canonical_url(soup: BeautifulSoup) -> str:
    for link in soup.find_all("link"):
        rel = link.get("rel") or []
        if isinstance(rel, str):
            rel = [rel]
        if any(str(r).lower() == "canonical" for r in rel):
            href = (link.get("href") or "").strip()
            if href:
                return href
    return ""


def _doc_title(soup: BeautifulSoup, path: Path) -> str:
    title_tag = soup.find("title")
    title = title_tag.get_text(" ", strip=True) if title_tag else ""
    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(" ", strip=True) if h1 else ""
    return title or path.stem


def _internal_links(soup: BeautifulSoup) -> List[str]:
    """站内 HTML 链接 → wikilink 目标名（stem）。

    这批语料没有跨文件链接，但保留这条能力：以后 HTML 之间互相引用时，
    双链图能把它们连起来，而不是每篇都成孤岛。
    """
    from urllib.parse import unquote, urlparse

    links: List[str] = []
    for anchor in soup.find_all("a", href=True):
        href = (anchor.get("href") or "").strip()
        if not href or href.startswith(("#", "mailto:", "javascript:", "tel:")):
            continue
        parsed = urlparse(href)
        if parsed.scheme or parsed.netloc:
            continue
        target = Path(unquote(parsed.path))
        if target.suffix.lower() not in _HTML_SUFFIXES:
            continue
        if target.stem:
            links.append(target.stem)
    return links


def _normalize_dom(soup: BeautifulSoup) -> None:
    for el in soup.select(_NOISE_SELECTORS):
        el.decompose()
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()

    # 编辑器导出的代码块是 <div class="code-pre"><code><span>…；先降级成 <pre>，
    # 让统一的 convert_pre 保留换行与缩进。
    for div in soup.find_all("div", class_="code-pre"):
        pre = soup.new_tag("pre")
        pre.string = div.get_text()
        div.replace_with(pre)

    # 手写 HTML 偶发把 <pre> 塞进 <h2>/<p> 等行内容器（浏览器会容忍），
    # 但 Markdown 的标题转换只吃行内文本，代码块会整段消失。统一提升到容器之后。
    inline_parents = {
        "h1", "h2", "h3", "h4", "h5", "h6", "p", "summary",
        "a", "strong", "b", "em", "i", "span", "code",
    }
    for pre in list(soup.find_all("pre")):
        container = None
        parent = pre.parent
        while parent is not None and parent.name in inline_parents:
            container = parent
            parent = parent.parent
        if container is not None:
            container.insert_after(pre)

    # <details> 默认折叠，但答案就在 DOM 里。summary 是 UI 文案，降级为加粗小标题，
    # 内容原样保留；details 本身改成 div，避免未知标签在 Markdown 里产生怪输出。
    for details in soup.find_all("details"):
        # 只认直接子节点：嵌套 details 的内层 summary 不能被外层误删。
        summary = details.find("summary", recursive=False)
        if summary:
            label = " ".join(summary.get_text(" ", strip=True).split())
            label = re.sub(r"^展开\s*[:：]\s*", "", label)
            summary.decompose()
            if label:
                p = soup.new_tag("p")
                strong = soup.new_tag("strong")
                strong.string = label
                p.append(strong)
                details.insert(0, p)
        details.name = "div"

    for img in soup.find_all("img"):
        src = (
            img.get("src")
            or img.get("data-src")
            or img.get("data-original")
            or ""
        ).strip()
        if not src and img.get("srcset"):
            src = img["srcset"].split(",")[0].strip().split(" ")[0]
        alt = img.get("alt") or ""
        if not src or src.lower().startswith("data:"):
            # assets.resolve_image 不消费 data URI；直接把 base64 写进 chunk 会把
            # 整篇正文的 embedding 稀释掉，这里只保留可读的 alt。
            img.replace_with(alt)
        else:
            img["src"] = src


def _int_attr(el, name: str, limit: int) -> int:
    try:
        value = int(str(el.get(name, 1)))
    except (TypeError, ValueError):
        return 1
    return max(1, min(value, limit))


def _cell_text(cell) -> str:
    for br in cell.find_all("br"):
        prev = br.previous_sibling
        nxt = br.next_sibling
        before = prev.get_text() if hasattr(prev, "get_text") else str(prev or "")
        after = nxt.get_text() if hasattr(nxt, "get_text") else str(nxt or "")
        # 表格里的换行多半是视觉折行：code-tutor-<br>agent 不能变成
        # "code-tutor- agent"，否则标识符在 BM25 里被切断。
        glue = "" if before.rstrip().endswith(("-", "_")) or after.lstrip().startswith(("-", "_")) else " "
        br.replace_with(glue)
    text = re.sub(r"\s+", " ", cell.get_text(" ", strip=True)).strip()
    return text.replace("|", r"\|")


def _table_to_markdown(table) -> str:
    """HTML 表格 → GFM 管道表，展开 colspan / rowspan。

    合并单元格在管道表里没有原生表达，这里按视觉布局展开成重复值；
    rowspan 跨行时把值填到后续行，保证列对齐、文本可检索。
    """
    rows = [tr for tr in table.find_all("tr") if tr.find_parent("table") is table]
    if not rows:
        return ""

    grid: List[List[str]] = []
    pending: dict[int, tuple[str, int]] = {}

    for tr in rows:
        cells = tr.find_all(["td", "th"], recursive=False)
        row: List[str] = []
        col = 0
        ci = 0
        while ci < len(cells) or any(start >= col for start in pending):
            if col in pending:
                text, remaining = pending[col]
                row.append(text)
                if remaining <= 1:
                    pending.pop(col, None)
                else:
                    pending[col] = (text, remaining - 1)
                col += 1
                continue
            if ci >= len(cells):
                row.append("")
                col += 1
                continue
            cell = cells[ci]
            ci += 1
            text = _cell_text(cell)
            colspan = _int_attr(cell, "colspan", 50)
            rowspan = _int_attr(cell, "rowspan", 200)
            for offset in range(colspan):
                row.append(text)
                if rowspan > 1:
                    pending[col + offset] = (text, rowspan - 1)
            col += colspan
        grid.append(row)

    if not grid:
        return ""
    ncols = max(len(row) for row in grid)
    if not ncols:
        return ""

    def fmt(row: List[str]) -> str:
        padded = row + [""] * (ncols - len(row))
        return "| " + " | ".join(padded) + " |"

    lines = [fmt(grid[0]), "| " + " | ".join(["---"] * ncols) + " |"]
    lines.extend(fmt(row) for row in grid[1:])
    caption = table.find("caption")
    prefix = ""
    if caption:
        caption_text = re.sub(r"\s+", " ", caption.get_text(" ", strip=True)).strip()
        if caption_text:
            prefix = f"**{caption_text}**\n\n"
    return f"\n\n{prefix}" + "\n".join(lines) + "\n\n"


def _code_language(pre) -> str:
    for node in (pre.find("code"), pre):
        if node is None:
            continue
        for cls in node.get("class", []):
            for prefix in ("language-", "lang-"):
                if cls.startswith(prefix):
                    return cls[len(prefix):]
    return ""


class _HtmlConverter(MarkdownConverter):
    def convert_pre(self, el, text, parent_tags):
        # 不用 markdownify 的 text：它经过子标签转换后会吃掉行首缩进，
        # 代码块一旦失真，检索到的代码片段就没法用。
        for br in el.find_all("br"):
            br.replace_with("\n")
        code = el.get_text().replace("\xa0", " ").strip("\n")
        if not code.strip():
            return ""
        fence = "```"
        while fence in code:
            fence += "`"
        language = _code_language(el)
        return f"\n\n{fence}{language}\n{code}\n{fence}\n\n"

    def convert_table(self, el, text, parent_tags):
        return _table_to_markdown(el)


def _dedup(items: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def load_html(path: Path, vault_root: Path) -> DocNode:
    """加载单个 HTML 文件，返回与 Markdown 笔记同构的 DocNode。"""
    html = decode_html_bytes(path.read_bytes()).replace("\r\n", "\n").replace("\r", "\n")
    soup = BeautifulSoup(html, "html.parser")
    title = _doc_title(soup, path)
    tags = _meta_keywords(soup)
    source_url = _canonical_url(soup)
    internal_links = _internal_links(soup)

    body = soup.body or soup
    _normalize_dom(body)
    converter = _HtmlConverter(
        heading_style="ATX",
        bullets="-",
        # 正文里的 snake_case 标识符很多（路径、函数名），转义下划线会把
        # BM25 的分词切碎，得不偿失。
        escape_underscores=False,
    )
    raw_md = re.sub(r"\n{3,}", "\n\n", converter.convert(str(body))).strip()

    front_matter = {"source_type": "html"}
    if tags:
        front_matter["tags"] = tags
    if source_url:
        front_matter["source_url"] = source_url

    return DocNode(
        filepath=str(path.relative_to(vault_root)),
        abs_path=path,
        raw_md=raw_md,
        front_matter=front_matter,
        title=title,
        tags=tags,
        wikilinks=_dedup(internal_links + extract_wikilinks(raw_md)),
        headings=extract_headings(raw_md),
    )
