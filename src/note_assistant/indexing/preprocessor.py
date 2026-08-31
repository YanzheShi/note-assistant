import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from note_assistant.indexing.types import DocNode, ExtractedChunk, Chunk


def _normalize_note_dir(note_dir: str) -> str:
    """归一化笔记所在目录：vault 根（pathlib 给 ``.``）统一成空串，分隔符统一为 ``/``。"""
    d = (note_dir or "").strip().replace("\\", "/")
    if d in (".", "", "/"):
        return ""
    return d.strip("/")


def _note_dir_of(filepath: str) -> str:
    """从笔记路径取所在目录，保持「相对 vault 根」口径（join 动作交给 resolve_image）。"""
    return _normalize_note_dir(str(Path(filepath or "").parent))


# 资产定位字段：图片 summary chunk 由 enricher 算出，需回绑到内联同一张图的正文 chunk，
# 否则来源面板拿到的只有裸相对路径（渲染必 404）。
_ASSET_KEYS = ("asset_id", "img_url", "render_hint")


# 所有占位符的统一形态（uuid4().hex[:8] → 8 位十六进制）。
# 用于把 context 窗口里混入的占位符洗掉：抽取是分阶段的（code → table → mermaid → image），
# 后一阶段取上下文时看到的文本已被前一阶段替换过，不清洗会把 "<CODE_UID_a3f2b1c8>"
# 这类噪声串当成语义上下文喂给 summary / 下游 VLM prompt。
_PLACEHOLDER_RE = re.compile(
    r"<CODE_UID_[0-9a-f]{8}>"
    r"|\[TABLE_UID_[0-9a-f]{8}\]"
    r"|\[MERMAID_UID_[0-9a-f]{8}\]"
    r"|\[IMAGE_UID_[0-9a-f]{8}\]"
)


def strip_placeholders(text: str) -> str:
    """剔除文本中的富结构占位符，并压掉因此产生的多余空白。"""
    if not text:
        return ""
    return re.sub(r"[ \t]{2,}", " ", _PLACEHOLDER_RE.sub(" ", text)).strip()


class RichPreprocessor:
    def __init__(self, *, image_enricher=None):
        """
        Args:
            image_enricher: 可选钩子，签名 `f(ext: ExtractedChunk, heading_path: str)
                -> Optional[tuple[str, dict]]`。返回 (富摘要文本, 额外 metadata) 时，
                图片 summary chunk 使用该结果（如 VLM 结构化理解 / SVG 原生解析的产出）；
                返回 None 时退化为默认摘要（只 alt+上下文）。

                默认 None：不触发任何取图/VLM，完全保持原行为（离线、无网、零成本）。
                生产环境由 ingestor 注入真实 enricher（见 indexing/understanding.py）。
        """
        self.extracted: list[ExtractedChunk] = []
        self._image_enricher = image_enricher
        # 当前笔记在 vault 内的所在目录（相对 vault 根，根目录为空串）。
        # Markdown 相对链接的语义是「相对所在笔记」，Obsidian 附件常放在笔记旁的
        # assets/ 目录里；不带这个信息，下游取图只能按 vault 根解析而整体落空。
        self._note_dir: str = ""
        # placeholder → heading_path，在 restore() 阶段回填。
        # 富结构被抽走时还不知道它落在哪一节，只有切分后的 chunk 才带 heading_path，
        # 因此由 restore() 反查，供 generate_summaries() 给 summary chunk 补章节定位。
        self._placeholder_heading: dict[str, str] = {}

    # ──────────────────────────────────────────────
    # 主入口
    # ──────────────────────────────────────────────

    def process(self, md_text: str, *, note_dir: str = "") -> str:
        """将富结构替换为占位符，返回清洗后的文本

        Args:
            md_text: 笔记原始 markdown
            note_dir: 该笔记在 vault 内的所在目录（相对 vault 根；根目录传空串）。
                图片抽取时记下它，供下游按「相对笔记」而非「相对 vault 根」解析附件。
        """
        self.extracted = []
        self._placeholder_heading = {}
        self._note_dir = _normalize_note_dir(note_dir)
        text = md_text

        # P1: 抽取 Mermaid（必须在 code fence 保护之前！否则 ```mermaid 会被
        #      _protect_code_fences 当普通代码块抢先捕获成 kind="code"，设计文档
        #      5.B.4.4 期望的 ExtractedChunk(kind="mermaid") 永远到不了）。
        text = self._extract_mermaid(text)

        # P2: code fence 保护（防 MarkdownHeaderTextSplitter 误解析；此时 mermaid
        #      已是 MERMAID_UID 占位符，不会再被误伤）
        text = self._protect_code_fences(text)

        # P3: 抽取表格
        text = self._extract_tables(text)

        # P4: 抽取图片
        text = self._extract_images(text, context_window=80)

        return text

    def process_with_meta(self, node: DocNode) -> tuple[str, list[Chunk]]:
        """
        处理文本 + 从 front_matter 生成额外可检索 chunk。
        入参是 DocNode，直接从 node 取 raw_md 和 front_matter。

        返回: (cleaned_text, extra_chunks)
            extra_chunks: list[Chunk] — front_matter 衍生的可检索 chunk
        """
        cleaned = self.process(node.raw_md, note_dir=_note_dir_of(node.filepath))
        extra_chunks = self._build_front_matter_chunks(node.front_matter)
        return cleaned, extra_chunks

    # ──────────────────────────────────────────────
    # 还原（切分后调用）
    # ──────────────────────────────────────────────

    def restore(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        将 chunk 中的占位符还原为原始内容。
        在 split() 之后、embed 之前调用。

        逻辑：
        1. 扫描每个 chunk 的 page_content，找出其中包含的占位符
        2. 用 self.extracted 的原始内容替换占位符
        3. 在 metadata 中记录该 chunk 含哪些富结构（供下游过滤/展示用）
        """
        restored = []
        for chunk in chunks:
            content = chunk.page_content
            metadata = dict(chunk.metadata)

            # 找出这个 chunk 里出现了哪些占位符
            found = self._find_placeholders(content)

            if found:
                # metadata 标记含哪些富结构
                metadata["has_code"] = any(e.kind == "code" for e in found)
                metadata["has_table"] = any(e.kind == "table" for e in found)
                metadata["has_mermaid"] = any(e.kind == "mermaid" for e in found)
                metadata["has_image"] = any(e.kind == "image" for e in found)

                # 回填 placeholder → heading_path（首次记录优先：children 先于 parents
                # 调用 restore，子块的 heading_path 更精确，不被父块的粗粒度路径覆盖）
                hp = metadata.get("heading_path") or ""
                if hp:
                    for ext in found:
                        self._placeholder_heading.setdefault(ext.placeholder, hp)

                # 还原占位符 → 原始内容
                for ext in found:
                    content = content.replace(ext.placeholder, ext.raw)

            restored.append(Chunk(
                page_content=content,
                metadata=metadata,
                kind=chunk.kind,
            ))
        return restored

    def _find_placeholders(self, text: str) -> list[ExtractedChunk]:
        """扫描文本，找出所有已知的占位符（O(n) 遍历 self.extracted）"""
        return [ext for ext in self.extracted if ext.placeholder in text]

    # ──────────────────────────────────────────────
    # 查询接口
    # ──────────────────────────────────────────────

    def get_extracted(self, kind: Optional[str] = None) -> list[ExtractedChunk]:
        """获取已抽取的富结构，可按 kind 过滤"""
        if kind:
            return [e for e in self.extracted if e.kind == kind]
        return list(self.extracted)

    # ──────────────────────────────────────────────
    # 富结构摘要 → 可检索 chunk
    # ──────────────────────────────────────────────

    def generate_summaries(self) -> list[Chunk]:
        """
        为抽取的富结构生成可检索的文本描述，作为独立 chunk 入库。

        问题：表格/mermaid/图片被抽走后，它们的内容在检索空间里就消失了。
        解决：给每个 ExtractedChunk 生成一段自然语言描述，单独嵌入一个 "summary chunk"，
              metadata 标记 source="extracted" + 对应 placeholder，
              命中后 restore 阶段可以还原原始内容。

        返回: list[Chunk]
        """
        summary_chunks = []
        for ext in self.extracted:
            # 各分支可往 meta_extra 注入额外 metadata（如 mermaid 的 render_hint）。
            meta_extra: dict = {}

            if ext.kind == "table":
                # 表格：caption 取前一行 + 列数信息
                col_count = max(ext.raw.count('|') - 1, 1)
                first_row = ext.raw.strip().split('\n')[0].strip('|').strip()
                summary = f"表格（约{col_count}列）: {ext.context or first_row}"

            elif ext.kind == "mermaid":
                # Mermaid：P1 原生解析层（5.B.4.4）——把图解析成 DiagramGraph，
                # 节点/边 label 拼成结构化文本入索引（比旧式「图类型 + 前一行 caption」
                # 信息密度高得多），并写 render_hint 供前端原生渲染（mermaid.render）。
                # 解析失败（罕见）降级为旧弱摘要，绝不中断索引。
                # P1-b：原始 mermaid 源码恒入 metadata（mermaid_src/raw_mermaid），并经
                #   classify_source → SourceInfo → SourceSchema → API 透传到前端
                #   原生渲染（frontend/components/mermaid.py 注入 mermaid.js，
                #   非 streamlit_mermaid 第三方图床）。render_hint 标记可安全渲染，
                #   前端无该标记时退化为代码展示，避免幻觉。
                meta_extra["mermaid_src"] = ext.raw
                meta_extra["render_hint"] = "mermaid:inline"
                summary = None
                try:
                    from note_assistant.indexing.diagrams import MermaidParser
                    dg = MermaidParser.parse(ext.raw, title=ext.context or "")
                    summary = f"Mermaid {dg.diagram_type} 图: {dg.raw_text}"
                    meta_extra["diagram_type"] = dg.diagram_type
                    meta_extra["has_diagram"] = True
                except Exception:
                    graph_type_match = re.match(r'(graph|sequenceDiagram|classDiagram|'
                                                r'stateDiagram|erDiagram|gantt|pie)',
                                                ext.raw, re.IGNORECASE)
                    graph_type = graph_type_match.group(1) if graph_type_match else "图"
                    meta_extra["diagram_type"] = graph_type
                    summary = f"Mermaid {graph_type} 图: {ext.context or '流程图'}"

            elif ext.kind == "image":
                # Image：路径 + alt + 上下文。src/alt 走 ext.meta（结构化），
                # 不再依赖把 alt 拼进 context 字符串的旧反模式。
                # P1-c/P1-d：若注入了 image_enricher（生产环境），则用其产出的
                # 结构化理解 / 原生解析结果富化本 chunk；否则保持默认摘要。
                src = ext.meta.get("src") or ext.raw
                alt = ext.meta.get("alt") or ""
                desc = " | ".join(p for p in (alt, ext.context) if p)
                summary = f"图片: {src} ({desc})" if desc else f"图片: {src}"
                meta_extra = {}
                if self._image_enricher is not None:
                    hp = self._placeholder_heading.get(ext.placeholder, "")
                    enriched = self._image_enricher(ext, hp)
                    if enriched is not None:
                        summary, meta_extra = enriched
                        # 资产定位回写 ext.meta：供 bind_inline_images() 把它绑到
                        # 内联了同一张图的正文 chunk（此刻只有 summary chunk 有 URL）
                        for k in _ASSET_KEYS:
                            if meta_extra.get(k):
                                ext.meta[k] = meta_extra[k]

            elif ext.kind == "code":
                # Code：取语言标记 + 前几行
                lang_match = re.match(r'```(\w+)', ext.raw)
                lang = lang_match.group(1) if lang_match else "代码"
                first_line = ext.raw.strip().split('\n')[1][:80] if '\n' in ext.raw else ext.raw[:80]
                summary = f"{lang} 代码块: {first_line}"

            else:
                summary = ext.raw[:200]

            meta = {
                "kind": ext.kind,
                "source": "extracted_summary",
                "placeholder": ext.placeholder,
            }
            # 章节定位：由 restore() 回填。缺失时留空，build_structural_prefix 会退化到文档级。
            hp = self._placeholder_heading.get(ext.placeholder)
            if hp:
                meta["heading_path"] = hp
            # 图片专用：把源地址带进 metadata，供 API 层组装 img_path / 前端渲染
            if ext.kind == "image":
                src = ext.meta.get("src")
                if src:
                    meta["img_src"] = src
                alt = ext.meta.get("alt")
                if alt:
                    meta["img_alt"] = alt
            # 其它分支注入的结构化 metadata（mermaid 的 render_hint / diagram_type 等）
            if meta_extra:
                meta.update(meta_extra)
            # L0-d 溯源：图片摘要未 enricher 富化时标注兜底来源（vlm/svg 由 enricher 写入，不被覆盖）
            if ext.kind == "image":
                meta.setdefault("trust", "alt_fallback")

            summary_chunks.append(Chunk(
                page_content=summary,
                metadata=meta,
                kind="extracted_summary",
            ))
        return summary_chunks

    def bind_inline_images(self, chunks: list[Chunk]) -> list[Chunk]:
        """把图片资产定位回绑到「正文内联了该图」的 chunk 上（原地修改）。

        背景：资产信息只长在图片 summary chunk 上。正文 chunk 经 restore() 还原出
        ``![alt](src)`` 后，只自带一个裸相对路径——来源面板据此渲染必然 404
        （「图片文件不可见」）。这里把同一张图的 ``/assets/{asset_id}`` 补进它的
        metadata，让面板与答案正文走同一个资产出口。

        调用时机必须在 generate_summaries() **之后**：资产定位是 enricher 在那一步
        才算出的。匹配只能按 ``ext.raw``（原始 markdown 语法）——占位符此刻已被
        restore() 吃掉。已有 img_url 的 chunk（图片 summary 自身）不覆盖。
        """
        images = [e for e in self.extracted
                  if e.kind == "image" and e.meta.get("img_url")]
        if not images:
            return chunks
        for chunk in chunks:
            if chunk.metadata.get("img_url"):
                continue
            content = chunk.page_content or ""
            for ext in images:
                # 一图一绑定：按抽取顺序取第一个命中的图，与 classify_source
                # 的「取正文第一张图」口径保持一致
                if ext.raw and ext.raw in content:
                    for k in _ASSET_KEYS:
                        if ext.meta.get(k):
                            chunk.metadata[k] = ext.meta[k]
                    break
        return chunks

    # ──────────────────────────────────────────────
    # Front Matter → 可检索 chunk
    # ──────────────────────────────────────────────

    def _build_front_matter_chunks(self, front_matter: dict) -> list[Chunk]:
        """
        把 YAML front_matter 中的 tags / aliases 转成可检索 chunk。
        这些内容不在正文里出现，但用户可能通过 tag 搜索。
        """
        chunks = []

        # tags → 一个独立 chunk
        tags = front_matter.get("tags")
        if tags:
            if isinstance(tags, list):
                tags_text = ", ".join(str(t) for t in tags)
            else:
                tags_text = str(tags)
            chunks.append(Chunk(
                page_content=f"Tags: {tags_text}",
                metadata={
                    "kind": "tags",
                    "source": "front_matter",
                },
                kind="front_matter",
            ))

        # aliases → 一个独立 chunk（解决 "RAG" 也能搜到 "Retrieval-Augmented Generation" 的问题）
        aliases = front_matter.get("aliases")
        if aliases:
            if isinstance(aliases, list):
                aliases_text = ", ".join(str(a) for a in aliases)
            else:
                aliases_text = str(aliases)
            chunks.append(Chunk(
                page_content=f"别名: {aliases_text}",
                metadata={
                    "kind": "aliases",
                    "source": "front_matter",
                },
                kind="front_matter",
            ))

        return chunks

    # ──────────────────────────────────────────────
    # 内部实现
    # ──────────────────────────────────────────────

    def _protect_code_fences(self, text: str) -> str:
        """保护 code fence 不被标题切分误伤"""
        def replace_fn(m):
            uid = f"<CODE_UID_{uuid.uuid4().hex[:8]}>"
            self.extracted.append(ExtractedChunk(uid, "code", uid, m.group(0), ""))
            return uid
        # 匹配 ``` ... ``` 块（含语言标记）。
        # 注意：mermaid 围栏由 _extract_mermaid 先行抽取（见 process() 顺序），
        # 这里只保护普通代码围栏，避免把 mermaid 当 code 捕获。
        return re.sub(r'```[\w]*\n.*?```', replace_fn, text, flags=re.DOTALL)

    def _extract_tables(self, text: str) -> str:
        """抽取 Markdown 表格"""
        def replace_fn(m):
            uid = f"[TABLE_UID_{uuid.uuid4().hex[:8]}]"
            raw_table = m.group(0)
            # 尝试提取表格前一行作为 caption（洗掉可能混入的 code 占位符）
            start = m.start()
            preceding = text[max(0, start-200):start].strip().split('\n')[-1] if start > 0 else ""
            self.extracted.append(
                ExtractedChunk(uid, "table", uid, raw_table, strip_placeholders(preceding))
            )
            return uid
        # Markdown 表格正则：至少两行，含分隔符行
        return re.sub(r'(?:^|\n)((?:\|.*\n){2,})', replace_fn, text)

    def _extract_mermaid(self, text: str) -> str:
        """抽取 Mermaid 图"""
        def replace_fn(m):
            uid = f"[MERMAID_UID_{uuid.uuid4().hex[:8]}]"
            start = m.start()
            # 取 mermaid 块前面一行作为上下文（caption），洗掉占位符噪声
            preceding = text[max(0, start-200):start].strip().split('\n')[-1] if start > 0 else ""
            self.extracted.append(
                ExtractedChunk(uid, "mermaid", uid, m.group(0), strip_placeholders(preceding))
            )
            return uid
        return re.sub(r'```mermaid\s*\n(.*?)```', replace_fn, text, flags=re.DOTALL)

    def _extract_images(self, text: str, context_window: int = 80) -> str:
        """
        抽取图片，保留上下文作为描述。

        关键约定：
        1. `raw` 存**完整 markdown 语法**（`![alt](path)` / `![[embed]]`），而不是裸路径——
           否则 restore() 还原出来是一段孤零零的路径字符串，渲染语法永久丢失。
        2. 占位符**独立成 token**（不再拼 `": {alt}"` 尾巴），否则 restore 后会变成
           `![alt](path): alt` 的重复文本。alt 已通过 `meta` 结构化承载。
        3. `meta.note_dir` 记下笔记所在目录——附件 `src` 的相对语义是相对笔记文件，
           下游 enricher 取图必须靠它，否则笔记旁的 `assets/x.svg` 永远解析不到。
        """
        def replace_fn(m):
            uid = f"[IMAGE_UID_{uuid.uuid4().hex[:8]}]"
            embed, alt, link = m.group(1), m.group(2), m.group(3)
            raw_target = embed or link or ""
            # Obsidian 尺寸后缀：![[img.png|300]] / ![[img.png|300x200]]
            src, _, dims = raw_target.partition("|")
            meta = {"src": src.strip(), "alt": (alt or "").strip(), "note_dir": self._note_dir}
            if dims.strip():
                meta["dims"] = dims.strip()
            # 前后 context_window 字符作为上下文；洗掉前序阶段留下的 code/table/mermaid 占位符
            context = text[max(0, m.start() - context_window): m.end() + context_window]
            self.extracted.append(
                ExtractedChunk(uid, "image", uid, m.group(0), strip_placeholders(context), meta=meta)
            )
            return uid
        # 匹配 ![[embed]] 和 ![alt](path)
        return re.sub(r'!\[\[([^\]]+)\]\]|!\[([^\]]*)\]\(([^)]+)\)', replace_fn, text)