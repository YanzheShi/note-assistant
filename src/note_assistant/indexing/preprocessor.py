import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

from note_assistant.indexing.types import ExtractedChunk


class RichPreprocessor:
    def __init__(self):
        self.extracted: list[ExtractedChunk] = []

    # ──────────────────────────────────────────────
    # 主入口
    # ──────────────────────────────────────────────

    def process(self, md_text: str) -> str:
        """将富结构替换为占位符，返回清洗后的文本"""
        self.extracted = []
        text = md_text

        # P1: code fence 保护（必须在最前面，防 MarkdownHeaderTextSplitter 误解析）
        text = self._protect_code_fences(text)

        # P2: 抽取表格
        text = self._extract_tables(text)

        # P3: 抽取 Mermaid
        text = self._extract_mermaid(text)

        # P4: 抽取图片
        text = self._extract_images(text, context_window=80)

        return text

    def process_with_meta(self, md_text: str, front_matter: dict) -> tuple[str, list[dict]]:
        """
        处理文本 + 从 front_matter 生成额外可检索 chunk。
        调用方应把 front_matter 从 VaultLoader 传进来。

        返回: (cleaned_text, extra_chunks)
            extra_chunks: list[{"page_content": str, "metadata": dict}]
        """
        cleaned = self.process(md_text)
        extra_chunks = self._build_front_matter_chunks(front_matter)
        return cleaned, extra_chunks

    # ──────────────────────────────────────────────
    # 还原（切分后调用）
    # ──────────────────────────────────────────────

    def restore(self, chunks: list[dict]) -> list[dict]:
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
            content = chunk["page_content"]
            metadata = dict(chunk.get("metadata", {}))

            # 找出这个 chunk 里出现了哪些占位符
            found = self._find_placeholders(content)

            if found:
                # metadata 标记含哪些富结构
                metadata["has_code"] = any(e.kind == "code" for e in found)
                metadata["has_table"] = any(e.kind == "table" for e in found)
                metadata["has_mermaid"] = any(e.kind == "mermaid" for e in found)
                metadata["has_image"] = any(e.kind == "image" for e in found)

                # 还原占位符 → 原始内容
                for ext in found:
                    content = content.replace(ext.placeholder, ext.raw)

            restored.append({
                "page_content": content,
                "metadata": metadata,
            })
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

    def generate_summaries(self) -> list[dict]:
        """
        为抽取的富结构生成可检索的文本描述，作为独立 chunk 入库。

        问题：表格/mermaid/图片被抽走后，它们的内容在检索空间里就消失了。
        解决：给每个 ExtractedChunk 生成一段自然语言描述，单独嵌入一个 "summary chunk"，
              metadata 标记 source="extracted" + 对应 placeholder，
              命中后 restore 阶段可以还原原始内容。

        返回: list[{"page_content": str, "metadata": dict}]
        """
        summary_chunks = []
        for ext in self.extracted:
            if ext.kind == "table":
                # 表格：caption 取前一行 + 列数信息
                col_count = max(ext.raw.count('|') - 1, 1)
                first_row = ext.raw.strip().split('\n')[0].strip('|').strip()
                summary = f"表格（约{col_count}列）: {ext.context or first_row}"

            elif ext.kind == "mermaid":
                # Mermaid：取图类型 + 前一行 caption
                graph_type_match = re.match(r'(graph|sequenceDiagram|classDiagram|'
                                            r'stateDiagram|erDiagram|gantt|pie)',
                                            ext.raw, re.IGNORECASE)
                graph_type = graph_type_match.group(1) if graph_type_match else "图"
                summary = f"Mermaid {graph_type} 图: {ext.context or '流程图'}"

            elif ext.kind == "image":
                # Image：保留路径 + alt + 上下文
                summary = f"图片: {ext.raw} ({ext.context})"

            elif ext.kind == "code":
                # Code：取语言标记 + 前几行
                lang_match = re.match(r'```(\w+)', ext.raw)
                lang = lang_match.group(1) if lang_match else "代码"
                first_line = ext.raw.strip().split('\n')[1][:80] if '\n' in ext.raw else ext.raw[:80]
                summary = f"{lang} 代码块: {first_line}"

            else:
                summary = ext.raw[:200]

            summary_chunks.append({
                "page_content": summary,
                "metadata": {
                    "kind": ext.kind,
                    "source": "extracted_summary",
                    "placeholder": ext.placeholder,
                }
            })
        return summary_chunks

    # ──────────────────────────────────────────────
    # Front Matter → 可检索 chunk
    # ──────────────────────────────────────────────

    def _build_front_matter_chunks(self, front_matter: dict) -> list[dict]:
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
            chunks.append({
                "page_content": f"Tags: {tags_text}",
                "metadata": {
                    "kind": "tags",
                    "source": "front_matter",
                }
            })

        # aliases → 一个独立 chunk（解决 "RAG" 也能搜到 "Retrieval-Augmented Generation" 的问题）
        aliases = front_matter.get("aliases")
        if aliases:
            if isinstance(aliases, list):
                aliases_text = ", ".join(str(a) for a in aliases)
            else:
                aliases_text = str(aliases)
            chunks.append({
                "page_content": f"别名: {aliases_text}",
                "metadata": {
                    "kind": "aliases",
                    "source": "front_matter",
                }
            })

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
        # 匹配 ``` ... ``` 块（含语言标记）
        return re.sub(r'```[\w]*\n.*?```', replace_fn, text, flags=re.DOTALL)

    def _extract_tables(self, text: str) -> str:
        """抽取 Markdown 表格"""
        def replace_fn(m):
            uid = f"[TABLE_UID_{uuid.uuid4().hex[:8]}]"
            raw_table = m.group(0)
            # 尝试提取表格前一行作为 caption
            start = m.start()
            preceding = text[max(0, start-200):start].strip().split('\n')[-1] if start > 0 else ""
            self.extracted.append(ExtractedChunk(uid, "table", uid, raw_table, preceding))
            return uid
        # Markdown 表格正则：至少两行，含分隔符行
        return re.sub(r'(?:^|\n)((?:\|.*\n){2,})', replace_fn, text)

    def _extract_mermaid(self, text: str) -> str:
        """抽取 Mermaid 图"""
        def replace_fn(m):
            uid = f"[MERMAID_UID_{uuid.uuid4().hex[:8]}]"
            start = m.start()
            # 取 mermaid 块前面一行作为上下文（caption）
            preceding = text[max(0, start-200):start].strip().split('\n')[-1] if start > 0 else ""
            self.extracted.append(ExtractedChunk(uid, "mermaid", uid, m.group(0), preceding))
            return uid
        return re.sub(r'```mermaid\s*\n(.*?)```', replace_fn, text, flags=re.DOTALL)

    def _extract_images(self, text: str, context_window: int = 80) -> str:
        """抽取图片，保留上下文作为描述"""
        def replace_fn(m):
            uid = f"[IMAGE_UID_{uuid.uuid4().hex[:8]}]"
            img_path = m.group(1) or m.group(3)  # ![[img]] 或 ![alt](path)
            alt = m.group(2) or ""
            # 取前后 context_window 字符作为上下文
            start = m.start()
            end = m.end()
            context = text[max(0, start-context_window):end+context_window]
            self.extracted.append(ExtractedChunk(uid, "image", uid, img_path, alt + " | " + context))
            return f"{uid}: {alt or img_path}"
        # 匹配 ![[embed]] 和 ![alt](path)
        return re.sub(r'!\[\[([^\]]+)\]\]|!\[([^\]]*)\]\(([^)]+)\)', replace_fn, text)