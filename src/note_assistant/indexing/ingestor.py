import chromadb
from pathlib import Path
from typing import List

from note_assistant.config import settings
from note_assistant.indexing.embedder import OllamaEmbedder
from note_assistant.indexing.types import Chunk
from note_assistant.indexing.vault_loader import VaultLoader
from note_assistant.indexing.preprocessor import RichPreprocessor
from note_assistant.indexing.splitter import make_splitters, split_v2


def build_structural_prefix(node, meta: dict, dir_: str) -> str:
    """
    构造层级结构前缀：目录 › 《文档名》› 标题路径(去掉与文档标题重复的 h1 段)。

    拼到 chunk.page_content 头部后，dense / BM25 / reranker 三处都能感知层级结构
    （机制 A，语义层）。示例：
        AI/Agents › 《Code Agent 架构》› 二、关键设计点

    设计见 docs/层级检索与结构优先设计方案.md。dir_ 显式传入（fm/summary chunk 的
    metadata 里没有 dir），title 取 node.title（fm/summary chunk 的 metadata 可能缺 title），
    因此本函数不依赖 meta 一定含这些字段。
    """
    title = node.title
    hp = (meta.get("heading_path") or "")
    hp_segs = [s.strip() for s in hp.split(" > ") if s.strip()]
    # heading_path 的 h1 段通常等于文档标题，去掉避免 "《标题》› 标题 > ..." 冗余
    if hp_segs and hp_segs[0] == title:
        hp_segs = hp_segs[1:]
    parts = []
    if dir_:
        parts.append(dir_)
    parts.append(f"《{title}》")
    if hp_segs:
        parts.append(" > ".join(hp_segs))
    return " › ".join(parts)


class Ingestor:
    """入库 ChromaDB，支持两路 chunk + 增量更新"""

    def __init__(self, persist_dir: str | Path | None = None):
        self.persist_dir = Path(persist_dir) if persist_dir else settings.chroma_persist_dir.resolve()
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = OllamaEmbedder()

    # ──────────────────────────────────────────────
    # ID 构建：filepath::index::kind
    # ──────────────────────────────────────────────
    @staticmethod
    def _make_id(filepath: str, index: int, kind: str) -> str:
        safe_fp = filepath.replace("/", "_").replace("\\", "_")
        return f"{safe_fp}::{index}::{kind}"

    # ──────────────────────────────────────────────
    # 增量：先删旧 chunk，再写新 chunk
    # ──────────────────────────────────────────────
    def upsert(self, chunks: List[Chunk]) -> int:
        """
        chunks: List[Chunk]
        统一 upsert，自动生成 ID + embedding。
        """
        if not chunks:
            return 0

        ids = []
        for i, c in enumerate(chunks):
            fp = c.metadata.get("filepath", "unknown")
            ids.append(self._make_id(fp, i, c.kind))

        # ChromaDB metadata 校验：空列表不接受；list 元素必须同类型且为 str/int/float/bool。
        # front matter 的 tags 可能混入整数（如年份 2026），统一转 str 以满足约束。
        metadatas = []
        for c in chunks:
            clean = {}
            for k, v in c.metadata.items():
                if isinstance(v, list):
                    if len(v) == 0:
                        continue  # ChromaDB 不接受空列表作为 metadata 值
                    clean[k] = [str(x) for x in v]
                else:
                    clean[k] = v
            metadatas.append(clean)

        embeddings = self.embedder.embed([c.page_content for c in chunks])
        documents = [c.page_content for c in chunks]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(ids)

    def delete_all(self) -> None:
        """清空 collection（全量重建时用）"""
        self.client.delete_collection(settings.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ──────────────────────────────────────────────
    # 全量索引入口
    # ──────────────────────────────────────────────
    def index_vault(self, vault_path=None, wipe: bool = True) -> dict:
        """
        一键全量索引：
        wipe=True: 清空后重建（默认）
        wipe=False: 只 upsert（追加，同名 ID 会覆盖）
        返回统计
        """
        if wipe:
            self.delete_all()

        loader = VaultLoader(vault_path)
        docs = loader.load_all()
        if not docs:
            return {"files": 0, "chunks": 0}

        hs, cs = make_splitters()
        preprocessor = RichPreprocessor()
        total_chunks = 0

        for node in docs:
            # 0. 计算相对目录（层级结构前缀用，vault 内相对于根的路径父目录）
            dir_ = str(Path(node.filepath).parent)
            if dir_ == ".":
                dir_ = ""

            # 1. preprocessor 抽取富结构 → cleaned text + front_matter chunks
            cleaned, fm_chunks = preprocessor.process_with_meta(node)

            # 2. header + recursive 切分
            chunks = split_v2(node, hs, cs)

            # 3. restore 还原占位符
            chunks = preprocessor.restore(chunks)

            # 4. 补 wikilinks + filepath + dir metadata，并拼结构前缀（机制 A：语义层感知层级）
            for c in chunks:
                c.metadata["wikilinks"] = node.wikilinks
                c.metadata["filepath"] = node.filepath
                c.metadata["title"] = node.title
                if dir_:
                    c.metadata["dir"] = dir_
                # ChromaDB 不接受空列表作为 metadata 值，tags 为空则不设置
                if node.tags:
                    c.metadata["tags"] = node.tags
                prefix = build_structural_prefix(node, c.metadata, dir_)
                c.page_content = f"{prefix}\n\n{c.page_content}"

            # 5. 辅路：富结构 summary chunks（补 metadata + 前缀）
            summary_chunks = preprocessor.generate_summaries()
            for sc in summary_chunks:
                sc.metadata["filepath"] = node.filepath
                sc.metadata["title"] = node.title
                if dir_:
                    sc.metadata["dir"] = dir_
                prefix = build_structural_prefix(node, sc.metadata, dir_)
                sc.page_content = f"{prefix}\n\n{sc.page_content}"

            # 5b. front_matter chunks：此前漏补 metadata，这里补 filepath/title/dir + 前缀
            for fc in fm_chunks:
                fc.metadata["filepath"] = node.filepath
                fc.metadata["title"] = node.title
                if dir_:
                    fc.metadata["dir"] = dir_
                prefix = build_structural_prefix(node, fc.metadata, dir_)
                fc.page_content = f"{prefix}\n\n{fc.page_content}"

            # 6. 入库
            all_chunks = chunks + summary_chunks + fm_chunks
            n = self.upsert(all_chunks)
            total_chunks += n

        return {"files": len(docs), "chunks": total_chunks}


if __name__ == "__main__":
    ing = Ingestor()
    stats = ing.index_vault(wipe=True)
    print(f"✅ 索引完成: {stats['files']} 篇 → {stats['chunks']} chunks")
