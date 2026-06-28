import chromadb
from pathlib import Path
from typing import List, Dict, Any

from note_assistant.config import settings
from note_assistant.indexing.embedder import OllamaEmbedder
from note_assistant.indexing.vault_loader import VaultLoader
from note_assistant.indexing.preprocessor import RichPreprocessor
from note_assistant.indexing.splitter import make_splitters, split_v2


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
    def upsert(self, chunks: List[Dict[str, Any]]) -> int:
        """
        chunks: [{"page_content": str, "metadata": dict}, ...]
        统一 upsert，自动生成 ID + embedding。
        """
        if not chunks:
            return 0

        ids = []
        for i, c in enumerate(chunks):
            fp = c["metadata"].get("filepath", "unknown")
            kind = c["metadata"].get("kind", "text")
            ids.append(self._make_id(fp, i, kind))

        embeddings = self.embedder.embed([c["page_content"] for c in chunks])
        metadatas = [c.get("metadata", {}) for c in chunks]
        documents = [c["page_content"] for c in chunks]

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
            # 1. preprocessor 抽取富结构 → cleaned text
            cleaned, _ = preprocessor.process_with_meta(node.raw_md, node.front_matter)

            # 2. header + recursive 切分
            chunks = split_v2(node, hs, cs)

            # 3. restore 还原占位符
            chunks = preprocessor.restore(chunks)

            # 4. 补 wikilinks + filepath metadata（整篇级）
            for c in chunks:
                c["metadata"]["wikilinks"] = node.wikilinks
                c["metadata"]["filepath"] = node.filepath
                c["metadata"]["title"] = node.title
                c["metadata"]["tags"] = node.tags
                c["metadata"]["kind"] = "text"

            # 5. 辅路：富结构 summary chunks
            summary_chunks = preprocessor.generate_summaries()
            for sc in summary_chunks:
                sc["metadata"]["filepath"] = node.filepath
                sc["metadata"]["title"] = node.title

            # 6. 入库
            all_chunks = chunks + summary_chunks
            n = self.upsert(all_chunks)
            total_chunks += n

        return {"files": len(docs), "chunks": total_chunks}


if __name__ == "__main__":
    ing = Ingestor()
    stats = ing.index_vault(wipe=True)
    print(f"✅ 索引完成: {stats["files"]} 篇 → {stats["chunks"]} chunks")
