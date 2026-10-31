import logging
import time

import ollama
from typing import List

from note_assistant.config import settings

logger = logging.getLogger(__name__)


class OllamaEmbedder:
    """Ollama bge-m3 embedding 封装"""

    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.embed_model
        self._client = ollama.Client(host=self.base_url)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入，返回向量列表（顺序与输入一致）。"""
        if not texts:
            return []
        batch_size = max(1, int(settings.embed_batch_size))
        vectors: List[List[float]] = []
        for start in range(0, len(texts), batch_size):
            vectors.extend(self._embed_batch(texts[start:start + batch_size]))
        return vectors

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """单批嵌入；Ollama 偶发 500（Compute error）时退避重试。"""
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                resp = self._client.embed(model=self.model, input=texts)
                return resp["embeddings"]
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt < 2:
                    logger.warning(
                        "ollama embed failed (attempt %d/3, %d texts): %s",
                        attempt + 1, len(texts), e,
                    )
                    time.sleep(1.5 * (attempt + 1))
        assert last_error is not None
        raise last_error

    def embed_one(self, text: str) -> List[float]:
        """单条快捷方法"""
        return self.embed([text])[0]


if __name__ == "__main__":
    # 快速测试
    e = OllamaEmbedder()
    v = e.embed_one("你好")
    print(f"dim={len(v)}, first3={v[:3]}")
