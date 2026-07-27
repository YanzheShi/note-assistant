import ollama
from typing import List

from note_assistant.config import settings


class OllamaEmbedder:
    """Ollama bge-m3 embedding 封装"""

    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.embed_model
        self._client = ollama.Client(host=self.base_url)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入，返回向量列表（每个长度=embed_dim）"""
        if not texts:
            return []
        resp = self._client.embed(model=self.model, input=texts)
        return resp["embeddings"]

    def embed_one(self, text: str) -> List[float]:
        """单条快捷方法"""
        return self.embed([text])[0]


if __name__ == "__main__":
    # 快速测试
    e = OllamaEmbedder()
    v = e.embed_one("你好")
    print(f"dim={len(v)}, first3={v[:3]}")
