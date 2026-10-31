# tests/indexing/test_embedder.py
"""OllamaEmbedder：分批顺序 + 瞬时失败重试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from note_assistant.config import settings
from note_assistant.indexing.embedder import OllamaEmbedder


class _FakeClient:
    def __init__(self, failures: int = 0):
        self.failures = failures
        self.calls = []

    def embed(self, model, input):
        self.calls.append(list(input))
        if self.failures:
            self.failures -= 1
            raise RuntimeError("Compute error")
        return {"embeddings": [[float(len(t))] for t in input]}


def test_embed_batches_and_preserves_order(monkeypatch):
    monkeypatch.setattr(settings, "embed_batch_size", 3)
    embedder = OllamaEmbedder()
    embedder._client = _FakeClient()

    result = embedder.embed(["a", "bb", "ccc", "dddd"])

    assert [len(c) for c in embedder._client.calls] == [3, 1]
    assert result == [[1.0], [2.0], [3.0], [4.0]]


def test_embed_retries_transient_failure(monkeypatch):
    monkeypatch.setattr(settings, "embed_batch_size", 8)
    monkeypatch.setattr("note_assistant.indexing.embedder.time.sleep", lambda _: None)
    embedder = OllamaEmbedder()
    embedder._client = _FakeClient(failures=1)

    assert embedder.embed(["x"]) == [[1.0]]
    assert len(embedder._client.calls) == 2  # 第一次失败，第二次成功
