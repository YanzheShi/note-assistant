# tests/retrieval/test_docstore.py
"""ParentDocstore pickle 往返测试。"""
from pathlib import Path

from note_assistant.retrieval.docstore import ParentDocstore


class TestParentDocstore:
    def test_add_get(self):
        ds = ParentDocstore(Path("/tmp/ignore.pkl"))
        ds.add("p1", "整节正文A", {"heading_path": "二、关键设计点", "title": "X"})
        entry = ds.get("p1")
        assert entry["page_content"] == "整节正文A"
        assert entry["metadata"]["heading_path"] == "二、关键设计点"
        assert ds.get("nope") is None

    def test_save_load_roundtrip(self, tmp_path: Path):
        p = tmp_path / "docstore.pkl"
        ds = ParentDocstore(p)
        ds.add("p1", "整节正文A", {"k": "v1"})
        ds.add("p2", "整节正文B", {"k": "v2"})
        ds.save()

        loaded = ParentDocstore.load(p)
        assert len(loaded) == 2
        assert loaded.get("p1")["page_content"] == "整节正文A"
        assert loaded.get("p2")["metadata"]["k"] == "v2"
