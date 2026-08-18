# tests/retrieval/test_types.py
"""RetrievalResult.identity_key：累积去重键语义测试。

锁定「图片与同节文本互斥」修复：旧键 (filepath, heading) 让 image summary chunk
与同章节父块/正文 chunk 互斥，谁分数高谁进 accumulated（分数竞速），
是「有的 query 图进 sources、有的不进」的直接原因。新键加入 kind + placeholder，
富结构 chunk 各自独立；正文 vs 正文行为与原先逐字节一致。
"""
from note_assistant.retrieval.types import RetrievalResult


def _r(metadata: dict, content: str = "x", score: float = 0.5) -> RetrievalResult:
    return RetrievalResult(score=score, page_content=content, metadata=metadata)


class TestIdentityKey:
    def test_text_vs_text_same_heading_deduped(self):
        # 普通正文 chunk：kind/placeholder 均空，等价旧键，重复仍去重
        a = _r({"filepath": "n.md", "heading_path": "H1"})
        b = _r({"filepath": "n.md", "heading_path": "H1"}, content="另一段正文")
        assert a.identity_key() == b.identity_key()

    def test_image_vs_text_same_heading_distinct(self):
        img = _r({"filepath": "n.md", "heading_path": "H1", "kind": "image",
                  "placeholder": "[IMAGE_UID_a1b2c3d4]"})
        txt = _r({"filepath": "n.md", "heading_path": "H1"})
        assert img.identity_key() != txt.identity_key()

    def test_parent_vs_image_same_heading_distinct(self):
        # v2b 父块（kind=parent）与图片摘要块同章节不再互斥
        parent = _r({"filepath": "n.md", "heading_path": "H1", "kind": "parent"})
        img = _r({"filepath": "n.md", "heading_path": "H1", "kind": "image",
                  "asset_id": "abc123def4567890"})
        assert parent.identity_key() != img.identity_key()

    def test_two_images_same_heading_distinct(self):
        # 同节两张图也不互斥（placeholder 为抽取期 uuid，天然唯一）
        a = _r({"filepath": "n.md", "heading_path": "H1", "kind": "image",
                "placeholder": "[IMAGE_UID_aaaaaaaa]"})
        b = _r({"filepath": "n.md", "heading_path": "H1", "kind": "image",
                "placeholder": "[IMAGE_UID_bbbbbbbb]"})
        assert a.identity_key() != b.identity_key()

    def test_same_placeholder_same_identity(self):
        # 同一 chunk 重复出现（多轮检索命中）仍应去重
        a = _r({"filepath": "n.md", "heading_path": "H1", "kind": "image",
                "placeholder": "[IMAGE_UID_aaaaaaaa]"}, score=0.9)
        b = _r({"filepath": "n.md", "heading_path": "H1", "kind": "image",
                "placeholder": "[IMAGE_UID_aaaaaaaa]"}, score=0.1)
        assert a.identity_key() == b.identity_key()

    def test_image_without_placeholder_falls_back_to_asset_id(self):
        # 无 placeholder 时退化用 asset_id 区分（同图去重、异图共存）
        a = _r({"filepath": "n.md", "heading_path": "H1", "kind": "image",
                "asset_id": "abc123def4567890"})
        b = _r({"filepath": "n.md", "heading_path": "H1", "kind": "image",
                "asset_id": "feedfacefeedface"})
        assert a.identity_key() != b.identity_key()
