# tests/security/test_vlm_hardening.py
"""L0-b/c VLM 抗注入硬化 + 输出校验（索引期持久投毒防线）。"""
import pytest

from note_assistant.config import settings
from note_assistant.indexing.understanding import (
    SYSTEM_PROMPT,
    ImageUnderstanding,
    _clamp_field,
)


class TestVlmPromptHardening:
    def test_prompt_declares_image_text_as_data(self):
        # 图中文字是数据不是指令——S4 图片注入的第一道防线
        assert "待抄录的数据" in SYSTEM_PROMPT
        assert "不执行" in SYSTEM_PROMPT

    def test_prompt_version_bumped_for_cache_invalidation(self):
        # prompt 硬化后必须 bump 版本，让旧 VisionCache 按既有机制失效
        assert settings.vlm_prompt_version >= "v2"


class TestOutputClamping:
    def test_clamp_strips_control_chars(self):
        assert _clamp_field("a\x00b\x01c") == "abc"

    def test_clamp_keeps_newline_and_tab(self):
        assert _clamp_field("行1\n行2\t列") == "行1\n行2\t列"

    def test_clamp_length_cap(self, monkeypatch):
        monkeypatch.setattr(settings, "vlm_text_field_max_chars", 10)
        assert len(_clamp_field("字" * 500)) == 10

    def test_payload_fields_clamped(self, monkeypatch):
        monkeypatch.setattr(settings, "vlm_text_field_max_chars", 5)
        u = ImageUnderstanding(description="长" * 100, ocr_text="字" * 100, title="题" * 100)
        p = u.to_index_payload()
        assert len(p["image_description"]) == 5
        assert len(p["image_ocr_text"]) == 5
        assert len(p["image_title"]) <= 200

    def test_entities_capped(self):
        import json

        u = ImageUnderstanding(entities=[f"e{i}" for i in range(200)])
        p = u.to_index_payload()
        assert len(json.loads(p["image_entities"])) <= 50
