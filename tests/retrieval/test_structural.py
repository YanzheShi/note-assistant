# tests/retrieval/test_structural.py
"""结构分（structural_score）单元测试。

验证：query 与 chunk 结构元数据（dir/title/heading_path）的重叠度计算、
标题精确命中（title_hit）的硬兜底、归一化与无信号场景。
见 docs/层级检索与结构优先设计方案.md。
"""
from note_assistant.retrieval.structural import structural_score, _normalize


class TestStructuralScore:
    def test_empty_query(self):
        score, hit = structural_score("", {"title": "A"})
        assert score == 0.0
        assert hit is False

    def test_title_exact_hit(self):
        meta = {
            "title": "Code Agent 架构",
            "heading_path": "一、背景 > 二、关键设计点",
            "dir": "AI/Agents",
        }
        score, hit = structural_score("Code Agent 架构的关键设计点是什么", meta)
        # 用户原场景：query 完整含文档标题 → 硬兜底 title_hit
        assert hit is True

    def test_title_hit_with_booktitle_punct(self):
        # 归一化应去掉《》等标点，仍能命中
        meta = {"title": "Code Agent 架构"}
        score, hit = structural_score("《Code Agent 架构》是什么", meta)
        assert hit is True

    def test_heading_overlap_no_title(self):
        meta = {
            "title": "Code Agent 架构",
            "heading_path": "一、背景 > 二、关键设计点",
            "dir": "AI/Agents",
        }
        score, hit = structural_score("关键设计点", meta)
        # 只命中 heading，不含完整标题 → 无 title_hit，但有重叠分
        assert hit is False
        assert score > 0.0

    def test_dir_only_low_score(self):
        meta = {"title": "X", "heading_path": "", "dir": "AI/Agents"}
        score, hit = structural_score("AI/Agents", meta)
        # dir 权重最低，且低于软阈值不应触发硬兜底
        assert hit is False
        assert score < 0.5

    def test_no_signal(self):
        meta = {
            "title": "Code Agent 架构",
            "heading_path": "二、关键设计点",
            "dir": "AI/Agents",
        }
        score, hit = structural_score("今天天气真好", meta)
        assert score == 0.0
        assert hit is False


class TestNormalize:
    def test_strips_booktitle_punct(self):
        assert _normalize("《Code Agent 架构》") == "CodeAgent架构"

    def test_strips_whitespace(self):
        assert _normalize("Code Agent 架构") == "CodeAgent架构"
