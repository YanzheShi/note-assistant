"""启动前切分策略切换（chunking_strategy）的 smoke 测试。

验证点：
- config 默认策略为 v2；
- v2 切分产出 heading_path（结构检索完整生效）；
- v1 切分不产出 heading_path（结构检索自动退化，见 docs 第十二节）。

不触发 embedder / Ollama，纯单元。
"""
import pytest

from pathlib import Path

from note_assistant.config import settings
from note_assistant.indexing.splitter import make_splitters, split_v1, split_v2
from note_assistant.indexing.types import DocNode


def _mk_node(md: str, title: str = "T", fp: str = "AI/x.md") -> DocNode:
    return DocNode(
        filepath=fp,
        abs_path=Path("/tmp/x.md"),
        raw_md=md,
        front_matter={},
        title=title,
        tags=[],
        wikilinks=[],
    )


SAMPLE = (
    "# 一、背景\n"
    "背景正文。\n\n"
    "## 二、关键设计点\n"
    "关键设计点正文。\n"
)


def test_default_strategy_is_v2():
    assert settings.chunking_strategy == "v2"


def test_v2_produces_heading_path():
    hs, cs = make_splitters()
    node = _mk_node(SAMPLE, title="Code Agent 架构")
    chunks = split_v2(node, hs, cs)
    assert chunks, "v2 应产出 chunk"
    assert all("heading_path" in c.metadata for c in chunks)
    assert any(
        "二、关键设计点" in c.metadata["heading_path"] for c in chunks
    ), "v2 应保留分级标题路径"


def test_v1_has_no_heading_path():
    hs, cs = make_splitters()
    node = _mk_node(SAMPLE, title="Code Agent 架构")
    chunks = split_v1(node, cs)
    assert chunks, "v1 应产出 chunk"
    assert all(
        "heading_path" not in c.metadata for c in chunks
    ), "v1 扁平切分不应有 heading_path（结构检索因此退化）"
