# tests/pipeline/test_image_neighbor_expand.py
"""#4 图片邻居扩展（设计 7.3）跨链路测试：

- 共享 expand_image_neighbors（image_answer）对「真实 VLM 图」也能检测：
  其 metadata 无 kind=="image"，而是 asset_id + image_description/ocr_text。
  旧的 str(meta.get("kind")) == "image" 判定会漏掉它们，导致生产环境邻居扩展永不触发。
- Agent 链路 generate_node 命中 image chunk 时把同章节文本邻居带进生成上下文，
  且不回写 state["accumulated"]（不污染 sources/Judge 证据，与 rag_chain 对齐）。
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from note_assistant.agent import agent as agent_mod
from note_assistant.agent.agent import generate_node
from note_assistant.config import settings
from note_assistant.pipeline.image_answer import _is_image_chunk, expand_image_neighbors
from note_assistant.retrieval.types import RetrievalResult


def _r(content, metadata):
    return RetrievalResult(score=0.9, page_content=content, metadata=metadata)


# ── 真实 VLM 图 chunk（无 kind=="image"，靠 asset_id + image_description 判定）──
def _vlm_image_chunk(heading="H1"):
    return _r(
        "图片理解：三层架构…",
        {
            "asset_id": "abc123def4567890",
            "image_description": "三层架构图",
            "image_ocr_text": "表示层/业务层/数据层",
            "heading_path": heading,
            "filepath": "n.md",
            "title": "架构",
        },
    )


def _legacy_image_chunk(heading="H1"):
    return _r(
        "图片: arch.png",
        {"kind": "image", "asset_id": "abc123def4567890", "heading_path": heading, "filepath": "n.md"},
    )


def _text_chunk(heading, content="正文"):
    return _r(content, {"kind": "text", "heading_path": heading, "filepath": "n.md"})


# ───────────────────────── 共享 helper 单测 ─────────────────────────
class TestExpandImageNeighbors:
    def test_vlm_image_chunk_detected(self, monkeypatch):
        """真实 VLM 图（无 kind）也能触发扩展 —— 这是 #4 要修的 bug 的回归测试。"""
        monkeypatch.setattr("note_assistant.config.settings.image_neighbor_expand", True)
        img = _vlm_image_chunk("H1")
        neighbor = _text_chunk("H1", "邻居正文")

        def fetch(paths):
            assert paths == ["H1"]
            return [neighbor]

        added = expand_image_neighbors([img], fetch)
        assert len(added) == 1
        assert added[0].page_content == "邻居正文"

    def test_legacy_kind_image_chunk_still_detected(self, monkeypatch):
        monkeypatch.setattr("note_assistant.config.settings.image_neighbor_expand", True)
        img = _legacy_image_chunk("H1")
        neighbor = _text_chunk("H1", "邻居正文")

        def fetch(paths):
            return [neighbor]

        added = expand_image_neighbors([img], fetch)
        assert len(added) == 1

    def test_no_image_no_expand(self, monkeypatch):
        monkeypatch.setattr("note_assistant.config.settings.image_neighbor_expand", True)
        called = []
        added = expand_image_neighbors([_text_chunk("H1")], lambda p: called.append(p) or [])
        assert added == []
        assert called == []  # fetch 根本不该被调用

    def test_disabled_returns_empty(self, monkeypatch):
        monkeypatch.setattr("note_assistant.config.settings.image_neighbor_expand", False)
        added = expand_image_neighbors([_vlm_image_chunk("H1")], lambda p: [_text_chunk("H1")])
        assert added == []

    def test_only_same_heading_and_skips_self_image(self, monkeypatch):
        monkeypatch.setattr("note_assistant.config.settings.image_neighbor_expand", True)
        img = _vlm_image_chunk("H1")
        # 同章节正文带出；不同章节正文不带；自身图（同 heading）跳过
        neighbors = [
            _text_chunk("H1", "同章节正文"),
            _text_chunk("H2", "异章节正文"),
            _vlm_image_chunk("H1"),  # 自身图，应跳过
        ]

        def fetch(paths):
            return neighbors

        added = expand_image_neighbors([img], fetch)
        assert [a.page_content for a in added] == ["同章节正文"]

    def test_budget_honored(self, monkeypatch):
        monkeypatch.setattr("note_assistant.config.settings.image_neighbor_expand", True)
        img = _vlm_image_chunk("H1")
        neighbors = [_text_chunk("H1", f"t{i}") for i in range(10)]

        def fetch(paths):
            return neighbors

        added = expand_image_neighbors([img], fetch, budget=3)
        assert len(added) == 3


# ───────────────────────── Agent 链路集成 ─────────────────────────
_LAST_MESSAGES = []


class _FakeLLM(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        _LAST_MESSAGES[:] = list(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="回答"))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        _LAST_MESSAGES[:] = list(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="回答"))])


def _agent_state(accumulated, **over):
    state = {
        "question": "架构图长什么样",
        "accumulated": accumulated,
        "iteration": 0,
        "history": [],
        "judge_verdict": None,
        "widen_context": False,
        "gate_overrode": False,
    }
    state.update(over)
    return state


class TestAgentGenerateExpandsNeighbors:
    def test_neighbor_enters_generation_context(self, monkeypatch):
        """命中 image chunk 时，同章节文本邻居进入生成上下文（#4 核心）。"""
        monkeypatch.setattr(agent_mod, "get_llm", lambda *a, **k: _FakeLLM())
        monkeypatch.setattr(
            "note_assistant.config.settings.image_neighbor_expand", True
        )
        neighbor = _text_chunk("H1", "邻居正文内容")
        monkeypatch.setattr(
            agent_mod, "_fetch_text_neighbors_by_heading",
            lambda paths: ([neighbor] if paths == ["H1"] else []),
        )

        accumulated = [_vlm_image_chunk("H1"), _text_chunk("H2", "其他正文")]
        _LAST_MESSAGES.clear()
        import asyncio

        asyncio.run(generate_node(_agent_state(accumulated)))

        human = _LAST_MESSAGES[-1]
        assert "邻居正文内容" in str(human.content)
        # 图片 chunk 的结构化块也应被渲染进上下文
        assert "三层架构" in str(human.content)

    def test_no_image_no_extra_fetch(self, monkeypatch):
        """无图片时，fetch 不被调用、上下文不含邻居。"""
        monkeypatch.setattr(agent_mod, "get_llm", lambda *a, **k: _FakeLLM())
        monkeypatch.setattr(
            "note_assistant.config.settings.image_neighbor_expand", True
        )
        fetched = []
        monkeypatch.setattr(
            agent_mod, "_fetch_text_neighbors_by_heading",
            lambda paths: (fetched.append(paths) or [_text_chunk("H1", "X")]),
        )

        accumulated = [_text_chunk("H1", "纯文本")]
        _LAST_MESSAGES.clear()
        import asyncio

        asyncio.run(generate_node(_agent_state(accumulated)))

        assert fetched == []  # 无图片 → fetch 根本不触发
        assert "X" not in str(_LAST_MESSAGES[-1].content)

    def test_does_not_pollute_accumulated(self, monkeypatch):
        """邻居只进上下文，不回写 state['accumulated']（不污染 sources/Judge 证据）。"""
        monkeypatch.setattr(agent_mod, "get_llm", lambda *a, **k: _FakeLLM())
        monkeypatch.setattr(
            "note_assistant.config.settings.image_neighbor_expand", True
        )
        neighbor = _text_chunk("H1", "邻居正文内容")
        monkeypatch.setattr(
            agent_mod, "_fetch_text_neighbors_by_heading",
            lambda paths: [neighbor] if paths == ["H1"] else [],
        )

        accumulated = [_vlm_image_chunk("H1")]
        _LAST_MESSAGES.clear()
        import asyncio

        state = _agent_state(accumulated)
        asyncio.run(generate_node(state))

        # accumulated 仍是原样（只有 image chunk），邻居未回写
        assert len(state["accumulated"]) == 1
        assert "邻居正文内容" not in state["accumulated"][0].page_content
