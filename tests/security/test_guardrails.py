# tests/security/test_guardrails.py
"""L1 提示词硬化：护栏条款追加 + 数据边界包裹 + 开关零回归。"""
import pytest

from note_assistant.config import settings
from note_assistant.security import guardrails as g


class TestGuardrailSwitch:
    def test_append_guardrail_enabled(self, monkeypatch):
        monkeypatch.setattr(settings, "security_guardrail_enabled", True)
        out = g.append_guardrail("你是助手。")
        assert out.startswith("你是助手。")
        assert "安全规则（最高优先级）" in out
        assert "不可信的外部数据" in out

    def test_disabled_all_passthrough(self, monkeypatch):
        """开关关闭：所有函数逐字节回退（零回归约定）。"""
        monkeypatch.setattr(settings, "security_guardrail_enabled", False)
        assert g.append_guardrail("你是助手。") == "你是助手。"
        assert g.wrap_retrieved_context("ctx") == "ctx"
        assert g.wrap_user_question("q") == "q"
        assert g.wrap_tool_result("t", "x") == "x"
        assert g.wrap_history_tuples([("human", "a")]) == [("human", "a")]


class TestWrapping:
    @pytest.fixture(autouse=True)
    def _enabled(self, monkeypatch):
        monkeypatch.setattr(settings, "security_guardrail_enabled", True)

    def test_wrap_retrieved_context(self):
        out = g.wrap_retrieved_context("笔记内容")
        assert out.startswith("<retrieved_context>")
        assert out.endswith("</retrieved_context>")
        assert "笔记内容" in out

    def test_wrap_user_question(self):
        assert g.wrap_user_question("架构图长什么样") == "<user_question>架构图长什么样</user_question>"

    def test_wrap_tool_result_with_name(self):
        out = g.wrap_tool_result("get_note", "内容")
        assert out.startswith('<tool_result name="get_note">')
        assert out.endswith("</tool_result>")

    def test_wrap_history_tuples_marks_first_and_last(self):
        hist = [("human", "q1"), ("ai", "a1"), ("human", "q2")]
        out = g.wrap_history_tuples(hist)
        assert out[0][1].startswith("<conversation_history>")
        assert out[-1][1].endswith("</conversation_history>")
        assert "<conversation_history>" not in out[1][1]
        assert hist[0][1] == "q1"  # 原序列不被修改

    def test_wrap_history_tuples_single_turn(self):
        out = g.wrap_history_tuples([("human", "q")])
        assert out[0][1].startswith("<conversation_history>")
        assert out[0][1].endswith("</conversation_history>")

    def test_wrap_history_messages_skips_system(self):
        pytest.importorskip("langchain_core")
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        msgs = [SystemMessage(content="长程摘要"), HumanMessage(content="q1"), AIMessage(content="a1")]
        out = g.wrap_history_messages(msgs)
        assert out[0].content == "长程摘要"  # SystemMessage 不包
        assert out[1].content.startswith("<conversation_history>")
        assert out[2].content.endswith("</conversation_history>")
        assert msgs[1].content == "q1"  # 原对象不被修改
