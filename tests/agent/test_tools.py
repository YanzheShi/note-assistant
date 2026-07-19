"""工具集测试（P6 扩展 + P7a 重试/降级）。

覆盖：
- HybridRetriever._build_where 过滤条件构造（纯函数，无外部服务）
- run_tool_call 重试与降级：hybrid 失败 → vector；全部失败 → 优雅兜底
"""
from note_assistant.agent.tools import run_tool_call
from note_assistant.retrieval.hybrid import HybridRetriever


def test_build_where_single_filepath():
    assert HybridRetriever._build_where(filepath="a.md") == {"filepath": "a.md"}


def test_build_where_heading_contains():
    assert HybridRetriever._build_where(heading="背景") == {"heading_path": {"$contains": "背景"}}


def test_build_where_multi_and():
    w = HybridRetriever._build_where(filepath="a.md", heading="背景", tag="ai")
    assert w == {"$and": [
        {"filepath": "a.md"},
        {"heading_path": {"$contains": "背景"}},
        {"tags": {"$contains": "ai"}},
    ]}


def test_build_where_none():
    assert HybridRetriever._build_where() is None
    assert HybridRetriever._build_where(filepath="", heading=None, tag=None) is None


def test_run_tool_call_fallback_hybrid_to_vector(monkeypatch):
    """hybrid_search 全部失败 → 自动降级为 vector_search（成功）。"""
    import note_assistant.agent.tools as tools_mod

    def fake_dispatch(name, args):
        if name == "hybrid_search":
            raise RuntimeError("hybrid 挂了")
        if name == "vector_search":
            return ("（降级向量结果）", [])
        raise RuntimeError(name)

    monkeypatch.setattr(tools_mod, "_dispatch", fake_dispatch)
    text, results = run_tool_call("hybrid_search", {"query": "q"})
    assert "降级" in text
    assert results == []


def test_run_tool_call_all_fail_returns_graceful(monkeypatch):
    """hybrid 与 vector 都失败 → 返回空结果 + 友好提示，不抛异常。"""
    import note_assistant.agent.tools as tools_mod

    def fake_dispatch(name, args):
        raise RuntimeError(f"{name} 挂了")

    monkeypatch.setattr(tools_mod, "_dispatch", fake_dispatch)
    text, results = run_tool_call("hybrid_search", {"query": "q"})
    assert results == []
    assert "跳过" in text


def test_run_tool_call_success(monkeypatch):
    import note_assistant.agent.tools as tools_mod

    def fake_dispatch(name, args):
        return ("ok", [{"x": 1}])

    monkeypatch.setattr(tools_mod, "_dispatch", fake_dispatch)
    text, results = run_tool_call("get_note", {"filepath": "a.md"})
    assert text == "ok"
    assert results == [{"x": 1}]


def test_run_tool_call_unknown_tool():
    """未知工具 → _dispatch 直接返回友好提示且不抛异常。"""
    text, results = run_tool_call("nonexistent", {})
    assert "未知工具" in text
    assert results == []
