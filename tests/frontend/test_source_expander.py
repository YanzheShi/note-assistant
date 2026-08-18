"""前端来源渲染逻辑测试（P2 图片显示修复锁）。

锁定两个真实 bug 的回归：
  1. /agent 链路来源只发 ``kind`` 字段（runner._sources_from_results），前端此前只读
     ``type``，导致图片来源被误标成 text、徽章统计失真（本该 image×1 + text×N）。
  2. /assets 图片需要后端地址拼 URL；前端此前没把已知的 API_URL 透传，只打印文字 +
     一句 caption，不出图。修复后 render_sources 接收 backend_base_url 并拼出完整 URL。
"""

import sys
from pathlib import Path

# 让 import frontend 可用（与 app.py 同样的 sys.path 注入）
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pytest

import frontend.components.source_expander as se


class _FakeSt:
    """最小 Streamlit 替身：记录调用，不触真渲染。"""

    def __init__(self):
        self.markdown_calls = []
        self.image_calls = []
        self.caption_calls = []

    def markdown(self, text, *a, **k):
        self.markdown_calls.append(text)

    def image(self, url, *a, **k):
        self.image_calls.append(url)

    def caption(self, text, *a, **k):
        self.caption_calls.append(text)

    def checkbox(self, *a, **k):
        return False

    def expander(self, *a, **k):
        # 用作 `with st.expander(...) as exp:` —— 返回支持上下文管理器的哑对象
        class _Ctx:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return _Ctx()


@pytest.fixture
def fake_st(monkeypatch):
    """用替身替换模块级 st，避免真实 Streamlit 运行时。"""
    fake = _FakeSt()
    monkeypatch.setattr(se, "st", fake)
    return fake


def _agent_sources():
    """模拟 /agent/ask_stream 真实返回：来源只有 kind 字段，无 type。"""
    return [
        {
            "filepath": "多模态 RAG.md",
            "title": "多模态 RAG 与传统 RAG 对比及落地注意事项（2026）",
            "heading": "多模态 RAG 与传统 RAG 对比及落地注意事项（2026）",
            "score": 2.01,
            "kind": "image",
            "img_url": "/assets/49bd84ae16661dc8",
            "render_hint": "image:webp",
        },
        {"filepath": "a.md", "title": "A", "score": 0.8, "kind": "text"},
        {"filepath": "b.md", "title": "B", "score": 0.7, "kind": "text"},
    ]


def test_agent_image_source_labeled_as_image_not_text(fake_st):
    """kind=image 的来源应被识别为 image（而非被 fallback 成 text）。"""
    se.render_sources(_agent_sources(), backend_base_url="http://localhost:8005")
    badge = next((m for m in fake_st.markdown_calls if "个来源" in m), "")
    assert "`image`×1" in badge, f"徽章应含 `image`×1，实际：{badge}"
    assert "`text`×2" in badge, f"徽章应含 `text`×2，实际：{badge}"


def test_assets_image_renders_with_backend_base_url(fake_st):
    """传 backend_base_url 时，/assets 图片应拼出完整 URL 并经 st.image 渲染。"""
    se.render_sources(_agent_sources(), backend_base_url="http://localhost:8005")
    assert fake_st.image_calls, "图片应被渲染（st.image 被调用）"
    assert any(
        url == "http://localhost:8005/assets/49bd84ae16661dc8"
        for url in fake_st.image_calls
    ), f"st.image 应收到完整 URL，实际：{fake_st.image_calls}"


def test_no_backend_base_url_shows_actionable_caption(fake_st):
    """无后端地址时不出图，但给可操作的提示（而非静默或旧的 P2 占位文案）。"""
    se.render_sources(_agent_sources(), backend_base_url="")
    assert not fake_st.image_calls, "无后端地址时不应当尝试渲染图片"
    cap = " ".join(fake_st.caption_calls)
    assert "API 地址" in cap, f"caption 应提示填 API 地址，实际：{cap}"
