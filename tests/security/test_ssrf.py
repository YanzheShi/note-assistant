# tests/security/test_ssrf.py
"""L0-a 索引期 SSRF 防御：远程图抓取主机策略 + magic-bytes 图片核验。

关键断言：被拒主机【fetcher 绝不被调用】（先判后抓）；
非图片响应字节绝不落盘成资产。
"""
import pytest

from note_assistant.indexing.assets import (
    _looks_like_image,
    check_remote_host,
    resolve_image,
)

# 最小 PNG 头（magic-bytes 判定足够，不需要合法完整文件）
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
GIF_BYTES = b"GIF89a" + b"\x00" * 16
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 16
WEBP_BYTES = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 8
SVG_BYTES = b"<svg xmlns='http://www.w3.org/2000/svg'><text>x</text></svg>"


class TestCheckRemoteHost:
    @pytest.mark.parametrize("url", [
        "http://127.0.0.1:11434/api/tags",
        "http://localhost/x.png",
        "http://0.0.0.0/x.png",
        "http://10.0.0.5/x.png",
        "http://192.168.1.2/x.png",
        "http://172.16.0.9/x.png",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/x.png",
    ])
    def test_blocks_private_and_metadata(self, url):
        ok, why = check_remote_host(url)
        assert not ok, f"应拦截: {url}"
        assert why

    def test_allows_public_host(self):
        ok, why = check_remote_host("https://example.com/img.png")
        assert ok and why == ""

    def test_allowlist_mode(self):
        ok, _ = check_remote_host(
            "https://oss.javaguide.cn/x.png",
            policy="allowlist", allowlist=["oss.javaguide.cn"],
        )
        assert ok
        ok, why = check_remote_host(
            "https://evil.com/x.png", policy="allowlist", allowlist=["oss.javaguide.cn"],
        )
        assert not ok and "allowlist" in why

    def test_policy_all_preserves_legacy_behavior(self):
        ok, _ = check_remote_host("http://127.0.0.1/x.png", policy="all")
        assert ok


class TestResolveImageSSRF:
    def test_private_host_fetcher_never_called(self):
        calls = []

        def fetcher(url):
            calls.append(url)
            return PNG_BYTES

        res = resolve_image("http://127.0.0.1:11434/x.png", fetcher=fetcher)
        assert not res.ok
        assert "host blocked" in res.error
        assert calls == []  # 先判后抓：拒绝时不发生任何网络调用

    def test_metadata_host_blocked_before_fetch(self):
        calls = []
        res = resolve_image(
            "http://169.254.169.254/latest/meta-data/",
            fetcher=lambda u: (calls.append(u), PNG_BYTES)[1],
        )
        assert not res.ok and calls == []

    def test_public_host_fetches_ok(self):
        res = resolve_image("https://example.com/x.png", fetcher=lambda u: PNG_BYTES)
        assert res.ok and res.asset is not None
        assert res.asset.source_kind == "remote"

    def test_non_image_response_rejected(self):
        # SSRF 响应若是 JSON/HTML 等非图片字节，拒绝入库
        res = resolve_image("https://example.com/x.png", fetcher=lambda u: b'{"json": true}')
        assert not res.ok
        assert "not an image" in res.error

    def test_allowlist_enforced_in_resolve(self):
        res = resolve_image(
            "https://evil.com/x.png", fetcher=lambda u: PNG_BYTES,
            host_policy="allowlist", host_allowlist=["example.com"],
        )
        assert not res.ok and "host blocked" in res.error


class TestLooksLikeImage:
    @pytest.mark.parametrize("data", [PNG_BYTES, GIF_BYTES, JPEG_BYTES, WEBP_BYTES, SVG_BYTES])
    def test_image_bytes_accepted(self, data):
        assert _looks_like_image(data)

    @pytest.mark.parametrize("data", [
        b"<html><body>hello</body></html>",
        b'{"error": "forbidden"}',
        b"",
    ])
    def test_non_image_rejected(self, data):
        assert not _looks_like_image(data)
