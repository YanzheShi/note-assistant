# tests/api/test_assets_endpoint.py
"""P2：GET /assets/{asset_id} 端点测试（设计 9.1）。

从 settings.assets_dir 按 asset_id 读取图片，返回二进制 + ETag + 长缓存；
不存在时 404。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client_and_dir(tmp_path, monkeypatch):
    # 把资产目录指到临时目录并放一张图
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "abc123def4567890.png").write_bytes(b"\x89PNG\r\n\x1a\n fake-png-bytes")
    monkeypatch.setattr("note_assistant.config.settings.assets_dir", assets_dir)

    from note_assistant.api.main import app
    return TestClient(app), assets_dir


class TestAssetsEndpoint:
    def test_serves_image_with_etag(self, client_and_dir):
        client, _ = client_and_dir
        resp = client.get("/assets/abc123def4567890")
        assert resp.status_code == 200
        assert resp.content == b"\x89PNG\r\n\x1a\n fake-png-bytes"
        assert resp.headers.get("ETag") == "abc123def4567890"
        assert "max-age=31536000" in resp.headers.get("Cache-Control", "")
        assert resp.headers.get("content-type") == "image/png"

    def test_missing_returns_404(self, client_and_dir):
        client, _ = client_and_dir
        resp = client.get("/assets/deadbeefdeadbeef")
        assert resp.status_code == 404

    def test_dir_missing_returns_404(self, tmp_path, monkeypatch):
        monkeypatch.setattr("note_assistant.config.settings.assets_dir", tmp_path / "nope")
        from note_assistant.api.main import app
        client = TestClient(app)
        resp = client.get("/assets/abc123def4567890")
        assert resp.status_code == 404
