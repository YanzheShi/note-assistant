# tests/api/test_source_endpoint.py
"""原文查看端点：安全回源 + 系统程序打开。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient


@pytest.fixture
def client_and_vault(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "index.html").write_text(
        "<html><head><title>原文</title></head><body><h1>原文正文</h1></body></html>",
        encoding="utf-8",
    )
    (vault / "note.md").write_text("# 笔记正文", encoding="utf-8")
    (vault / "secret.txt").write_text("secret", encoding="utf-8")
    (vault / ".hidden.html").write_text("<p>hidden</p>", encoding="utf-8")
    (tmp_path / "outside.html").write_text("<p>outside</p>", encoding="utf-8")

    monkeypatch.setattr("note_assistant.config.settings.vault_path", vault)
    from note_assistant.api.main import app

    return TestClient(app, client=("127.0.0.1", 50000)), vault


class TestSourceEndpoint:
    def test_serves_html_with_sandbox_csp(self, client_and_vault):
        client, _ = client_and_vault
        resp = client.get("/source/index.html")
        assert resp.status_code == 200
        assert "原文正文" in resp.text
        assert resp.headers["content-type"].startswith("text/html")
        csp = resp.headers.get("Content-Security-Policy", "")
        assert "sandbox" in csp
        assert "default-src 'none'" in csp
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("ETag")

    def test_nested_windows_style_path(self, client_and_vault):
        client, vault = client_and_vault
        (vault / "sub").mkdir()
        (vault / "sub" / "doc.html").write_text("<p>嵌套</p>", encoding="utf-8")

        assert client.get("/source/sub/doc.html").status_code == 200
        assert client.get("/source/sub%5Cdoc.html").status_code == 200

    def test_serves_markdown_as_plain_text(self, client_and_vault):
        client, _ = client_and_vault
        resp = client.get("/source/note.md")
        assert resp.status_code == 200
        assert "笔记正文" in resp.text
        assert resp.headers["content-type"].startswith("text/plain")

    def test_rejects_traversal_and_non_indexed_suffix(self, client_and_vault):
        client, _ = client_and_vault
        assert client.get("/source/secret.txt").status_code == 404
        assert client.get("/source/.hidden.html").status_code == 404
        assert client.get("/source/%2e%2e/outside.html").status_code == 404

    def test_resolve_rejects_escape(self, client_and_vault):
        from note_assistant.api.main import _resolve_vault_file

        with pytest.raises(HTTPException):
            _resolve_vault_file("../outside.html")

    def test_open_with_system_app(self, client_and_vault, monkeypatch):
        client, vault = client_and_vault
        import note_assistant.api.main as main_mod

        opened = []
        monkeypatch.setattr(main_mod.os, "startfile", lambda p: opened.append(p), raising=False)

        resp = client.post("/source/open", json={"filepath": "index.html"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "opened"
        assert opened and str(vault / "index.html") == opened[0]

    def test_open_rejects_non_local_client(self, client_and_vault):
        from note_assistant.api.main import app

        remote = TestClient(app)  # 默认 client=("testclient", 50000)
        resp = remote.post("/source/open", json={"filepath": "index.html"})
        assert resp.status_code == 403
