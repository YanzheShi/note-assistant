# tests/indexing/test_enricher_asset.py
"""P2：make_image_enricher 把 asset_id / img_url 写进 chunk metadata，并把资产落盘 assets_dir。

验证 SVG 分支（离线、零 VLM）：enricher 返回 meta 含 asset_id + img_url(/assets/{id})，
且资产已写入 assets_dir，使 /assets 端点可统一服务本地/远程图。
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


def test_svg_enricher_returns_asset_id_and_img_url(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    svg = vault / "diagram.svg"
    svg.write_text('<svg viewBox="0 0 10 10"><rect/></svg>')

    assets_dir = tmp_path / "assets"
    monkeypatch.setattr("note_assistant.config.settings.assets_dir", assets_dir)
    # 生产 enricher 已接 VisionCache：隔离到 tmp_path，不碰真实 data/vision_cache.sqlite
    monkeypatch.setattr("note_assistant.config.settings.vision_cache_path", tmp_path / "vc.sqlite")

    from note_assistant.indexing.understanding import make_image_enricher

    # G6：enricher 总开关默认 False，这里显式开启以验证 svg 资产落盘
    monkeypatch.setattr("note_assistant.config.settings.image_understand_enabled", True)
    enricher = make_image_enricher(str(vault))
    ext = SimpleNamespace(meta={"src": "diagram.svg"}, raw="![[diagram.svg]]", context="")
    result = enricher(ext, "H1")

    assert result is not None
    summary, meta = result
    assert meta.get("render_hint") == "svg:inline"
    assert meta.get("asset_id")
    assert meta["img_url"] == f"/assets/{meta['asset_id']}"
    # 资产已落盘，/assets 端点可服务
    assert assets_dir.exists()
    assert list(assets_dir.glob(f"{meta['asset_id']}.*"))


def test_enricher_wires_vision_cache_in_production(tmp_path, monkeypatch):
    """审计修复：生产 enricher 必须接入 VisionCache（此前恒 cache=None）。

    否则每次重建索引都全量烧 VLM；接入后重索引命中 (asset_id|prompt_ver|model|context)
    即复用，成本趋零。验证：need_vlm 分支调用 understand_image 时 cache 为 VisionCache 实例。
    """
    vault = tmp_path / "vault"
    vault.mkdir()
    # >5KB 避开 decorative 体积阈值；PNG 头通过 magic 判定
    (vault / "img.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 6000)

    monkeypatch.setattr("note_assistant.config.settings.assets_dir", tmp_path / "assets")
    monkeypatch.setattr("note_assistant.config.settings.image_understand_enabled", True)
    monkeypatch.setattr("note_assistant.config.settings.vision_cache_path", tmp_path / "vc.sqlite")

    import note_assistant.indexing.understanding as understanding
    from note_assistant.indexing.understanding import make_image_enricher

    captured = {}

    def fake_understand(client, image_bytes, mime, context, *, cache=None, **kw):
        captured["cache"] = cache
        return understanding._fallback_understanding(context)

    monkeypatch.setattr(understanding, "understand_image", fake_understand)

    enricher = make_image_enricher(str(vault))
    ext = SimpleNamespace(meta={"src": "img.png"}, raw="![[img.png]]", context="")
    result = enricher(ext, "H1")

    assert result is not None
    assert isinstance(captured.get("cache"), understanding.VisionCache)
    assert captured["cache"].path == str(tmp_path / "vc.sqlite")
