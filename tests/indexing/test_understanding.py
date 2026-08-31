"""P1-c VLM 理解层 + 资产层 测试（全部离线、确定性）。

设计要点：唯一的真实网络调用（VLM / 远程下载）都在可注入接口背后，
本文件全部用 fake client / 注入 fetcher 验证逻辑，绝不触发真实网络。
"""
import struct

import pytest

from note_assistant.indexing.assets import (
    ImageAsset,
    resolve_image,
)
from note_assistant.indexing.understanding import (
    ImageUnderstander,
    ImageUnderstanding,
    UnderstandingContext,
    VisionCache,
    grading_route,
    make_image_enricher,
    understand_image,
)
from note_assistant.indexing.preprocessor import RichPreprocessor
from note_assistant.indexing.types import ExtractedChunk


# ───────────────────────── 分级路由 ─────────────────────────
def _asset(mime="image/png", w=800, h=600, size=500_000, origin="x.png"):
    return ImageAsset(asset_id="a", source_kind="local", origin=origin,
                      local_path="", mime=mime, width=w, height=h, bytes_size=size)


def test_grading_route_svg_is_use_svg():
    a = _asset(mime="image/svg+xml", w=0, h=0)
    assert grading_route(a) == "use_svg"


def test_grading_route_small_image_decorative():
    a = _asset(w=80, h=80)  # <100x100
    assert grading_route(a) == "decorative"


def test_grading_route_extreme_aspect_decorative():
    a = _asset(w=2000, h=50)  # 宽高比 > 10
    assert grading_route(a) == "decorative"


def test_grading_route_tiny_file_decorative():
    a = _asset(w=800, h=600, size=3 * 1024)  # <5KB
    assert grading_route(a) == "decorative"


def test_grading_route_logo_name_decorative():
    a = _asset(origin="site-logo.png")
    assert grading_route(a) == "decorative"


def test_grading_route_normal_need_vlm():
    a = _asset(w=800, h=600, size=500_000, origin="arch.png")
    assert grading_route(a) == "need_vlm"


# ───────────────────────── ImageUnderstanding 校验 ─────────────────────────
def test_image_understanding_coerces():
    u = ImageUnderstanding.from_raw_dict({
        "image_type": "architecture_diagram",
        "title": "架构图",
        "entities": "MoE, Attention、KV Cache",  # 字符串 → 列表
        "confidence": "0.9",                      # 字符串 → float
        "ocr_text": None,                        # None → 默认
    })
    assert u.entities == ["MoE", "Attention", "KV Cache"]
    assert u.confidence == 0.9
    assert u.ocr_text == ""


def test_image_understanding_defaults_on_missing():
    u = ImageUnderstanding.from_raw_dict({})
    assert u.entities == []
    assert u.confidence == 0.0
    assert u.image_type == ""


# ───────────────────────── 缓存 ─────────────────────────
def test_vision_cache_roundtrip(tmp_path):
    cache = VisionCache(tmp_path / "vc.sqlite")
    u = ImageUnderstanding(image_type="photo", title="t", confidence=0.8)
    cache.put("asset1", "ctx1", "model-x", "v1", u)
    got = cache.get("asset1", "ctx1", "model-x", "v1")
    assert got is not None
    assert got.title == "t" and got.confidence == 0.8
    # 不同 prompt_version → 未命中
    assert cache.get("asset1", "ctx1", "model-x", "v2") is None
    # 不同 context → 未命中（同图不同笔记各存一份）
    assert cache.get("asset1", "ctx2", "model-x", "v1") is None
    cache.close()


# ───────────────────────── understand_image 核心 ─────────────────────────
_VALID_JSON = (
    '{"image_type":"architecture_diagram","title":"三层架构",'
    '"summary_short":"记忆分层","description":"画了三层","ocr_text":"Layer1",'
    '"entities":["L1","L2"],"data_points":"","mermaid_equivalent":"","confidence":0.9}'
)


def test_understand_image_valid_json(tmp_path):
    calls = []

    def fake_client(system, user_text, image_bytes, mime):
        calls.append(1)
        return _VALID_JSON

    ctx = UnderstandingContext(note_title="n", heading_path="h", alt="a")
    cache = VisionCache(tmp_path / "vc.sqlite")
    u = understand_image(fake_client, b"img", "image/png", ctx,
                         cache=cache, model_id="m", prompt_version="v1", asset_id="id1")
    assert u.title == "三层架构"
    assert u.entities == ["L1", "L2"]
    assert u.confidence == 0.9
    # 缓存命中 → 第二次不调 client
    u2 = understand_image(fake_client, b"img", "image/png", ctx,
                          cache=cache, model_id="m", prompt_version="v1", asset_id="id1")
    assert u2.title == "三层架构"
    assert len(calls) == 1
    cache.close()


def test_understand_image_json_error_then_retry_then_fallback():
    state = {"n": 0}

    def flaky_client(system, user_text, image_bytes, mime):
        state["n"] += 1
        if state["n"] == 1:
            return "not json at all"   # 第一次失败
        raise RuntimeError("boom")     # 第二次也失败 → 降级

    ctx = UnderstandingContext(alt="图注")
    u = understand_image(flaky_client, b"img", "image/png", ctx, max_retries=1)
    # 重试 1 次后仍失败 → 降级为 alt+上下文，confidence=0
    assert u.confidence == 0.0
    assert "图注" in (u.title + u.summary_short + u.description)
    assert state["n"] == 2


def test_understand_image_client_raises_fallback():
    def bad_client(system, user_text, image_bytes, mime):
        raise ConnectionError("network down")

    ctx = UnderstandingContext(alt="alt文本")
    u = understand_image(bad_client, b"img", "image/png", ctx, max_retries=0)
    assert u.confidence == 0.0
    assert u.image_type == "photo"


# ───────────────────────── 编排器预算护栏 ─────────────────────────
def test_understander_budget_exhausted(tmp_path, monkeypatch):
    def fake_client(system, user_text, image_bytes, mime):
        return _VALID_JSON

    # ImageUnderstander 默认 cache 走 settings.vision_cache_path：隔离到 tmp_path
    monkeypatch.setattr("note_assistant.config.settings.vision_cache_path", tmp_path / "vc.sqlite")
    uu = ImageUnderstander(max_calls=0, concurrency=1)
    ctx = UnderstandingContext(alt="x")
    # 预算为 0 → 不入 VLM，返回 pending 降级（confidence 0）
    result = uu.understand_sync(b"img", "image/png", ctx, client=fake_client)
    assert result.confidence == 0.0
    assert uu.stats["skipped"] == 1


def test_understander_calls_vlm_and_counts(tmp_path):
    def fake_client(system, user_text, image_bytes, mime):
        return _VALID_JSON

    uu = ImageUnderstander(max_calls=5, concurrency=2, cache=VisionCache(tmp_path / "vc.sqlite"))
    ctx = UnderstandingContext(alt="x")
    result = uu.understand_sync(b"img", "image/png", ctx,
                                asset_id="a1", client=fake_client)
    assert result.title == "三层架构"
    assert uu.stats["vlm_called"] == 1
    uu.cache.close()


# ───────────────────────── assets.resolve_image ─────────────────────────
def _write_png(path, w=200, h=150):
    # 最小合法 PNG 头（只需 IHDR 的宽高供 _read_dimensions 读取；不要求完整图像数据）
    data = (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\x0d" + b"IHDR"
        + struct.pack(">II", w, h) + b"\x08\x06\x00\x00\x00"
        + b"\x00\x00\x00\x00"
    )
    # 补到 >5KB，避免触发「体积<5KB → 装饰图」预判（仅测试夹具需要；真实图片远大于此）
    if len(data) < 6 * 1024:
        data = data + b"\x00" * (6 * 1024 - len(data))
    path.write_bytes(data)
    return data


def test_resolve_local_png(tmp_path):
    p = tmp_path / "arch.png"
    _write_png(p, 200, 150)
    res = resolve_image(str(p), vault_path=tmp_path)
    assert res.ok
    assert res.asset.mime == "image/png"
    assert res.asset.width == 200 and res.asset.height == 150
    assert res.asset.source_kind == "local"
    assert res.asset.data == p.read_bytes()


def test_resolve_remote_with_fetcher(tmp_path):
    payload = b"\x89PNG\r\n\x1a\n" + b"\x00" * 30

    def fake_fetch(url):
        return payload

    res = resolve_image("https://example.com/x.png", vault_path=tmp_path,
                        assets_dir=tmp_path / "assets", fetcher=fake_fetch)
    assert res.ok
    assert res.asset.source_kind == "remote"
    assert res.asset.data == payload
    # 落盘缓存
    assert (tmp_path / "assets").exists()


def test_resolve_remote_disabled(tmp_path):
    res = resolve_image("https://example.com/x.png", vault_path=tmp_path,
                        allow_remote_fetch=False)
    assert not res.ok
    assert "remote fetch disabled" in res.error


def test_resolve_missing(tmp_path):
    res = resolve_image("nope.png", vault_path=tmp_path)
    assert not res.ok


def test_resolve_relative_to_note_dir(tmp_path):
    """相对附件按「笔记所在目录」解析（回归：只按 vault 根会让笔记旁 assets/ 全落空）。"""
    note_dir = tmp_path / "note"
    (note_dir / "assets").mkdir(parents=True)
    png = _write_png(note_dir / "assets" / "arch.png")

    # vault 相对的 note_dir
    res = resolve_image("assets/arch.png", vault_path=tmp_path, note_dir="note")
    assert res.ok and res.asset.source_kind == "local"
    # 绝对 note_dir（无 vault_path 也能用）
    assert resolve_image("assets/arch.png", note_dir=str(note_dir)).ok
    # 不传 note_dir：维持原「只按 vault 根」行为，纯追加不改旧语义
    assert not resolve_image("assets/arch.png", vault_path=tmp_path).ok

    # 两处同名时笔记目录优先——否则会取错图
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "arch.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"root")
    res2 = resolve_image("assets/arch.png", vault_path=tmp_path, note_dir="note")
    assert res2.ok and res2.asset.data == png


def test_resolve_too_large(tmp_path):
    p = tmp_path / "big.png"
    _write_png(p, 200, 150)
    # 把文件撑大超过护栏
    p.write_bytes(p.read_bytes() + b"\x00" * 100)
    res = resolve_image(str(p), vault_path=tmp_path, image_max_bytes=10)
    assert not res.ok
    assert "too large" in res.error


# ───────────────────────── preprocessor 接入 enricher ─────────────────────────
def _img_ext(src="arch.png", alt="架构图", context="前文"):
    return ExtractedChunk(
        uid="[IMAGE_UID_00000001]", kind="image",
        placeholder="[IMAGE_UID_00000001]",
        raw=f"![{alt}]({src})", context=context,
        meta={"src": src, "alt": alt},
    )


def test_preprocessor_enricher_enriches_image():
    fake = lambda ext, hp: ("图片理解：一张好图", {"image_type": "photo", "image_confidence": 0.9})
    pp = RichPreprocessor(image_enricher=fake)
    pp.extracted = [_img_ext()]
    chunks = pp.generate_summaries()
    img = [c for c in chunks if c.metadata.get("kind") == "image"]
    assert img, "应有 image summary chunk"
    assert img[0].page_content == "图片理解：一张好图"
    assert img[0].metadata["image_type"] == "photo"
    assert img[0].metadata["image_confidence"] == 0.9


def test_preprocessor_no_enricher_keeps_default(tmp_path):
    """无 enricher 时图片摘要保持原行为（离线、零成本、不破现有测试契约）。"""
    pp = RichPreprocessor()
    pp.extracted = [_img_ext()]
    chunks = pp.generate_summaries()
    img = [c for c in chunks if c.metadata.get("kind") == "image"]
    assert img[0].page_content.startswith("图片: arch.png")


def test_preprocessor_enricher_none_falls_back():
    """enricher 返回 None → 退化为默认摘要。"""
    fake = lambda ext, hp: None
    pp = RichPreprocessor(image_enricher=fake)
    pp.extracted = [_img_ext()]
    chunks = pp.generate_summaries()
    img = [c for c in chunks if c.metadata.get("kind") == "image"]
    assert img[0].page_content.startswith("图片: arch.png")


# ───────────────────────── make_image_enricher（raster 路径，离线）─────────────────────────
def test_make_image_enricher_raster_calls_vlm(tmp_path, monkeypatch):
    p = tmp_path / "arch.png"
    _write_png(p, 200, 150)  # 非装饰（>100x100）
    calls = []

    # 注入 fake VLM client：monkeypatch understanding._default_vlm_call 通过闭包不易，
    # 这里直接验证 enricher 在 resolve 成功 + 非装饰时调用 understand_image —— 用 cache 拦截。
    # 为彻底离线，临时把 VLM 配置置空并用 fake：通过 monkeypatch 替换模块级 _default_vlm_call。
    import note_assistant.indexing.understanding as U

    orig = U._default_vlm_call

    def fake_vlm(system, user_text, image_bytes, mime):
        calls.append(1)
        return _VALID_JSON

    U._default_vlm_call = fake_vlm
    try:
        # G6：enricher 总开关默认 False，这里显式开启以验证 raster→VLM 路由
        monkeypatch.setattr("note_assistant.config.settings.image_understand_enabled", True)
        # 生产 enricher 已接 VisionCache：隔离到 tmp_path，不碰真实 data/vision_cache.sqlite
        monkeypatch.setattr("note_assistant.config.settings.vision_cache_path", tmp_path / "vc.sqlite")
        enricher = make_image_enricher(tmp_path)
        ext = _img_ext(src=str(p))
        out = enricher(ext, "章节")
        assert out is not None
        summary, meta = out
        assert "三层架构" in summary
        assert meta["image_type"] == "architecture_diagram"
        assert len(calls) == 1
    finally:
        U._default_vlm_call = orig


def test_make_image_enricher_decorative_skips_vlm(tmp_path, monkeypatch):
    p = tmp_path / "logo.png"
    _write_png(p, 40, 40)  # 装饰图（<100x100）
    import note_assistant.indexing.understanding as U
    calls = []
    orig = U._default_vlm_call

    def fake_vlm(system, user_text, image_bytes, mime):
        calls.append(1)
        return _VALID_JSON

    U._default_vlm_call = fake_vlm
    try:
        # G6：enricher 总开关默认 False，这里显式开启以验证装饰图跳过 VLM
        monkeypatch.setattr("note_assistant.config.settings.image_understand_enabled", True)
        # 生产 enricher 已接 VisionCache：隔离到 tmp_path，不碰真实 data/vision_cache.sqlite
        monkeypatch.setattr("note_assistant.config.settings.vision_cache_path", tmp_path / "vc.sqlite")
        enricher = make_image_enricher(tmp_path)
        ext = _img_ext(src=str(p))
        out = enricher(ext, "章节")
        assert out is None  # 装饰图 → 降级，不调 VLM
        assert calls == []
    finally:
        U._default_vlm_call = orig
