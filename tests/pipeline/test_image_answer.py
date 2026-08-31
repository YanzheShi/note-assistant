# tests/pipeline/test_image_answer.py
"""P2 共享工具（pipeline/image_answer.py）测试：图意图识别 / image 上下文渲染 / [[IMG:]] 后处理。

纯函数、零外部依赖，直接断言闭环行为。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from note_assistant.retrieval.types import RetrievalResult
from note_assistant.pipeline.image_answer import (
    MAX_AUTO_IMAGES,
    ImageMarkerStreamer,
    append_missing_images,
    ensure_image_selected,
    finalize_answer_images,
    has_image_intent,
    has_teachable_image_assets,
    missing_images_block,
    render_image_block,
    collect_image_assets,
    postprocess_answer,
)


def _chunk(metadata: dict | None = None, content: str = "正文", score: float = 0.8) -> RetrievalResult:
    return RetrievalResult(score=score, page_content=content, metadata=metadata or {})


# ── 图意图识别 ──
class TestHasImageIntent:
    def test_chinese_image_words(self):
        assert has_image_intent("记忆系统架构图长什么样")
        assert has_image_intent("这张流程图怎么画")

    def test_english_diagram(self):
        assert has_image_intent("show me the architecture diagram")

    def test_no_intent(self):
        assert not has_image_intent("RAG 是什么")
        assert not has_image_intent("")

    def test_partial_word_not_matched(self):
        # "图像" 含 "图" 应命中；纯无关词不命中
        assert has_image_intent("图像分类")
        assert not has_image_intent("今天天气真好")


# ── image chunk 上下文渲染 ──
class TestRenderImageBlock:
    def test_image_chunk_renders_with_marker(self):
        chunk = _chunk({
            "kind": "image",
            "image_type": "architecture_diagram",
            "image_title": "记忆系统架构",
            "image_description": "三层架构",
            "image_ocr_text": "Perception / Storage",
            "asset_id": "abc123def4567890",
        })
        block = render_image_block(chunk)
        assert block is not None
        assert "【图片·architecture_diagram】记忆系统架构" in block
        assert "三层架构" in block
        assert "Perception / Storage" in block
        assert "[[IMG:abc123def4567890]]" in block

    def test_text_chunk_returns_none(self):
        assert render_image_block(_chunk({"kind": "text"})) is None
        assert render_image_block(_chunk({})) is None

    def test_image_by_asset_and_description_without_kind(self):
        # 没有明确 kind 但带 asset_id + description 也判为图片块
        chunk = _chunk({"asset_id": "deadbeefcafebabe", "image_description": "一张图"})
        assert render_image_block(chunk) is not None

    def test_empty_image_chunk_falls_back_to_page_content(self):
        # 无 desc / 无 ocr / 无 asset：渲染空块没有信息量，返回 None
        # 让调用方回退 page_content（至少还有「图片: src (上下文)」摘要）
        chunk = _chunk({"kind": "image"}, content="图片: x.png (一些上下文)")
        assert render_image_block(chunk) is None

    def test_image_chunk_with_asset_only_still_renders(self):
        # 只有 asset（无理解字段）仍可引用出图，保留渲染
        chunk = _chunk({"kind": "image", "asset_id": "abc123def4567890"})
        block = render_image_block(chunk)
        assert block is not None
        assert "[[IMG:abc123def4567890]]" in block


# ── 资产收集 ──
class TestCollectImageAssets:
    def test_collects_asset_id_and_url(self):
        chunks = [
            _chunk({"asset_id": "abc123def4567890", "img_url": "/assets/abc123def4567890",
                    "image_title": "架构图", "kind": "image"}),
            _chunk({"asset_id": "feedfacefeedface", "img_url": "/assets/feedfacefeedface", "kind": "image"}),
        ]
        assets = collect_image_assets(chunks)
        assert assets["abc123def4567890"]["img_url"] == "/assets/abc123def4567890"
        assert assets["abc123def4567890"]["title"] == "架构图"
        assert "feedfacefeedface" in assets

    def test_no_asset_no_entry(self):
        assert collect_image_assets([_chunk({"kind": "text"})]) == {}


# ── [[IMG:]] 后处理 ──
class TestPostprocessAnswer:
    def test_replaces_marker(self):
        chunks = [_chunk({
            "asset_id": "abc123def4567890",
            "img_url": "/assets/abc123def4567890",
            "image_title": "记忆架构",
            "kind": "image",
        })]
        answer = "见 [[IMG:abc123def4567890]] 所示。"
        out = postprocess_answer(answer, chunks)
        assert out == "见 ![记忆架构](/assets/abc123def4567890) 所示。"
        assert "[[IMG:" not in out

    def test_unknown_asset_stripped(self):
        # 无对应资产：直接删除标记（不幻觉造图，也不留裸标记噪音；2026-08-31 修订）
        out = postprocess_answer("见 [[IMG:deadbeefdeadbeef]] 所示。", [])
        assert "[[IMG:" not in out
        assert out == "见  所示。"

    def test_multiple_markers(self):
        chunks = [
            _chunk({"asset_id": "aaaaaaaabbbbbbbb", "img_url": "/assets/aaaaaaaabbbbbbbb",
                    "image_title": "图A", "kind": "image"}),
            _chunk({"asset_id": "ccccccccdddddddd", "img_url": "/assets/ccccccccdddddddd",
                    "image_title": "图B", "kind": "image"}),
        ]
        answer = "对比 [[IMG:aaaaaaaabbbbbbbb]] 与 [[IMG:ccccccccdddddddd]]"
        out = postprocess_answer(answer, chunks)
        assert "![图A](/assets/aaaaaaaabbbbbbbb)" in out
        assert "![图B](/assets/ccccccccdddddddd)" in out

    def test_no_marker_unchanged(self):
        assert postprocess_answer("普通回答，无图。", [_chunk({"kind": "image"})]) == "普通回答，无图。"

    def test_empty_answer(self):
        assert postprocess_answer("", []) == ""


# ── 图片教学开关（system prompt 条件化依据）──
class TestHasTeachableImageAssets:
    def test_true_when_resolvable(self):
        chunks = [_chunk({"kind": "image", "asset_id": "abc123def4567890",
                          "img_url": "/assets/abc123def4567890"})]
        assert has_teachable_image_assets(chunks)

    def test_false_without_img_url(self):
        # asset_id 存在但 img_url 缺失 → 后处理替换不了 → 不教学
        chunks = [_chunk({"kind": "image", "asset_id": "abc123def4567890"})]
        assert not has_teachable_image_assets(chunks)

    def test_false_without_assets(self):
        assert not has_teachable_image_assets([_chunk({"kind": "image"})])
        assert not has_teachable_image_assets([])
        assert not has_teachable_image_assets(None)


# ── rerank 图片保位（图意图 boost 被交叉编码器清零后的确定性护栏）──
class TestEnsureImageSelected:
    AID = "abc123def4567890"

    def _img(self, score=0.1):
        return _chunk(
            {"kind": "image", "asset_id": self.AID, "img_url": f"/assets/{self.AID}",
             "image_title": "架构图"},
            content="图片理解：三层架构", score=score,
        )

    def _text(self, score, content="正文"):
        return _chunk({"kind": "text"}, content=content, score=score)

    def test_pins_image_when_intent_and_cut(self):
        # 图意图 query：rerank top-k 把图片挤掉了 → 从全量里补回最高分图片
        ranked = [self._text(0.9, "t1"), self._text(0.8, "t2"), self._text(0.7, "t3"), self._img(0.2)]
        selected = ranked[:2]  # 截断后无图
        out = ensure_image_selected("架构图长什么样", ranked, selected)
        assert len(out) == 2  # 长度不变（替换末位）
        assert any(r.metadata.get("kind") == "image" for r in out)
        # 被替换掉的是分数最低的 t2，且结果按分降序
        assert all("t2" != r.page_content for r in out)
        assert out == sorted(out, key=lambda r: r.score, reverse=True)

    def test_no_intent_no_pin(self):
        ranked = [self._text(0.9), self._text(0.8), self._img(0.2)]
        selected = ranked[:2]
        out = ensure_image_selected("RAG 的检索流程是什么", ranked, selected)
        assert out == selected  # 无图意图：不干预

    def test_already_has_image_no_change(self):
        ranked = [self._img(0.95), self._text(0.9), self._text(0.8)]
        selected = ranked[:2]
        assert ensure_image_selected("架构图长什么样", ranked, selected) == selected

    def test_no_image_candidate_no_change(self):
        ranked = [self._text(0.9), self._text(0.8)]
        selected = ranked[:2]
        assert ensure_image_selected("架构图长什么样", ranked, selected) == selected

    def test_empty_selected_returns_image(self):
        ranked = [self._text(0.9), self._img(0.2)]
        out = ensure_image_selected("架构图长什么样", ranked, [])
        assert len(out) == 1 and out[0].metadata.get("kind") == "image"


# ── 确定性补图（LLM 未写 [[IMG:]] 标记时的兜底显示）──
class TestMissingImagesBlock:
    AID = "abc123def4567890"

    def _chunks(self, n=1):
        return [
            _chunk({"kind": "image",
                    "asset_id": f"{i:016x}",
                    "img_url": f"/assets/{i:016x}",
                    "image_title": f"图{i}"})
            for i in range(1, n + 1)
        ]

    def test_unreferenced_image_appended(self):
        block = missing_images_block("纯文本回答，没有图。", self._chunks())
        assert "相关图片" in block
        assert f"![图1](/assets/{1:016x})" in block

    def test_referenced_image_not_duplicated(self):
        chunks = self._chunks()
        answer = f"见 ![图1](/assets/{1:016x}) 所示。"
        assert missing_images_block(answer, chunks) == ""

    def test_finalize_with_marker_not_duplicated(self):
        # 标记替换成功后，替换出的 img_url 即被视为已引用，不再补第二次
        chunks = self._chunks()
        out = finalize_answer_images(f"见 [[IMG:{1:016x}]] 所示。", chunks)
        assert out.count(f"/assets/{1:016x}") == 1
        assert "相关图片" not in out

    def test_append_without_marker(self):
        chunks = self._chunks()
        out = append_missing_images("没有标记的回答。", chunks)
        assert out.startswith("没有标记的回答。")
        assert f"![图1](/assets/{1:016x})" in out

    def test_no_assets_no_block(self):
        assert missing_images_block("回答", [_chunk({"kind": "text"})]) == ""
        assert missing_images_block("回答", []) == ""

    def test_asset_without_img_url_not_appended(self):
        # 无 img_url 的 asset（索引未闭环）绝不凭空造图
        chunks = [_chunk({"kind": "image", "asset_id": "ffff"})]
        assert missing_images_block("回答", chunks) == ""

    def test_auto_append_cap(self):
        block = missing_images_block("无图", self._chunks(MAX_AUTO_IMAGES + 2))
        assert block.count("![") == MAX_AUTO_IMAGES


# ── 流式后处理（标记跨 token 边界是最容易漏替换的地方）──
class TestImageMarkerStreamer:
    AID = "aaaaaaaabbbbbbbb"

    def _chunks(self):
        return [_chunk({"asset_id": self.AID, "img_url": f"/assets/{self.AID}",
                        "image_title": "架构图", "kind": "image"})]

    def _run(self, tokens, chunks=None):
        s = ImageMarkerStreamer(chunks if chunks is not None else self._chunks())
        out = "".join(s.feed(t) for t in tokens)
        return out + s.flush()

    def test_marker_split_across_tokens(self):
        # 标记被切成 6 段，仍必须正确替换
        tokens = ["见图：", "[[", "IMG", ":", self.AID, "]]", "，如上。"]
        assert self._run(tokens) == f"见图：![架构图](/assets/{self.AID})，如上。"

    def test_marker_in_single_token(self):
        assert self._run([f"A[[IMG:{self.AID}]]B"]) == f"A![架构图](/assets/{self.AID})B"

    def test_char_by_char(self):
        text = f"前 [[IMG:{self.AID}]] 后"
        assert self._run(list(text)) == f"前 ![架构图](/assets/{self.AID}) 后"

    def test_plain_text_passthrough(self):
        assert self._run(["纯", "文本", "回答"]) == "纯文本回答"

    def test_unknown_asset_marker_stripped(self):
        # 未知 asset_id：删除标记，不幻觉也不留噪音（2026-08-31 修订）
        tokens = ["x", "[[IMG:", "ffffffffffffffff", "]]", "y"]
        assert self._run(tokens) == "xy"

    def test_unclosed_marker_stripped_on_flush(self):
        # 流结束时标记未闭合：截断的标记片段剥离，其余文本保留
        assert self._run(["文本 [[IMG:", self.AID]) == "文本 "

    def test_no_assets_unknown_hex_stripped(self):
        # 无图片资产时也不能透传：未解析的十六进制标记要被剥离（2026-08-31 修订）
        out = self._run(["a", "[[IMG:", "ffffffffffffffff", "]]", "b"], chunks=[])
        assert out == "ab"

    def test_no_assets_nonhex_marker_unchanged(self):
        # 非十六进制的 [[IMG:...]] 形状文本不认作标记，原样保留（与 postprocess 同口径）
        out = self._run(["a", "[[IMG:", "x", "]]", "b"], chunks=[])
        assert out == "a[[IMG:x]]b"

    def test_partial_open_prefix_not_swallowed(self):
        # 以 "[[" 结尾但后面不是 IMG:，不能把这两个字符吞掉
        assert self._run(["文本[[", "wikilink]]"]) == "文本[[wikilink]]"

    def test_streams_incrementally_before_marker(self):
        # 关键：标记之前的内容必须立刻吐出，不能整段缓冲（否则流式失效）
        s = ImageMarkerStreamer(self._chunks())
        assert s.feed("这是一段很长的前置回答") == "这是一段很长的前置回答"

    def test_two_markers_in_stream(self):
        aid2 = "ccccccccdddddddd"
        chunks = self._chunks() + [
            _chunk({"asset_id": aid2, "img_url": f"/assets/{aid2}",
                    "image_title": "流程图", "kind": "image"})
        ]
        tokens = ["对比 [[IMG:", self.AID, "]] 和 [[IMG:", aid2, "]] 两张"]
        out = self._run(tokens, chunks=chunks)
        assert out == f"对比 ![架构图](/assets/{self.AID}) 和 ![流程图](/assets/{aid2}) 两张"
