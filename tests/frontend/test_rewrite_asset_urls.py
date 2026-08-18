# tests/frontend/test_rewrite_asset_urls.py
"""答案正文图片 URL 重写测试（frontend/utils.rewrite_asset_urls）。

锁定「标记替换成功了图却不显示」的最后一公里断点：
后端图片闭环产出 ![title](/assets/{id}) 相对路径，直接进 st.markdown 会解析到
Streamlit 自己的端口而非后端 → 404。渲染前必须拼后端全地址；存原文、渲染时重写，
用户改 API 地址后历史消息仍可渲染。
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from frontend.utils import rewrite_asset_urls


class TestRewriteAssetUrls:
    def test_rewrites_relative_asset_url(self):
        md = "看图 ![架构图](/assets/abc123def4567890) 结束"
        out = rewrite_asset_urls(md, "http://localhost:8005")
        assert out == "看图 ![架构图](http://localhost:8005/assets/abc123def4567890) 结束"

    def test_multiple_images_all_rewritten(self):
        md = "![](/assets/a)\n![](/assets/b)"
        out = rewrite_asset_urls(md, "http://localhost:8005")
        assert out.count("http://localhost:8005/assets/") == 2

    def test_base_trailing_slash_normalized(self):
        out = rewrite_asset_urls("![](/assets/x)", "http://localhost:8005/")
        assert "(http://localhost:8005/assets/x)" in out
        assert "//assets" not in out

    def test_no_assets_unchanged(self):
        assert rewrite_asset_urls("普通文本回答", "http://localhost:8005") == "普通文本回答"

    def test_empty_text(self):
        assert rewrite_asset_urls("", "http://localhost:8005") == ""

    def test_empty_base_unchanged(self):
        # 未配置 API 地址：原样返回，行为不劣于改造前
        md = "![](/assets/x)"
        assert rewrite_asset_urls(md, "") == md

    def test_absolute_url_untouched(self):
        md = "![x](http://other-host/assets/y)"
        assert rewrite_asset_urls(md, "http://localhost:8005") == md
