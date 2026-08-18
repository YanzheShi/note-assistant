"""
debug_ingest.py — 单笔记逐步跑通 Obsidian RAG 索引主链，打印每阶段中间态。

为什么用这个而不是直接跑 index_vault：
    index_vault 是全库一键重建，跑一次要等全部笔记切分+embedding。
    这个脚本只取「一篇笔记」走完 ①~⑧ 全部节点，每步打印关键中间变量，
    方便你在 PyCharm 里配合断点，看清数据形态是怎么一路变的。

用法（PyCharm 直接 Run，或命令行）：
    uv run python scripts/debug_ingest.py                 # 跑 vault 第一篇笔记
    uv run python scripts/debug_ingest.py --file 01-xxx/笔记.md   # 指定某篇
    uv run python scripts/debug_ingest.py --strategy v2b         # 切换切分策略
    uv run python scripts/debug_ingest.py --commit               # 真正写入 ChromaDB
    uv run python scripts/debug_ingest.py --embed                # 单独测 Ollama embedding(dim)

想用 IDE 断点：
    在下方标了  # <<< BP  的行取消注释 breakpoint()，或在 PyCharm
    对应源码行点红点（断点清单见对话 / BREAKPOINTS 说明）。
    默认不写库（安全）；加 --commit 才真正 upsert（不 wipe，只追加/覆盖同名 id）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 让脚本在 uv run / 任意 cwd 下都能 import 到包（src/note_assistant）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from note_assistant.config import settings
from note_assistant.indexing.vault_loader import VaultLoader
from note_assistant.indexing.preprocessor import RichPreprocessor
from note_assistant.indexing.splitter import make_splitters, split_v1, split_v2, split_v2b
from note_assistant.indexing.ingestor import build_structural_prefix, Ingestor


def banner(t: str) -> None:
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None, help="指定单篇 md 相对 vault 的路径")
    ap.add_argument("--strategy", default=None, choices=["v1", "v2", "v2b"])
    ap.add_argument("--commit", action="store_true", help="真正写入 ChromaDB（默认只打印）")
    ap.add_argument("--embed", action="store_true", help="单独测试 Ollama embedding 维度")
    args = ap.parse_args()

    if args.strategy:
        settings.chunking_strategy = args.strategy  # type: ignore[assignment]

    print(f"chunking_strategy = {settings.chunking_strategy}")
    print(f"vault_path        = {settings.vault_path}")

    # ---------- ① VaultLoader ----------
    loader = VaultLoader()
    paths = loader.scan()
    if not paths:
        print("[WARN] 没扫到 .md，检查 vault_path 配置")
        return
    target = None
    if args.file:
        rel = args.file.replace("\\", "/")
        target = next((p for p in paths if str(p).replace("\\", "/").endswith(rel)), None)
        if target is None:
            print(f"[WARN] 没找到 {args.file}")
            return
    else:
        target = paths[0]

    banner("① VaultLoader.load_file")
    node = loader.load_file(target)
    print("filepath  :", node.filepath)
    print("title     :", node.title)
    print("tags      :", node.tags)
    print("wikilinks :", node.wikilinks[:5], "(共 %d)" % len(node.wikilinks))
    print("headings# :", len(node.headings), "→", [(h["level"], h["text"]) for h in node.headings[:3]])
    # breakpoint()  # <<< BP: vault_loader.py:57 —— 看 DocNode 全貌 / FM 容错后的 title 兜底

    # ---------- ② RichPreprocessor.process ----------
    preprocessor = RichPreprocessor()
    banner("② RichPreprocessor.process_with_meta")
    cleaned, fm_chunks = preprocessor.process_with_meta(node)
    print("cleaned[:200] :", repr(cleaned[:200]))
    print("extracted#    :", len(preprocessor.get_extracted()))
    for ext in preprocessor.get_extracted():
        print("   -", ext.kind, "placeholder=", ext.placeholder)
    for fc in fm_chunks:
        print("   fm_chunk:", fc.kind, "|", fc.page_content)
    # breakpoint()  # <<< BP: preprocessor.py:17 —— 看占位符是否把 code/table/mermaid 抽走了
    # breakpoint()  # <<< BP: preprocessor.py:162 —— 看 tags/aliases 是否生成了 fm_chunk

    # ---------- ③ splitter ----------
    hs, cs = make_splitters()
    banner("③ split (%s)" % settings.chunking_strategy)
    if settings.chunking_strategy == "v1":
        chunks = split_v1(node, cs)
        parents = []
    elif settings.chunking_strategy == "v2b":
        res = split_v2b(node, hs, cs)
        chunks, parents = res["children"], res["parents"]
    else:
        chunks = split_v2(node, hs, cs)
        parents = []
    print("chunk#    :", len(chunks))
    for i, c in enumerate(chunks[:3]):
        print(f"  [{i}] hp={c.metadata.get('heading_path','N/A')!r} | {c.page_content[:50]!r}")
    # breakpoint()  # <<< BP: splitter.py:122 —— 看 header_sp.split_text 的父块
    # breakpoint()  # <<< BP: splitter.py:141 —— 看 heading_path 拼接（h1 空是否 skip）

    # ---------- ④ restore ----------
    banner("④ restore 占位符还原")
    chunks = preprocessor.restore(chunks)
    parents = preprocessor.restore(parents)
    for i, c in enumerate(chunks[:3]):
        print(f"  [{i}] has_code={c.metadata.get('has_code')} has_table={c.metadata.get('has_table')} "
              f"has_image={c.metadata.get('has_image')}")
        print("      ->", repr(c.page_content[:80]))
    # breakpoint()  # <<< BP: preprocessor.py:52 —— 确认占位符已还原成原始富结构

    # ---------- ⑤ 补 metadata + 结构前缀 ----------
    banner("⑤ 补 metadata + 结构前缀")
    dir_ = str(Path(node.filepath).parent)
    if dir_ == ".":
        dir_ = ""
    for c in chunks:
        c.metadata["wikilinks"] = node.wikilinks
        c.metadata["filepath"] = node.filepath
        c.metadata["title"] = node.title
        if dir_:
            c.metadata["dir"] = dir_
        if node.tags:
            c.metadata["tags"] = node.tags
        prefix = build_structural_prefix(node, c.metadata, dir_)
        c.page_content = f"{prefix}\n\n{c.page_content}"
    print("首块 page_content 头部:")
    print(repr(chunks[0].page_content[:120]))
    # breakpoint()  # <<< BP: ingestor.py:14 —— 看结构前缀（目录 › 《标题》 › 路径）拼接结果

    # ---------- ⑥ generate_summaries ----------
    banner("⑥ generate_summaries")
    summary_chunks = preprocessor.generate_summaries()
    for sc in summary_chunks:
        print("   -", sc.kind, "|", sc.page_content[:70])
    # breakpoint()  # <<< BP: preprocessor.py:106 —— 看各类 summary 文本生成

    # ---------- ⑦ (v2b) docstore ----------
    if parents:
        banner("⑦ v2b 父块 → ParentDocstore (临时路径，不覆盖正式库)")
        tmp_path = PROJECT_ROOT / "data" / "docstore_debug.pkl"
        from note_assistant.retrieval.docstore import ParentDocstore
        docstore = ParentDocstore(tmp_path)
        for p in parents:
            p.metadata["filepath"] = node.filepath
            p.metadata["title"] = node.title
            p.metadata["wikilinks"] = node.wikilinks
            if dir_:
                p.metadata["dir"] = dir_
            docstore.add(p.metadata["parent_id"], p.page_content, p.metadata)
        docstore.save()
        print(f"父块数={len(parents)} 已写入临时 {tmp_path}")
        # breakpoint()  # <<< BP: ingestor.py:203 —— 看父块内容 / parent_id 规则

    # ---------- ⑧ upsert / embed ----------
    all_chunks = chunks + summary_chunks + fm_chunks
    banner("⑧ upsert → embed → ChromaDB")
    print("待入库 chunk# :", len(all_chunks))

    if args.embed:
        from note_assistant.indexing.embedder import OllamaEmbedder
        emb = OllamaEmbedder()
        v = emb.embed_one("测试 embedding 维度")
        print("embed dim     :", len(v))
        # breakpoint()  # <<< BP: embedder.py:19 —— 看 Ollama 实际返回结构 / 异常

    if args.commit:
        ing = Ingestor()
        n = ing.upsert(all_chunks)
        print(f"已 upsert {n} 条（未 wipe，仅追加/覆盖同名 id）")
        # breakpoint()  # <<< BP: ingestor.py:79 —— 看 metadata 清洗（空列表/类型转换）
        # breakpoint()  # <<< BP: ingestor.py:91 —— 看 embed 批量输入
    else:
        print("[skip] 未加 --commit，未写入 ChromaDB（只看中间态）")

    print("\n✅ 单笔记全链路走通，各阶段中间态如上。")


if __name__ == "__main__":
    main()
