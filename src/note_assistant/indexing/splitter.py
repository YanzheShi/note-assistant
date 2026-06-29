from typing import List, Dict, Any
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from note_assistant.config import settings
from note_assistant.indexing.types import DocNode, Chunk
from note_assistant.indexing.vault_loader import VaultLoader


# ================================================================
# 常量（不用改，Obsidian 标题层级固定）
# ================================================================
_HEADER_CONFIG = [
    ("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4"),
]


# ================================================================
# 工厂：两层 splitter（参数全暴露，你下午调）
# ================================================================
def make_splitters(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    separators: List[str] | None = None,
    keep_sep: bool = True,
    return_each_line: bool = False,
) -> tuple[MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter]:
    """
    v2/v3 共用工厂，你下午要调的核心参数：
    - separators：重点确认 "。" 的位置是否在 " " 之前（中文切分关键）
    - return_each_line：你的笔记 ## 起手，选 False（避免短标题单独成chunk）
    """
    cs = chunk_size or settings.chunk_size
    co = chunk_overlap or settings.chunk_overlap
    seps = separators or ["\n\n", "\n", "。", " ", ""]

    header_sp = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADER_CONFIG,
        return_each_line=return_each_line,
    )

    child_sp = RecursiveCharacterTextSplitter(
        chunk_size=cs,
        chunk_overlap=co,
        separators=seps,
        keep_separator=keep_sep,
    )

    return header_sp, child_sp

# ================================================================
# 边界转换：仅内部调用，不对外暴露（你下午不用改，但要理解作用）
# ================================================================
def docnode_to_lc_doc(node: DocNode) -> Document:
    """
    业务层 DocNode → LC Document，仅用于调用 LC Splitter/VectorStore
    注意：不转 front_matter（嵌套结构，Chroma 不支持）
    """
    metadata = {
        "filepath": node.filepath,
        "title": node.title,
        "wikilinks": node.wikilinks,
        # 不转front_matter（嵌套结构，Chroma不支持），双链直接从DocNode.headings取行号
    }
    # ChromaDB 不接受空列表作为 metadata 值，tags 为空则不设置
    if node.tags:
        metadata["tags"] = node.tags
    return Document(
        page_content=node.raw_md,
        metadata=metadata,
    )


# ================================================================
# v1: 单层 Recursive（基线，你下午填实现）
# ================================================================
def split_v1(node: DocNode, sp: RecursiveCharacterTextSplitter) -> List[Chunk]:
    """
    基线版本：纯 Recursive 切分，不保留标题层级
    你下午要实现的核心点：
    1. 调用 split_documents 前，先把 node 转成 LC Document
    2. metadata 仅包含 DocNode 基础字段，无 heading_path
    3. 确认 chunk 末尾是否被硬切（对比加 "。" 前后的差异）
    """

    lc_doc = docnode_to_lc_doc(node)
    chunks = sp.split_documents([lc_doc])

    result = []

    for i, chunk in enumerate(chunks):
        meta = dict(chunk.metadata)

        # 基线版不添加heading_path，因为没有标题层级信息
        meta.update({"chunk_index": i})
        result.append(Chunk(
            page_content=chunk.page_content,
            metadata=meta,
        ))

    return result

# ================================================================
# v2: 两层 Header + Recursive（生产路，你下午填实现）
# ================================================================
def split_v2(
    node: DocNode,
    header_sp: MarkdownHeaderTextSplitter,
    child_sp: RecursiveCharacterTextSplitter,
) -> List[Chunk]:
    """
    生产路候选：HeaderSplitter 保标题层级 → Recursive 细切
    核心决策点：
    1. HeaderSplitter 调用 split_text，入参是 node.raw_md（不是 LC Document）
    2. heading_path 拼接逻辑：h1 空时 skip（适配 ## 起手笔记，避免 "> 一、xxx" 畸形路径）
    3. wikilinks 整篇级缝入每个 chunk（Day3 可升级到 chunk 级）
    4. h1/h2/h3 正确继承到子 chunk 的 metadata
    """
    # 1. 得到父块（List[Document]）
    parent_chunks = header_sp.split_text(node.raw_md)
    print(f"Debug: 切出 {len(parent_chunks)} 个父块")
    for i, pc in enumerate(parent_chunks):
        print(f"  父块{i}: h1={pc.metadata.get('h1')!r}, h2={pc.metadata.get('h2')!r}")
        print(f"  内容预览: {pc.page_content[:80]!r}")

    # 2. 细切父块（继承metadata）
    fine_chunks = child_sp.split_documents(parent_chunks)

    # 构建整篇级 metadata（与 v1 保持一致）
    base_meta = {
        "filepath": node.filepath,
        "title": node.title,
        "wikilinks": node.wikilinks,
    }
    if node.tags:
        base_meta["tags"] = node.tags

    result = []
    for fc in fine_chunks:
        # 先放整篇级 metadata，再用 chunk 自身的 h1-h4 覆盖
        meta = {**base_meta, **fc.metadata}
        # 从metadata里取h1-h4，拼接成路径
        hp_parts = [meta.get(k, "") for k in ["h1", "h2", "h3", "h4"]]

        hp = " > ".join(p for p in hp_parts if p) or "无标题"
        meta["heading_path"] = hp
        result.append(Chunk(
            page_content=fc.page_content,
            metadata=meta,
        ))

    return result


# ================================================================
# v2b: Parent-Child 双存预留（Day3 评估，你下午不用实现）
# ================================================================
def split_v2b(
    node: DocNode,
    header_sp: MarkdownHeaderTextSplitter,
    child_sp: RecursiveCharacterTextSplitter,
) -> Dict[str, List[Chunk]]:
    """
    预留 Hierarchical Parent-Child 模式：
    - parents：HeaderSplitter 切出的未细切父 chunk
    - children：细切后的子 chunk
    - child_to_parent：子 chunk 到父 chunk 的映射
    你下午不用实现，DECISIONS.md 里先写清楚「当前 vault 篇短，暂不开这个」的理由
    """
    raise NotImplementedError("TODO: Day3 评估后实现")


# ================================================================
# v3: v2 + chunk 级 wikilinks 重扫（Day3 预研，你下午不用实现）
# ================================================================
def split_v3(
    node: DocNode,
    header_sp: MarkdownHeaderTextSplitter,
    child_sp: RecursiveCharacterTextSplitter,
) -> List[Chunk]:
    """
    预留 chunk 级 wikilinks 能力：
    每个 chunk 的 content 重扫 [[...]]，仅保留本 chunk 出现的链接
    你下午不用实现，DECISIONS.md 里写清楚「Day1 优先整篇级，后续按需升级」的理由
    """
    raise NotImplementedError("TODO: Day3 预研后实现")


# ================================================================
# 验证入口框架（你下午填对比逻辑）
# ================================================================
if __name__ == "__main__":
    loader = VaultLoader()
    nodes = loader.load_all()  # nodes 是 List[DocNode]，业务层核心结构
    print(f"📁 加载 {len(nodes)} 篇笔记，测试第一篇")
    test_node = nodes[1]

    if not nodes:
        print("⚠️ 没有加载到笔记，请检查vault路径")
        exit(1)

    print(f"测试笔记: {test_node.filepath}")
    print(f"笔记标题树: {[(h['level'], h['text']) for h in test_node.headings[:3]]}")

    # 初始化splitter
    _, sp_v1 = make_splitters()  # v1只用child splitter
    hs_v2, cs_v2 = make_splitters()  # v2用header+child

    result = split_v1(test_node, sp_v1)

    for i, c in enumerate(result):
        hp = c.metadata.get("heading_path", "N/A")
        print(f"[{i}] hp={hp} | {c.page_content[:60]!r}...")

    print('-' * 100)

    result = split_v2(test_node, hs_v2, cs_v2)

    for i, c in enumerate(result):
        hp = c.metadata.get("heading_path", "N/A")
        print(f"[{i}] hp={hp} | {c.page_content[:60]!r}...")

    # TODO: 初始化 splitter（调用 make_splitters）
    # TODO: 分别调用 v1/v2 切分测试笔记
    # TODO: 打印对比：v1/v2 的 chunk 数、heading_path、末尾5字（确认 "。" 是否生效）
    # TODO: 可选：估算全库 chunk 数，记录到 DECISIONS.md