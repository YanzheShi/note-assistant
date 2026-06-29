"""
知识图谱：基于 Obsidian [[wikilinks]] 构建笔记关联图，支持 GraphRAG 式一跳扩展。

架构：
    VaultLoader.scan() → DocNode 列表
         ↓
    WikiGraph.build_from_docs() → 有向图 (NetworkX DiGraph)
         ↓
    WikiGraph.expand(hit_files, hop=1) → [(filepath, decay_score), ...]
         ↓
    注入 RAGChain：rerank 后扩展邻居 chunks → 附加到 context
"""

import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set

import networkx as nx

from note_assistant.config import settings
from note_assistant.indexing.types import DocNode


class WikiGraph:
    """
    基于 wikilinks 的有向知识图谱。

    节点 = 文档（filepath），边 = wikilink 引用关系。
    例如：笔记A 中有 [[笔记B]] → 有一条边 A → B。
    """

    def __init__(self):
        self.G: nx.DiGraph = nx.DiGraph()

    # ───────────────────────────────
    # 建图
    # ───────────────────────────────

    def build_from_docs(self, docs: list[DocNode], link_resolver=None) -> None:
        """
        【核心逻辑待实现】从所有文档的 wikilinks 建图。

        流程：
        1. 所有文件注册为节点（带 front_matter 属性）
        2. 遍历每个文档的 wikilinks，建边
        3. 找不到目标的 wikilink → 建 stub 节点（保留链接信息）

        Args:
            docs: VaultLoader.load_all() 返回的 DocNode 列表
                  每个节点必须有 .filepath 和 .wikilinks 属性
            link_resolver: 可选的模糊匹配函数 (link_text, docs) → filepath。
                          默认用内置的 _resolve_link（文件名 + aliases 匹配）。
        """
        # 提示：遍历 docs，注册节点，调用 link_resolver 或 self._resolve_link 建边
        for doc in docs:
            self.G.add_node(doc.filepath, front_matter=doc.front_matter)

        # 遍历wikilink 建边
        for doc in docs:
            source = doc.filepath
            for link in doc.wikilinks:
                # wikilink 保存的是文件名， 需要解析出来路径，因为图节点是路径
                link_path = self._resolve_link_to_realpath(link, docs)
                if link_path:
                    if link_path != source:
                        # 需要排除自环
                        self.G.add_edge(source, link_path)
                else:
                    #  链接了暂时不存在的节点， 先占位，便于以后增加笔记后能链接上去
                    stub = f"[[{link}]]"  # 如 "[[不存在的笔记]]"
                    self.G.add_node(stub,  front_matter={"stub": True, "link_text": link})
                    if stub != source:  # 排除自环
                        self.G.add_edge(source, stub)

        pass

    def _resolve_link_to_realpath(self, link: str, docs: list[DocNode]) -> Optional[str]:
        """
        【核心逻辑待实现】模糊匹配 wikilink → 实际文件路径。
        wikilink 只是保存的文件名，没有实际路径， 建图要把这个文件名跟对应的 doc匹配上
        例如: link="RAG 概念" → 匹配到 "RAG 概念.md"

        匹配策略（按优先级）：
        1. 文件名匹配（去掉 .md，大小写不敏感）
        2. front matter aliases 匹配
        3. 找不到 → 返回 None（build_from_docs 会建 stub 节点）

        Args:
            link: wikilink 文本（如 "FlashAttention 基础"）
            docs: 所有 DocNode 列表

        Returns:
            匹配到的 filepath，或 None
        """
        # 提示：遍历 docs，先比文件名（stem），再比 front_matter aliases

        lower_name = link.strip().lower()

        for doc in docs:
            if lower_name == doc.title.lower() or link == Path(doc.filepath).stem.lower():
                return doc.filepath
            else:
                aliases = doc.front_matter.get("aliases", [])
                if lower_name in [a.lower().strip() for a in aliases]:
                    return doc.filepath
                
        # 无对应文档
        return None

    # ───────────────────────────────
    # 扩展
    # ───────────────────────────────

    def expand(
        self,
        hit_files: Set[str],
        hop: int = 1,
        max_neighbors: int = 5
    ) -> List[Tuple[str, float]]:
        """
        【核心逻辑待实现】从命中文件出发，BFS hop 跳 → 返回 [(filepath, decay_score)]

        decay_score 衰减策略：
            hop=1 → 1.0
            hop=2 → 0.5
            hop=3 → 0.25
            ... 公式：decay = 1.0 / (2 ** (h - 1))

        Args:
            hit_files: 检索命中的文件路径集合
            hop: 扩展跳数（默认 1）
            max_neighbors: 最多返回多少个邻居

        Returns:
            [(filepath, decay_score), ...]，按 decay_score 降序
        """
        neighbors = []
        visited = set(hit_files)

        current_level = set(hit_files)
        for h in range(1, hop + 1):
            next_level = set()
            decay = 1.0 / (2 ** (h - 1))  # 1.0, 0.5, 0.25, ...

            for node in current_level:
                if node not in self.G:
                    continue
                for successor in self.G.successors(node):
                    if successor not in visited:
                        visited.add(successor)
                        next_level.add(successor)
                        neighbors.append((successor, decay))

            current_level = next_level

        # 限制最多扩展数量
        return neighbors[:max_neighbors]

    # ───────────────────────────────
    # 查询
    # ───────────────────────────────

    def get_successors(self, filepath: str) -> List[str]:
        """获取某文件的直接下游（它引用的笔记）"""
        if filepath not in self.G:
            return []
        return list(self.G.successors(filepath))

    def get_predecessors(self, filepath: str) -> List[str]:
        """获取某文件的直接上游（引用它的笔记）"""
        if filepath not in self.G:
            return []
        return list(self.G.predecessors(filepath))

    @property
    def node_count(self) -> int:
        return self.G.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.G.number_of_edges()

    # ───────────────────────────────
    # 持久化
    # ───────────────────────────────

    def save(self, path: str | Path | None = None) -> None:
        """保存图到 pickle 文件"""
        p = Path(path) if path else settings.bm25_index_path.with_suffix(".graph")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self.G, f)

    def load(self, path: str | Path | None = None) -> None:
        """从 pickle 文件加载图"""
        p = Path(path) if path else settings.bm25_index_path.with_suffix(".graph")
        with open(p, "rb") as f:
            self.G = pickle.load(f)

    # ───────────────────────────────
    # 调试
    # ───────────────────────────────

    def summary(self) -> str:
        """返回图的摘要信息（调试用）"""
        lines = [
            f"WikiGraph: {self.node_count} nodes, {self.edge_count} edges",
        ]
        # 找出度最高的节点
        if self.G.number_of_nodes() > 0:
            out_degrees = dict(self.G.out_degree())
            top_3 = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:3]
            lines.append("Top out-degree:")
            for node, deg in top_3:
                lines.append(f"  {node} → {deg} links")
        return "\n".join(lines)
