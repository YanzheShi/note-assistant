from pathlib import Path
import re
import yaml
from typing import List, Dict, Any

from note_assistant.indexing.types import DocNode  # 导入DocNode
from note_assistant.indexing.ignore import is_ignored
from note_assistant.config import settings


class VaultLoader:
    def __init__(self, vault_path: str | Path | None = None):
        self.vault_path = Path(vault_path) if vault_path else settings.vault_path.resolve()

    def scan(self) -> List[Path]:
        """扫所有.md，按 indexing.ignore 规则排除隐藏目录与配置的忽略目录"""
        md_paths = []
        for p in self.vault_path.rglob("*.md"):
            if not is_ignored(p.relative_to(self.vault_path)):
                md_paths.append(p)
        return md_paths

    def _parse_front_matter(self, text: str) -> Dict[str, Any]:
        """解析front matter，无fm返回空字典"""
        match = re.match(r'^---\s*\n(.*?)\n---', text, re.DOTALL)
        if match:
            try:
                return yaml.safe_load(match.group(1)) or {}
            except Exception:
                return {}
        return {}

    def _extract_wikilinks(self, text: str) -> List[str]:
        """提取[[link]]，去重保序"""
        links = re.findall(r'\[\[([^\]\|]+)(?:\|[^\]]+)?\]\]', text)
        seen = set()
        result = []
        for link in links:
            if link not in seen:
                seen.add(link)
                result.append(link)
        return result

    def _extract_headings(self, text: str) -> List[Dict[str, Any]]:
        """提取标题树，带行号"""
        headings = []
        for i, line in enumerate(text.split('\n'), 1):
            match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if match:
                headings.append({
                    "level": len(match.group(1)),
                    "text": match.group(2).strip(),
                    "line": i
                })
        return headings

    def load_file(self, md_path: Path) -> DocNode:
        """加载单篇笔记，返回DocNode"""
        raw_full = md_path.read_text(encoding="utf-8")
        # 剥离front matter，取正文
        fm = self._parse_front_matter(raw_full)
        # 去掉front matter部分（如果有）
        if raw_full.lstrip().startswith("---\n"):
            raw_md = re.sub(r'^---\s*\n.*?\n---\s*\n', '', raw_full, flags=re.DOTALL)
        else:
            raw_md = raw_full

        # 提取特征
        title = fm.get("title", md_path.stem)
        tags = fm.get("tags", []) if isinstance(fm.get("tags"), list) else []
        wikilinks = self._extract_wikilinks(raw_md)
        headings = self._extract_headings(raw_md)

        return DocNode(
            filepath=str(md_path.relative_to(self.vault_path)),
            abs_path=md_path,
            raw_md=raw_md,
            front_matter=fm,
            title=title,
            tags=tags,
            wikilinks=wikilinks,
            headings=headings
        )

    def load_all(self) -> List[DocNode]:
        """加载所有笔记，返回DocNode列表"""
        return [self.load_file(p) for p in self.scan()]


# 测试用
if __name__ == "__main__":
    loader = VaultLoader()
    nodes = loader.load_all()
    print(f"📁 加载 {len(nodes)} 篇笔记")
    for node in nodes[:2]:
        print(f"\n📄 {node.filepath}")
        print(f"标题: {node.title}")
        print(f"标签: {node.tags}")
        print(f"双链: {node.wikilinks[:3]}")
        print(f"标题树: {[h['text'] for h in node.headings[:3]]}")