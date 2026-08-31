"""资产层（设计文档第四章，P1-c/P1-d 的取图底座）。

职责：把笔记里的图片 `src`（远程 URL / 相对路径 / Obsidian 短名 `![[x]]`）
解析成**可送入 VLM 或原生解析器的字节 + mime**，并记录来源类型。

设计要点（对齐设计文档 4.1~4.3）：
- `ImageAsset` 以内容哈希（sha256[:16]）为主键，跨笔记同图天然去重。
- 远程图下载走标准库 `urllib`（不引入 httpx 新依赖），先写临时文件再原子 rename，
  失败标记 `missing` 并降级，绝不因图片失败中断整库索引。
- `image_allow_remote_fetch=False`（隐私开关）时只处理本地资产。
- 远程下载是唯一的**真实网络调用**点，测试通过注入 `fetcher` 完全离线。

本模块刻意保持「取图」的最小职责，VLM 调用 / 原生解析在 `understanding.py` /
`svg.py` 中完成。
"""
from __future__ import annotations

import hashlib
import ipaddress
import os
import re
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.parse import urlparse

from note_assistant.indexing.ignore import is_ignored


# 当作图片处理的扩展名（SVG 同时是结构化图，路由交给理解层）
IMAGE_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg",
}
# 结构化图表源文件（原生解析层处理，不当普通图片）
CHART_EXTS = {
    ".excalidraw", ".drawio", ".xml", ".plantuml", ".puml", ".mmd", ".mermaid",
}

_DEFAULT_UA = "note-assistant/1.0"


@dataclass
class ImageAsset:
    """物理图片资产 —— 以内容哈希为主键，跨笔记天然去重。"""
    asset_id: str                 # sha256(bytes)[:16]，主键
    source_kind: str              # "local" | "remote" | "missing"
    origin: str                   # 原始 URL 或 vault 内绝对路径
    local_path: str               # 本地路径（远程图缓存到 data/assets/<id>.<ext>；纯内存取图为空串）
    mime: str                     # image/png | image/svg+xml | ...
    width: int = 0
    height: int = 0
    bytes_size: int = 0
    etag: str = ""                # 远程图的 HTTP ETag（增量判断用）
    data: bytes = b""             # 解析出的原始字节（供 SVG 解析 / VLM 即时消费，用完即弃）


@dataclass
class ImageResolveResult:
    ok: bool
    asset: Optional[ImageAsset] = None
    error: str = ""


def mime_for_ext(path: str) -> str:
    """按扩展名推断 mime；SVG 特判为 image/svg+xml，其余统一 image/<ext>。"""
    ext = Path(path).suffix.lower()
    if ext == ".svg":
        return "image/svg+xml"
    if ext:
        return f"image/{ext.lstrip('.')}"
    return "image/png"


def _ext_for_mime(mime: str, fallback: str = ".bin") -> str:
    if mime == "image/svg+xml":
        return ".svg"
    if mime.startswith("image/"):
        return "." + mime[len("image/"):]
    return fallback


def _content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _is_svg_bytes(data: bytes) -> bool:
    head = data.lstrip()[:64].lower()
    return head.startswith(b"<?xml") or head.startswith(b"<svg") or b"<svg" in head


def _read_dimensions(data: bytes, mime: str) -> tuple[int, int]:
    """尽力而为地读取图片尺寸（不引入 Pillow 等新依赖）。

    支持 PNG / GIF / JPEG 基础头与 SVG viewBox；其它格式返回 (0,0)。
    返回 (0,0) 时分级路由不会因尺寸误判为 decorative（安全）。
    """
    if mime == "image/svg+xml" or _is_svg_bytes(data):
        m = re.search(rb'viewBox\s*=\s*["\']?\s*[\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)', data)
        if m:
            return int(float(m.group(1))), int(float(m.group(2)))
        m = re.search(rb'width\s*=\s*["\']?\s*([\d.]+)', data)
        h = re.search(rb'height\s*=\s*["\']?\s*([\d.]+)', data)
        if m and h:
            return int(float(m.group(1))), int(float(h.group(1)))
        return (0, 0)
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        import struct
        w, h = struct.unpack(">II", data[16:24])
        return (w, h)
    if data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
        import struct
        w, h = struct.unpack("<HH", data[6:10])
        return (w, h)
    if data[:2] == b"\xff\xd8":  # JPEG
        return _jpeg_dimensions(data)
    return (0, 0)


def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
    import struct
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            h, w = struct.unpack(">HH", data[i + 5:i + 9])
            return (w, h)
        # 段长度
        if i + 4 >= len(data):
            break
        seg_len = struct.unpack(">H", data[i + 2:i + 4])[0]
        i += 2 + seg_len
    return (0, 0)


class AttachmentIndex:
    """Obsidian 短名 `![[Pasted image.png]]` 解析（复刻最短唯一路径规则）。

    按图片扩展名 rglob 全 vault（排除隐藏目录与 indexing.ignore 配置的忽略目录），
    建 `filename → [paths]` 索引。
    """

    def __init__(self, vault_path: Path):
        self.vault_path = Path(vault_path)
        self._by_name: Dict[str, List[Path]] = {}
        self._build()

    def _build(self) -> None:
        if not self.vault_path.exists():
            return
        for ext in IMAGE_EXTS | CHART_EXTS:
            for p in self.vault_path.rglob(f"*{ext}"):
                if is_ignored(p.relative_to(self.vault_path)):
                    continue
                self._by_name.setdefault(p.name, []).append(p)

    def resolve(self, name: str) -> Optional[Path]:
        """解析短名：精确文件名匹配优先，唯一命中即返回，重名返回首个。"""
        name = name.strip()
        if not name:
            return None
        if name in self._by_name:
            paths = self._by_name[name]
            return paths[0] if paths else None
        # 退一步：带扩展名的精确匹配已覆盖；这里不再模糊匹配，避免误命中
        return None


def _default_fetch(url: str) -> bytes:
    """标准库下载远程图（唯一的真实网络调用点），可被测试注入替换。"""
    req = urllib.request.Request(url, headers={"User-Agent": _DEFAULT_UA})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.read()


# ── L0-a SSRF 主机策略（设计文档 §4-L0）────────────────────────────
# 远程图抓取是系统内唯一 URL 拉取点：笔记可嵌入任意 URL，索引器若无条件抓取，
# 等于替恶意内容访问内网（本机 Ollama / 云元数据 / 内网服务）。
_BLOCKED_NETS = [
    ipaddress.ip_network(n)
    for n in (
        "0.0.0.0/8", "127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12",
        "192.168.0.0/16", "169.254.0.0/16",        # IPv4：环回/私网/链路本地(含元数据)
        "::1/128", "fc00::/7", "fe80::/10",         # IPv6：环回/私网/链路本地
    )
]


def check_remote_host(
    url: str,
    policy: str = "block_private",
    allowlist: Optional[List[str]] = None,
) -> tuple[bool, str]:
    """远程抓取主机策略判定，返回 ``(是否放行, 拒绝原因)``。

    - ``all``：现状行为（不设防，仅保留兼容）；
    - ``allowlist``：仅放行白名单域名（精确匹配）；
    - ``block_private``（默认）：host 为字面 IP 且落在环回/私网/链路本地/元数据
      网段 → 拒绝。**已知限制**：域名形态不解析 DNS（DNS rebinding 不在本层拦截，
      高安全需求请用 allowlist）。
    """
    if policy == "all":
        return True, ""
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False, "invalid url"
    if not host:
        return False, "no host"
    if policy == "allowlist":
        if host in set(allowlist or []):
            return True, ""
        return False, f"host not in allowlist: {host}"
    # block_private
    if host in ("localhost",):  # 字面域名形态的环回也要拦（不止字面 IP）
        return False, "blocked host: localhost"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return True, ""  # 其它域名：放行（限制见 docstring）
    for net in _BLOCKED_NETS:
        if ip.version == net.version and ip in net:
            return False, f"blocked network range: {host}"
    return True, ""


def _looks_like_image(data: bytes) -> bool:
    """magic-bytes 图片核验：SSRF 响应可能不是图片（如 JSON/HTML），拒绝入库。"""
    if not data:
        return False
    if _is_svg_bytes(data):
        return True
    if data[:8] == b"\x89PNG\r\n\x1a\n":  # PNG
        return True
    if data[:6] in (b"GIF87a", b"GIF89a"):  # GIF
        return True
    if data[:2] == b"\xff\xd8":  # JPEG
        return True
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":  # WebP
        return True
    return False


def _candidate_bases(
    vault_path: Optional[Path], note_dir: Optional[str]
) -> List[Path]:
    """相对 src 的解析基准目录，按优先级排列：笔记所在目录 > vault 根。

    Markdown 相对链接的语义是「相对所在笔记」，Obsidian 默认把附件放在笔记旁的
    ``assets/`` 目录；只按 vault 根解析会让这类图整体落空（索引期取不到图 →
    没有 asset_id → 前端只剩裸标记）。vault 根的旧口径保留为兜底，纯尾部追加，
    原本能解析到的图行为不变。
    """
    bases: List[Path] = []
    if note_dir:
        nd = Path(note_dir)
        if nd.is_absolute():
            bases.append(nd)
        elif vault_path is not None:
            bases.append(Path(vault_path) / nd)
    if vault_path is not None:
        bases.append(Path(vault_path))
    return bases


def _write_cached(local_path: str, data: bytes) -> None:
    """先写临时文件再原子 rename，防半截文件。"""
    parent = os.path.dirname(local_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=parent or ".")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, local_path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def resolve_image(
    src: str,
    *,
    vault_path: Optional[Path] = None,
    note_dir: Optional[str] = None,
    allow_remote_fetch: bool = True,
    assets_dir: Optional[Path] = None,
    fetcher: Optional[Callable[[str], bytes]] = None,
    image_max_bytes: int = 10 * 1024 * 1024,
    host_policy: str = "block_private",
    host_allowlist: Optional[List[str]] = None,
) -> ImageResolveResult:
    """把图片 src 解析为字节 + mime + 资产元数据。

    Args:
        src: 笔记里的地址（远程 URL / 相对路径 / Obsidian 短名）。
        vault_path: vault 根（用于相对路径与短名解析）。
        note_dir: 该图片所在笔记的目录（绝对路径，或相对 ``vault_path`` 的路径）。
            相对 ``src`` 优先按它解析——Markdown 附件的相对语义是相对笔记文件，
            Obsidian 常把图放在笔记旁的 ``assets/`` 里。省略时退化为纯 vault 根口径。
        allow_remote_fetch: 隐私开关，False 时远程图直接判 missing。
        assets_dir: 远程图本地缓存目录；None 时远程图仅取内存不落盘。
        fetcher: 注入式下载函数（测试用）；默认 urllib。
        image_max_bytes: 超过此体积的图跳过（省 token）。
        host_policy: L0-a SSRF 主机策略（block_private / allowlist / all）。
        host_allowlist: allowlist 模式下的域名白名单。

    Returns:
        ImageResolveResult：ok=False 时 asset=None、error 描述原因。
    """
    src = (src or "").strip()
    if not src:
        return ImageResolveResult(ok=False, error="empty src")

    # ── 远程 URL ──
    if src.lower().startswith(("http://", "https://")):
        if not allow_remote_fetch:
            return ImageResolveResult(ok=False, error="remote fetch disabled")
        # L0-a：先判主机再抓取——拒绝环回/私网/链路本地/元数据地址（SSRF 防护）
        ok, why = check_remote_host(src, policy=host_policy, allowlist=host_allowlist)
        if not ok:
            return ImageResolveResult(ok=False, error=f"host blocked: {why}")
        try:
            data = (fetcher or _default_fetch)(src)
        except Exception as e:  # 网络/超时/4xx → 降级，不中断
            return ImageResolveResult(ok=False, error=f"fetch failed: {e}")
        if len(data) > image_max_bytes:
            return ImageResolveResult(ok=False, error="too large")
        # L0-a：响应必须是图片字节，否则拒绝（防把 SSRF 响应体当资产落盘）
        if not _looks_like_image(data):
            return ImageResolveResult(ok=False, error="fetched bytes are not an image")
        mime = mime_for_ext(src)
        if _is_svg_bytes(data):
            mime = "image/svg+xml"
        asset_id = _content_hash(data)
        local_path = ""
        if assets_dir is not None:
            assets_dir = Path(assets_dir)
            assets_dir.mkdir(parents=True, exist_ok=True)
            local_path = str(assets_dir / f"{asset_id}{_ext_for_mime(mime)}")
            _write_cached(local_path, data)
        w, h = _read_dimensions(data, mime)
        return ImageResolveResult(
            ok=True,
            asset=ImageAsset(
                asset_id=asset_id, source_kind="remote", origin=src,
                local_path=local_path, mime=mime,
                width=w, height=h, bytes_size=len(data), data=data,
            ),
        )

    # ── 本地（绝对路径 / 笔记目录相对 / vault 相对 / 短名）──
    candidate: Optional[Path] = None
    as_path = Path(src)
    if as_path.is_absolute() and as_path.exists():
        candidate = as_path
    else:
        for base in _candidate_bases(vault_path, note_dir):
            p = base / src
            if p.exists():
                candidate = p
                break
        if candidate is None and vault_path is not None:
            # Obsidian 短名：![[x.png]] → 全局唯一文件名解析
            idx = AttachmentIndex(vault_path)
            resolved = idx.resolve(src)
            if resolved is not None:
                candidate = resolved

    if candidate is None or not candidate.exists():
        return ImageResolveResult(ok=False, error="not found locally")

    try:
        data = candidate.read_bytes()
    except Exception as e:
        return ImageResolveResult(ok=False, error=f"read failed: {e}")
    if len(data) > image_max_bytes:
        return ImageResolveResult(ok=False, error="too large")
    mime = mime_for_ext(str(candidate))
    if _is_svg_bytes(data):
        mime = "image/svg+xml"
    asset_id = _content_hash(data)
    w, h = _read_dimensions(data, mime)
    # P2：本地图也统一复制一份到 assets_dir，使 /assets/{asset_id} 对本地/远程图一致可用，
    # 前端只需走 URL，不必再拼 vault 相对路径。
    local_path = str(candidate)
    if assets_dir is not None:
        assets_dir = Path(assets_dir)
        assets_dir.mkdir(parents=True, exist_ok=True)
        cached = str(assets_dir / f"{asset_id}{_ext_for_mime(mime)}")
        _write_cached(cached, data)
        local_path = cached
    return ImageResolveResult(
        ok=True,
        asset=ImageAsset(
            asset_id=asset_id, source_kind="local",
            origin=str(candidate), local_path=local_path,
            mime=mime, width=w, height=h, bytes_size=len(data), data=data,
        ),
    )
