# -*- coding: utf-8 -*-
"""阿里云百炼决策模型客户端 —— TypeSafe System One 协议封装。

协议：POST {host}/compatible-mode/v1/systemone
  body = {model, state, questions{id: {type: choice|noul|score, instructions, criteria}}}
  resp = {answers{id: {type, choice|noul|score, probabilities, confidence}}, usage, latency_ms}
一次前向返回全部问题的判定 + 校准概率，不生成文本。

设计约定（与零回归约定对齐）：
- get_decision_client() 在 settings.decision_model_enabled=False 时返回 None，
  agent 侧零开销、零行为变化；
- decide() 不重试不吞异常——超时/网络/解析异常一律向上抛，由调用方（router/reflect）
  回退到原 LLM 路径，回退本身就是最稳妥的重试；
- Session 全局单例复用 TLS 连接（逐请求新建连接时墙钟 ~445ms，其中 89% 是网络 RTT）；
- 业务封装 route_intent / judge_verdict 把 prompt 模板收口在此模块，
  agent.py 不出现决策模型的协议细节。

文档：https://docs.bailian.console.aliyun.com/zh/model-studio/decision-model-api
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

import requests

from note_assistant.config import settings

logger = logging.getLogger(__name__)

API_PATH = "/compatible-mode/v1/systemone"
MODEL_NAME = "decision-model-preview"

# 未显式配置域名时的探测列表（华北2 北京）。DashScope 兼容域名对该新特性 404，
# 仅作排在前面的规范探测；试用域名是实测可用的快速通道。
FALLBACK_HOSTS = [
    "https://dashscope.aliyuncs.com",
    "https://trial.cn-beijing.maas.aliyuncs.com",
]

# 探测成功后的工作域名缓存（进程级）。并发写竞争无害——多线程写同一正确值。
_WORKING_HOST: str | None = None
_WORKING_HOST_LOCK = threading.Lock()

# ── 决策模型用量累计（跨请求进程级，供评测报表做成本对比）──
# calls=成功次数；failures=回退/失败次数；input_tokens=百炼计费维度；server_ms 累计。
# 每次成功后同步落盘 data/dm_stats.json（评测进程退出后报表脚本可读）。
DM_STATS: dict = {"calls": 0, "failures": 0, "input_tokens": 0, "server_ms_total": 0.0}
_STATS_LOCK = threading.Lock()


def _record_success(data: dict) -> None:
    usage = (data or {}).get("usage") or {}
    with _STATS_LOCK:
        DM_STATS["calls"] += 1
        DM_STATS["input_tokens"] += int(usage.get("input_tokens") or 0)
        DM_STATS["server_ms_total"] += float((data or {}).get("latency_ms") or 0.0)
    _dump_stats()


def _record_failure() -> None:
    with _STATS_LOCK:
        DM_STATS["failures"] += 1
    _dump_stats()


def _dump_stats() -> None:
    try:
        from note_assistant.config import PROJECT_ROOT

        path = PROJECT_ROOT / "data" / "dm_stats.json"
        with _STATS_LOCK:
            snapshot = dict(DM_STATS)
        snapshot["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001  统计落盘失败不影响主链路
        logger.debug("decision_model.stats_dump_failed", exc_info=True)


def reset_dm_stats() -> None:
    """评测/测试辅助：清零累计并删除落盘文件。"""
    global DM_STATS
    with _STATS_LOCK:
        DM_STATS = {"calls": 0, "failures": 0, "input_tokens": 0, "server_ms_total": 0.0}
    try:
        from note_assistant.config import PROJECT_ROOT

        (PROJECT_ROOT / "data" / "dm_stats.json").unlink(missing_ok=True)
    except OSError:
        pass

_SESSION: requests.Session | None = None
_SESSION_LOCK = threading.Lock()


def _get_session() -> requests.Session:
    """全局 Session 单例：复用 TLS 连接池。urllib3 连接池线程安全。"""
    global _SESSION
    if _SESSION is None:
        with _SESSION_LOCK:
            if _SESSION is None:
                _SESSION = requests.Session()
    return _SESSION


def _resolve_hosts() -> list[str]:
    """优先级：显式配置 > 已缓存工作域名 > fallback 探测列表。"""
    hosts: list[str] = []
    if settings.decision_model_host:
        hosts.append(settings.decision_model_host.rstrip("/"))
    if _WORKING_HOST and _WORKING_HOST not in hosts:
        hosts.append(_WORKING_HOST)
    for h in FALLBACK_HOSTS:
        if h not in hosts:
            hosts.append(h)
    return hosts


class DecisionClient:
    """百炼决策模型客户端。线程安全（共享 Session + 连接池）。"""

    def __init__(self, api_key: str | None = None, timeout: int | None = None):
        self.api_key = api_key or settings.decision_model_api_key
        if not self.api_key:
            # config 留空回落到百炼标准环境变量
            import os

            self.api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        self.timeout = timeout or settings.decision_model_timeout
        if not self.api_key:
            raise ValueError(
                "决策模型缺少 API Key：请配置 DECISION_MODEL_API_KEY 或环境变量 DASHSCOPE_API_KEY"
            )

    # ──────────────────────────────────────────────
    # 协议层
    # ──────────────────────────────────────────────

    def decide(self, state: Any, questions: dict) -> dict:
        """一次前向请求。返回响应 JSON（answers/usage/latency_ms + _latency_ms_server）。

        依次探测 hosts，成功后缓存工作域名；全部失败抛 RuntimeError，
        其他异常（超时/连接）原样向上抛，由调用方回退 LLM。
        """
        body = {"model": MODEL_NAME, "state": state, "questions": questions}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        session = _get_session()
        last_err: Exception | None = None
        for host in _resolve_hosts():
            url = host + API_PATH
            try:
                resp = session.post(url, json=body, headers=headers, timeout=self.timeout)
            except requests.RequestException as e:
                logger.warning("decision_model.host_unreachable host=%s err=%s", host, type(e).__name__)
                last_err = e
                _record_failure()
                continue
            if resp.status_code == 200:
                with _WORKING_HOST_LOCK:
                    globals()["_WORKING_HOST"] = host
                data = resp.json()
                data["_server_ms"] = data.get("latency_ms")
                data["_host"] = host
                _record_success(data)
                return data
            logger.warning(
                "decision_model.http_error host=%s status=%s body=%s",
                host, resp.status_code, resp.text[:200],
            )
            last_err = RuntimeError(f"decision model HTTP {resp.status_code}")
            _record_failure()
        raise RuntimeError(f"决策模型所有域名均失败：{last_err}")

    # ──────────────────────────────────────────────
    # 业务封装 ①：router 意图识别（替代 agent.py::router 的 LLM JSON）
    # ──────────────────────────────────────────────

    ROUTER_QUESTIONS = {
        "needs_search": {
            "type": "choice",
            "instructions": (
                "该问题是否需要检索用户的个人知识库（Obsidian 笔记，含项目文档、"
                "技术笔记、个人记录）才能回答？凡是询问用户笔记/项目/文档内容的都判 search；"
                "纯闲聊、寒暄、与个人知识库无关的通用常识判 chat。"
            ),
            "criteria": {
                "search": "需要检索个人知识库（笔记/文档/项目记录）",
                "chat": "无需检索，直接对话即可（闲聊/常识/通用问题）",
            },
        },
        "has_reference": {
            "type": "noul",
            "instructions": "问题中是否包含指代词（它/那/这个/前者）或不完整表述，"
                            "需要结合上一轮对话才能理解？",
        },
    }

    def route_intent(self, question: str, condensed: str = "") -> dict:
        """router 意图判定。返回 {"needs_search": bool, "p_search": float, "p_reference": float}。"""
        state = question
        if condensed and condensed != question:
            state = f"{question}\n（若原问题含指代或不完整，可参考已消歧的改写：{condensed}）"
        data = self.decide(state, self.ROUTER_QUESTIONS)
        ans = data["answers"]["needs_search"]
        needs = str(ans.get("choice", "search")) == "search"
        probs = {str(k): float(v) for k, v in (ans.get("probabilities") or {}).items()}
        ref = data["answers"].get("has_reference", {})
        return {
            "needs_search": needs,
            "p_search": probs.get("search", 1.0 if needs else 0.0),
            "p_reference": float(ref.get("noul", 0.0)),
            "server_ms": data.get("latency_ms"),
        }

    # ──────────────────────────────────────────────
    # 业务封装 ②：Judge 五类 verdict（替代 agent.py::reflect 的 LLM JSON）
    # 注意：只产 verdict + 校准概率，不产 rewritten_query / clarify_question
    # （决策模型不生成文本；生成字段由 agent.py 在需要时追加 LLM 调用）。
    # ──────────────────────────────────────────────

    JUDGE_QUESTIONS = {
        "verdict": {
            "type": "choice",
            "instructions": (
                "判定已检索到的知识库片段是否足以回答用户问题，五选一。"
                "注意：片段可能只是部分相关，要看是否覆盖问题的全部要点。"
            ),
            "criteria": {
                "sufficient": "证据已包含回答问题所需的核心信息，可以直接生成答案",
                "need_rewrite": "证据不足，但换一个更精准的检索词可能找到",
                "need_more": "证据不足，需要补充检索（换工具/扩大范围）",
                "give_up": "知识库中大概率没有能回答该问题的内容",
                "need_clarify": "片段指向多个互不相干的主题，问题本身有歧义需要向用户澄清",
            },
        },
        "sufficient_prob": {
            "type": "noul",
            "instructions": "当前证据是否足以直接回答用户问题？（与 verdict 独立的校准概率）",
        },
    }

    def judge_verdict(
        self, question: str, evidence_text: str, iteration: int, max_iter: int
    ) -> dict:
        """Judge 充分性判定。返回 {"verdict", "probs", "p_sufficient", "server_ms"}。

        evidence_text 由调用方构造（应已经过 wrap_retrieved_context 等防注入包裹）。
        """
        state = (
            f"用户问题：{question}\n\n"
            f"=== 已检索到的知识库片段 ===\n{evidence_text}\n\n"
            f"已检索轮次：{iteration}/{max_iter}"
        )
        data = self.decide(state, self.JUDGE_QUESTIONS)
        ans = data["answers"]["verdict"]
        probs = {str(k): float(v) for k, v in (ans.get("probabilities") or {}).items()}
        sp = data["answers"].get("sufficient_prob", {})
        return {
            "verdict": str(ans.get("choice", "sufficient")),
            "probs": probs,
            "p_sufficient": float(sp.get("noul", 0.0)),
            "confidence": ans.get("confidence"),
            "server_ms": data.get("latency_ms"),
        }


_CLIENT: DecisionClient | None = None
_CLIENT_LOCK = threading.Lock()


def get_decision_client() -> DecisionClient | None:
    """工厂：未启用返回 None（零开销）；配置缺失/构造失败返回 None 并告警。"""
    global _CLIENT
    if not settings.decision_model_enabled:
        return None
    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:
                try:
                    _CLIENT = DecisionClient()
                except Exception:  # noqa: BLE001  缺 key 等配置问题：降级为不启用
                    logger.warning("decision_model.init_failed", exc_info=True)
                    return None
    return _CLIENT


def reset_decision_client() -> None:
    """测试辅助：清空单例（settings 被monkeypatch 后重建）。"""
    global _CLIENT
    with _CLIENT_LOCK:
        _CLIENT = None
