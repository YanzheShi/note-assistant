# -*- coding: utf-8 -*-
"""决策模型客户端单测：协议解析、域名探测/回退、业务封装映射、工厂开关。"""

import json

import pytest
import requests

from note_assistant.config import settings
from note_assistant.decision import client as dm


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = json.dumps(self._payload, ensure_ascii=False)

    def json(self):
        return self._payload


class _FakeSession:
    """按序返回预置响应/异常，记录调用 URL。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[str] = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append(url)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _sample_ok_payload():
    return {
        "model": "decision-model-preview",
        "request_id": "r-1",
        "answers": {
            "needs_search": {
                "type": "choice", "choice": "search", "confidence": 0.88,
                "probabilities": {"search": 0.94, "chat": 0.06},
            },
            "has_reference": {"type": "noul", "noul": 0.12},
        },
        "usage": {"input_tokens": 130},
        "latency_ms": 48.6,
    }


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """每个用例：清工作域名缓存 + 恢复默认开关，避免跨用例污染。"""
    monkeypatch.setattr(dm, "_WORKING_HOST", None)
    monkeypatch.setattr(settings, "decision_model_enabled", False)
    monkeypatch.setattr(settings, "decision_model_host", "")
    yield


def test_decide_parses_and_caches_working_host(monkeypatch):
    ok = _FakeResponse(200, _sample_ok_payload())
    fake = _FakeSession([ok, ok])
    monkeypatch.setattr(dm, "_get_session", lambda: fake)

    c = dm.DecisionClient(api_key="k")
    data = c.decide("state", {"q": {"type": "noul"}})
    assert data["answers"]["needs_search"]["choice"] == "search"
    assert fake.calls[0] == dm.FALLBACK_HOSTS[0] + dm.API_PATH  # 首次按探测列表
    assert dm._WORKING_HOST == dm.FALLBACK_HOSTS[0]             # 成功后缓存

    c.decide("state", {"q": {"type": "noul"}})
    assert fake.calls[1] == dm._WORKING_HOST + dm.API_PATH      # 第二次直接用缓存域名


def test_decide_falls_over_on_http_error(monkeypatch):
    fake = _FakeSession([
        _FakeResponse(404, {}),
        _FakeResponse(200, _sample_ok_payload()),
    ])
    monkeypatch.setattr(dm, "_get_session", lambda: fake)
    c = dm.DecisionClient(api_key="k")
    data = c.decide("state", {})
    assert fake.calls[0].startswith("https://dashscope.aliyuncs.com")
    assert fake.calls[1].startswith("https://trial.cn-beijing.maas.aliyuncs.com")
    assert data["answers"]["needs_search"]["choice"] == "search"
    assert dm._WORKING_HOST == "https://trial.cn-beijing.maas.aliyuncs.com"


def test_decide_raises_when_all_hosts_fail(monkeypatch):
    fake = _FakeSession([
        _FakeResponse(404, {}),
        _FakeResponse(500, {}),
    ])
    monkeypatch.setattr(dm, "_get_session", lambda: fake)
    c = dm.DecisionClient(api_key="k")
    with pytest.raises(RuntimeError):
        c.decide("state", {})


def test_decide_network_exception_propagates(monkeypatch):
    fake = _FakeSession([requests.ConnectionError("boom"), _FakeResponse(200, _sample_ok_payload())])
    monkeypatch.setattr(dm, "_get_session", lambda: fake)
    c = dm.DecisionClient(api_key="k")
    data = c.decide("state", {})  # 第一个域名网络异常 → 探测下一个并成功
    assert data["answers"]["needs_search"]["choice"] == "search"


def test_route_intent_mapping(monkeypatch):
    fake = _FakeSession([_FakeResponse(200, _sample_ok_payload())])
    monkeypatch.setattr(dm, "_get_session", lambda: fake)
    c = dm.DecisionClient(api_key="k")
    r = c.route_intent("笔记里 v2 和 v2b 的区别？", condensed="v2 与 v2b 区别")
    assert r["needs_search"] is True
    assert r["p_search"] == pytest.approx(0.94)
    assert r["p_reference"] == pytest.approx(0.12)
    assert r["server_ms"] == pytest.approx(48.6)


def test_judge_verdict_mapping(monkeypatch):
    payload = {
        "answers": {
            "verdict": {
                "type": "choice", "choice": "sufficient", "confidence": 0.91,
                "probabilities": {"sufficient": 0.9, "need_rewrite": 0.05,
                                  "need_more": 0.03, "give_up": 0.01, "need_clarify": 0.01},
            },
            "sufficient_prob": {"type": "noul", "noul": 0.9},
        },
        "latency_ms": 51.0,
    }
    fake = _FakeSession([_FakeResponse(200, payload)])
    monkeypatch.setattr(dm, "_get_session", lambda: fake)
    c = dm.DecisionClient(api_key="k")
    r = c.judge_verdict("q", "evidence", iteration=1, max_iter=3)
    assert r["verdict"] == "sufficient"
    assert r["probs"]["sufficient"] == pytest.approx(0.9)
    assert r["p_sufficient"] == pytest.approx(0.9)


def test_get_decision_client_disabled_returns_none(monkeypatch):
    monkeypatch.setattr(settings, "decision_model_enabled", False)
    assert dm.get_decision_client() is None


def test_get_decision_client_enabled_without_key_returns_none(monkeypatch):
    monkeypatch.setattr(settings, "decision_model_enabled", True)
    monkeypatch.setattr(settings, "decision_model_api_key", "")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    dm.reset_decision_client()
    try:
        assert dm.get_decision_client() is None  # 缺 key：构造失败降级为不启用
    finally:
        dm.reset_decision_client()


def test_get_decision_client_enabled_with_key(monkeypatch):
    monkeypatch.setattr(settings, "decision_model_enabled", True)
    monkeypatch.setattr(settings, "decision_model_api_key", "k-test")
    dm.reset_decision_client()
    try:
        c = dm.get_decision_client()
        assert isinstance(c, dm.DecisionClient)
    finally:
        dm.reset_decision_client()
