"""TokenMeter + TokenUsageCallbackHandler 单元测试（三种 cache token 来源兼容）。"""
from __future__ import annotations

from note_assistant.llm.usage import TokenMeter, TokenUsageCallbackHandler, get_token_handler


class _Msg:
    def __init__(self, um):
        self.usage_metadata = um


class _Gen:
    def __init__(self, msg):
        self.message = msg


class _Resp:
    def __init__(self, um=None, llm_output=None):
        self.generations = [[_Gen(_Msg(um))]] if um is not None else [[]]
        self.llm_output = llm_output or {}


def test_meter_basic_and_hit_rate():
    m = TokenMeter()
    assert m.cache_hit_rate() == 0.0
    m.add(prompt=1000, completion=200, cache_read=500, cache_creation=100)
    assert m.prompt_tokens == 1000
    assert m.completion_tokens == 200
    assert m.cache_read_tokens == 500
    assert m.cache_creation_tokens == 100
    assert m.total_tokens == 1200
    assert m.llm_calls == 1
    assert m.cache_hit_rate() == 0.5
    d = m.to_dict()
    assert d["cache_hit_rate"] == 0.5
    assert d["llm_calls"] == 1


def test_handler_no_meter_is_noop():
    h = TokenUsageCallbackHandler()
    # meter=None：不抛、不累加（线上默认路径零副作用）
    h.on_llm_end(_Resp(um={"input_tokens": 10, "output_tokens": 5}))


def test_handler_accumulates_when_meter_set():
    m = TokenMeter()
    h = TokenUsageCallbackHandler()
    h.set_meter(m)
    h.on_llm_end(_Resp(um={"input_tokens": 10, "output_tokens": 5,
                           "input_token_details": {"cache_read": 3}}))
    assert m.prompt_tokens == 10
    assert m.cache_read_tokens == 3
    assert m.llm_calls == 1
    # set_meter 返回旧值（便于调用方恢复现场）
    assert h.set_meter(None) is m


def test_cache_openai_cached_tokens():
    m = TokenMeter()
    h = TokenUsageCallbackHandler()
    h.set_meter(m)
    h.on_llm_end(_Resp(um={"input_tokens": 1000, "output_tokens": 100,
                           "input_token_details": {"cache_read": 700}}))
    assert m.cache_read_tokens == 700
    assert m.cache_hit_rate() == 0.7


def test_cache_deepseek_via_llm_output():
    m = TokenMeter()
    h = TokenUsageCallbackHandler()
    h.set_meter(m)
    # 走路径 B：message.usage_metadata 为 None，llm_output.usage 带 prompt_tokens_details.cached_tokens
    h.on_llm_end(_Resp(llm_output={"usage": {
        "prompt_tokens": 500, "completion_tokens": 50,
        "prompt_tokens_details": {"cached_tokens": 300}}}))
    assert m.prompt_tokens == 500
    assert m.cache_read_tokens == 300


def test_cache_legacy_top_level_keys():
    m = TokenMeter()
    h = TokenUsageCallbackHandler()
    h.set_meter(m)
    h.on_llm_end(_Resp(um={"input_tokens": 800, "output_tokens": 80,
                           "cache_read_input_tokens": 200,
                           "cache_creation_input_tokens": 50}))
    assert m.cache_read_tokens == 200
    assert m.cache_creation_tokens == 50


def test_get_token_handler_singleton():
    assert get_token_handler() is get_token_handler()
