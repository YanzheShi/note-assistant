# tests/test_config.py
import os
from pathlib import Path
import pytest
from pydantic import ValidationError

# 注意：如果项目根没把 src/ 加入 PYTHONPATH，可能需要
# import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
from note_assistant.config import Settings


def test_env_override(monkeypatch):
    """monkeypatch 能覆盖 .env，字段类型正确"""
    monkeypatch.setenv("VAULT_PATH", "/tmp/test_vault")
    monkeypatch.setenv("EMBED_DIM", "768")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-dummy")

    s = Settings(_env_file=None)  # 忽略真实 .env，纯测 env
    assert s.vault_path == Path("/tmp/test_vault")
    assert s.embed_dim == 768          # str->int 自动转
    assert s.deepseek_api_key.get_secret_value() == "sk-dummy"


def test_defaults_without_env():
    """不配 .env 时默认值仍生效"""
    s = Settings(_env_file=None)  # _env_file=None 跳过 .env
    assert s.ollama_base_url == "http://localhost:11434"
    assert s.embed_model == "bge-m3:latest"
    assert s.chunk_size == 800


def test_required_field_missing(monkeypatch):
    """如果把必填字段（假如你把 api_key 改成无默认值），应报 ValidationError"""
    # 当前我们是 default=""，所以这步测的是"未来如果改必填不会忘"
    # 先演示：int 给个非数字会炸
    monkeypatch.setenv("EMBED_DIM", "not_a_number")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)