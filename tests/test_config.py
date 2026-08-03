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
    # 主 LLM 通道走 AGENT_API_KEY，需能被 env 覆盖。
    # 必须显式 setenv：config.py 导入时 load_dotenv 已把真实 .env 灌进 os.environ，
    # 只靠 _env_file=None 隔离不掉（同 test_chunking_strategy 的坑）。
    monkeypatch.setenv("AGENT_API_KEY", "sk-dummy-agent")

    s = Settings(_env_file=None)  # 忽略真实 .env，纯测 env
    assert s.vault_path == Path("/tmp/test_vault")
    assert s.embed_dim == 768          # str->int 自动转
    assert s.agent_api_key == "sk-dummy-agent"


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


def test_vlm_settings_from_env(monkeypatch):
    """P1-c：VLM_* 与 AGENT_* 完全独立，能从 env 加载且类型正确。

    不依赖真实 .env（避免泄露/耦合密钥），纯测字段契约。
    """
    monkeypatch.setenv("VLM_API_KEY", "sk-test-vlm")
    monkeypatch.setenv("VLM_BASE_URL", "https://api.example.cn/v1")
    monkeypatch.setenv("VLM_MODEL", "vlm-test")
    monkeypatch.setenv("IMAGE_UNDERSTAND_ENABLED", "false")
    monkeypatch.setenv("IMAGE_VLM_MAX_CALLS_PER_RUN", "100")
    monkeypatch.setenv("IMAGE_MAX_BYTES", "5242880")

    s = Settings(_env_file=None)
    assert s.vlm_api_key == "sk-test-vlm"
    assert s.vlm_base_url == "https://api.example.cn/v1"
    assert s.vlm_model == "vlm-test"
    # 护栏默认值可覆盖
    assert s.image_understand_enabled is False
    assert s.image_vlm_max_calls_per_run == 100
    assert s.image_max_bytes == 5242880
    # 与 AGENT_* 不串台：AGENT 字段不受影响（默认仍按 .env 之外无值，这里仅确认字段存在）
    assert isinstance(s.agent_api_key, str)


def test_vlm_defaults_off_when_unset(monkeypatch):
    """VLM_* 未配置时默认为空串 + 总开关默认开启。

    注意：config.py 导入时 load_dotenv 已把真实 .env 灌进 os.environ，
    pydantic-settings 即使 _env_file=None 仍读 os.environ，所以这里必须
    delenv 掉真实 VLM_*（同 test_env_override 的坑）。
    """
    for var in ("VLM_API_KEY", "VLM_BASE_URL", "VLM_MODEL", "IMAGE_UNDERSTAND_ENABLED"):
        monkeypatch.delenv(var, raising=False)
    s = Settings(_env_file=None)
    assert s.vlm_api_key == ""
    assert s.vlm_base_url == ""
    assert s.vlm_model == ""
    assert s.image_understand_enabled is False
    # v2 = 抗注入硬化版 prompt（L0-b）；bump 版本让旧 VisionCache 按既有机制失效
    assert s.vlm_prompt_version == "v2"