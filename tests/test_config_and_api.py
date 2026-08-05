from pathlib import Path

import pytest

from anime_gen import AnimePipeline, ControlMode, GenParams, GenResult
from anime_gen.config import DEFAULT_CONTROLNET_MODELS, load_config


def test_documented_public_api_is_exported():
    assert AnimePipeline is not None
    assert GenParams is not None
    assert GenResult is not None
    assert ControlMode.LINEART.value == "lineart"


def test_missing_explicit_config_fails_fast(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="配置文件不存在"):
        load_config(str(tmp_path / "missing.yaml"))


def test_builtin_defaults_include_controlnet_models(monkeypatch, tmp_path: Path):
    import anime_gen.config as config_module

    monkeypatch.delenv("ANIME_GEN_CONFIG", raising=False)
    monkeypatch.setattr(config_module, "__file__", str(tmp_path / "pkg" / "config.py"))
    config = config_module.load_config()
    assert config.models.controlnet == DEFAULT_CONTROLNET_MODELS
    assert config.models.base_model_id == "Yntec/AnythingV5"
    assert config.inference.device == "auto"
    assert config.inference.scheduler == "dpmpp_2m_karras"
    assert config.low_vram.enable_attention_slicing is False
    assert config.low_vram.enable_xformers is False


def test_invalid_config_section_has_clear_error(tmp_path: Path):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("inference: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inference"):
        load_config(str(config_path))
