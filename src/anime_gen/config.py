"""
配置管理模块 — 从 YAML 文件加载并合并默认配置
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml  # type: ignore


@dataclass
class ModelConfig:
    base_model_id: str = "andite/anything-v5.0"
    base_model_path: str = "./models/base"
    vae_model_id: Optional[str] = None
    controlnet: dict = field(default_factory=dict)
    lora_dir: str = "./models/lora"


@dataclass
class InferenceConfig:
    device: str = "cuda"
    torch_dtype: str = "fp16"
    default_steps: int = 20
    default_cfg: float = 7.0
    default_width: int = 512
    default_height: int = 768
    default_batch_size: int = 1
    default_seed: int = -1


@dataclass
class LowVRAMConfig:
    enable_attention_slicing: bool = True
    enable_vae_slicing: bool = True
    enable_vae_tiling: bool = True
    enable_model_cpu_offload: bool = False
    enable_sequential_cpu_offload: bool = False
    enable_xformers: bool = True
    enable_tome: bool = False
    tome_ratio: float = 0.5


@dataclass
class OutputConfig:
    dir: str = "./outputs"
    format: str = "png"
    save_metadata: bool = True


@dataclass
class WebUIConfig:
    host: str = "127.0.0.1"
    port: int = 7860
    share: bool = False
    theme: str = "soft"


@dataclass
class AppConfig:
    models: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    low_vram: LowVRAMConfig = field(default_factory=LowVRAMConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    webui: WebUIConfig = field(default_factory=WebUIConfig)


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """加载配置文件, 未指定的字段使用默认值"""
    import os

    if config_path is None:
        # 优先找环境变量, 其次找项目下的 configs/default.yaml
        config_path = os.environ.get(
            "ANIME_GEN_CONFIG",
            str(Path(__file__).parent.parent.parent / "configs" / "default.yaml"),
        )

    config_dict = {}
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}

    # 递归合并到 dataclass
    return _dict_to_config(config_dict)


def _dict_to_config(d: dict) -> AppConfig:
    """将嵌套字典转为 AppConfig"""
    models_d = d.get("models", {})
    inference_d = d.get("inference", {})
    low_vram_d = d.get("low_vram", {})
    output_d = d.get("output", {})
    webui_d = d.get("webui", {})

    return AppConfig(
        models=ModelConfig(
            base_model_id=models_d.get("base_model_id", "andite/anything-v5.0"),
            base_model_path=models_d.get("base_model_path", "./models/base"),
            vae_model_id=models_d.get("vae_model_id"),
            controlnet=models_d.get("controlnet", {}),
            lora_dir=models_d.get("lora_dir", "./models/lora"),
        ),
        inference=InferenceConfig(**inference_d) if inference_d else InferenceConfig(),
        low_vram=LowVRAMConfig(**low_vram_d) if low_vram_d else LowVRAMConfig(),
        output=OutputConfig(**output_d) if output_d else OutputConfig(),
        webui=WebUIConfig(**webui_d) if webui_d else WebUIConfig(),
    )


# 全局单例
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config
