"""
配置管理模块 — 从 YAML 文件加载并合并默认配置
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml  # type: ignore


DEFAULT_CONTROLNET_MODELS = {
    "pose": "lllyasviel/control_v11p_sd15_openpose",
    "lineart": "lllyasviel/control_v11p_sd15_lineart",
    "scribble": "lllyasviel/control_v11p_sd15_scribble",
    "canny": "lllyasviel/control_v11p_sd15_canny",
    "depth": "lllyasviel/control_v11p_sd15_depth",
    "softedge": "lllyasviel/control_v11p_sd15_softedge",
}


@dataclass
class ModelConfig:
    base_model_id: str = "andite/anything-v5.0"
    base_model_path: str = "./models/base"
    vae_model_id: Optional[str] = None
    controlnet: dict[str, str | None] = field(
        default_factory=lambda: DEFAULT_CONTROLNET_MODELS.copy()
    )
    controlnet_dir: str = "./models/controlnet"
    lora_dir: str = "./models/lora"


@dataclass
class InferenceConfig:
    device: str = "auto"
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

    explicit_path = config_path is not None
    if config_path is None:
        # 优先找环境变量, 其次找源码项目下的 configs/default.yaml。
        env_path = os.environ.get("ANIME_GEN_CONFIG")
        if env_path:
            config_path = env_path
            explicit_path = True
        else:
            config_path = str(
                Path(__file__).parent.parent.parent / "configs" / "default.yaml"
            )

    config_dict = {}
    config_file = Path(config_path) if config_path else None
    if config_file and config_file.exists():
        with config_file.open("r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}
    elif explicit_path and config_file:
        raise FileNotFoundError(f"配置文件不存在: {config_file}")

    # 递归合并到 dataclass
    return _dict_to_config(config_dict)


def _dict_to_config(d: dict) -> AppConfig:
    """将嵌套字典转为 AppConfig"""
    if not isinstance(d, dict):
        raise ValueError("配置文件根节点必须是 YAML mapping")

    def section(name: str) -> dict:
        value = d.get(name, {})
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError(f"配置项 '{name}' 必须是 YAML mapping")
        return value

    models_d = section("models")
    inference_d = section("inference")
    low_vram_d = section("low_vram")
    output_d = section("output")
    webui_d = section("webui")
    controlnet_d = models_d.get(
        "controlnet", DEFAULT_CONTROLNET_MODELS.copy()
    )
    if controlnet_d is None:
        controlnet_d = {}
    if not isinstance(controlnet_d, dict):
        raise ValueError("配置项 'models.controlnet' 必须是 YAML mapping")

    return AppConfig(
        models=ModelConfig(
            base_model_id=models_d.get("base_model_id", "andite/anything-v5.0"),
            base_model_path=models_d.get("base_model_path", "./models/base"),
            vae_model_id=models_d.get("vae_model_id"),
            controlnet=controlnet_d,
            controlnet_dir=models_d.get("controlnet_dir", "./models/controlnet"),
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
