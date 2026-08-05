"""
模型下载工具 — 默认下载基础模型，ControlNet 按需选择。

用法:
    python -m anime_gen.download                    # 仅基础模型
    python -m anime_gen.download --all-controlnet   # 基础模型 + 全部 ControlNet
    python -m anime_gen.download --controlnet pose,lineart  # 指定 ControlNet
"""

import argparse
import fnmatch
import logging
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from .config import AppConfig, load_config

logger = logging.getLogger(__name__)
COMMON_BASE_PATTERNS = [
    "model_index.json",
    "feature_extractor/*",
    "scheduler/*",
    "tokenizer/*",
    "*/config.json",
]
CONTROLNET_PATTERNS = ["config.json", "diffusion_pytorch_model.*"]


def _hub_kwargs() -> dict[str, str]:
    """尊重 huggingface_hub 的 HF_ENDPOINT 环境变量。"""
    endpoint = os.environ.get("HF_ENDPOINT")
    return {"endpoint": endpoint} if endpoint else {}


def _model_dir_ready(path: Path, marker: str) -> bool:
    weights_root = path / "unet" if marker == "model_index.json" else path
    has_weights = any(weights_root.rglob("*.safetensors")) or any(
        weights_root.rglob("*.bin")
    )
    return (path / marker).exists() and has_weights


def _base_allow_patterns(config: AppConfig) -> list[str]:
    variant = config.models.variant
    weights = f"*/*.{variant}.safetensors" if variant else "*/*.safetensors"
    return [*COMMON_BASE_PATTERNS, weights]


def _estimate_download_size(repo_id: str, allow_patterns: list[str]) -> int | None:
    try:
        endpoint = os.environ.get("HF_ENDPOINT")
        info = HfApi(endpoint=endpoint).model_info(repo_id, files_metadata=True)
        return sum(
            sibling.size or 0
            for sibling in info.siblings
            if any(
                fnmatch.fnmatch(sibling.rfilename, pattern)
                for pattern in allow_patterns
            )
        )
    except Exception as exc:
        logger.warning("无法预估 %s 下载体积: %s", repo_id, exc)
        return None


def _log_download_size(repo_id: str, allow_patterns: list[str]):
    size = _estimate_download_size(repo_id, allow_patterns)
    if size is not None:
        logger.info("预计下载体积: %.2f GiB", size / 1024**3)


def download_base_model(config: AppConfig):
    """下载 SD 基础模型"""
    model_id = config.models.base_model_id
    local_path = Path(config.models.base_model_path)

    if _model_dir_ready(local_path, "model_index.json"):
        logger.info(f"基础模型已存在: {local_path}")
        return

    logger.info(f"下载基础模型: {model_id} → {local_path}")
    local_path.mkdir(parents=True, exist_ok=True)
    allow_patterns = _base_allow_patterns(config)
    _log_download_size(model_id, allow_patterns)

    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_path),
        allow_patterns=allow_patterns,
        **_hub_kwargs(),
    )
    logger.info(f"✓ 基础模型下载完成: {local_path}")


def download_controlnet_models(config: AppConfig, modes: list[str] | None = None):
    """下载 ControlNet 模型"""
    cn_config = config.models.controlnet

    if modes is None:
        modes = [k for k, v in cn_config.items() if v]

    controlnet_dir = Path(config.models.controlnet_dir)

    for mode in modes:
        model_id = cn_config.get(mode)
        if model_id is None:
            available = ", ".join(sorted(cn_config))
            raise ValueError(f"未知或未配置的 ControlNet: {mode}；可选: {available}")

        local_path = controlnet_dir / mode
        if _model_dir_ready(local_path, "config.json"):
            logger.info(f"ControlNet [{mode}] 已存在: {local_path}")
            continue

        logger.info(f"下载 ControlNet [{mode}]: {model_id}")
        local_path.mkdir(parents=True, exist_ok=True)
        _log_download_size(model_id, CONTROLNET_PATTERNS)

        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_path),
            allow_patterns=CONTROLNET_PATTERNS,
            **_hub_kwargs(),
        )
        logger.info(f"✓ ControlNet [{mode}] 下载完成")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="下载 Anime Gen Pipeline 所需模型")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--base-only", action="store_true", help="仅下载基础模型（默认）"
    )
    group.add_argument(
        "--controlnet",
        type=str,
        default=None,
        help="指定 ControlNet 类型, 逗号分隔 (如: pose,lineart,canny)",
    )
    group.add_argument(
        "--all-controlnet",
        action="store_true",
        help="下载全部已配置 ControlNet（通常超过 10 GiB）",
    )
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    args = parser.parse_args()

    config = load_config(args.config)

    print("=" * 60)
    print("  Anime Gen Pipeline — 模型下载")
    print("=" * 60)

    if args.controlnet:
        modes = [m.strip() for m in args.controlnet.split(",")]
        download_controlnet_models(config, modes)
    elif args.all_controlnet:
        download_base_model(config)
        download_controlnet_models(config)
    else:
        download_base_model(config)

    print("\n✓ 所有模型下载完成!")
    print(f"  基础模型: {config.models.base_model_path}")
    print(f"  ControlNet: {config.models.controlnet_dir}")
    print("\n提示: 运行 'anime-gen-webui' 启动 Web 界面")


if __name__ == "__main__":
    main()
