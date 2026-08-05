"""
模型下载工具 — 一键下载所需的基础模型和 ControlNet 模型

用法:
    python -m anime_gen.download           # 下载全部
    python -m anime_gen.download --base-only          # 仅基础模型
    python -m anime_gen.download --controlnet pose,lineart  # 指定 ControlNet
"""

import argparse
import logging
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from .config import load_config, AppConfig

logger = logging.getLogger(__name__)

# 国内用户可以通过 HF 镜像加速
HF_ENDPOINT = "https://huggingface.co"
# HF_ENDPOINT = "https://hf-mirror.com"  # 国内镜像


def download_base_model(config: AppConfig):
    """下载 SD 基础模型"""
    model_id = config.models.base_model_id
    local_path = Path(config.models.base_model_path)

    if local_path.exists() and list(local_path.glob("*.safetensors")):
        logger.info(f"基础模型已存在: {local_path}")
        return

    logger.info(f"下载基础模型: {model_id} → {local_path}")
    local_path.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_path),
        endpoint=HF_ENDPOINT,
        ignore_patterns=["*.ckpt", "*.pt"],  # 只下 safetensors
        resume_download=True,
    )
    logger.info(f"✓ 基础模型下载完成: {local_path}")


def download_controlnet_models(config: AppConfig, modes: list[str] | None = None):
    """下载 ControlNet 模型"""
    cn_config = config.models.controlnet

    if modes is None:
        modes = [k for k, v in cn_config.items() if v]

    controlnet_dir = Path("./models/controlnet")

    for mode in modes:
        model_id = cn_config.get(mode)
        if model_id is None:
            logger.warning(f"跳过 {mode}: 未配置模型ID")
            continue

        local_path = controlnet_dir / mode
        if local_path.exists() and list(local_path.glob("*.safetensors")):
            logger.info(f"ControlNet [{mode}] 已存在: {local_path}")
            continue

        logger.info(f"下载 ControlNet [{mode}]: {model_id}")
        local_path.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_path),
            endpoint=HF_ENDPOINT,
            ignore_patterns=["*.ckpt", "*.pt"],
            resume_download=True,
        )
        logger.info(f"✓ ControlNet [{mode}] 下载完成")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="下载 Anime Gen Pipeline 所需模型")
    parser.add_argument("--base-only", action="store_true", help="仅下载基础模型")
    parser.add_argument(
        "--controlnet", type=str, default=None,
        help="指定 ControlNet 类型, 逗号分隔 (如: pose,lineart,canny)",
    )
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    args = parser.parse_args()

    config = load_config(args.config)

    print("=" * 60)
    print("  Anime Gen Pipeline — 模型下载")
    print("=" * 60)

    if args.base_only:
        download_base_model(config)
    elif args.controlnet:
        modes = [m.strip() for m in args.controlnet.split(",")]
        download_controlnet_models(config, modes)
    else:
        # 全部下载
        download_base_model(config)
        download_controlnet_models(config)

    print("\n✓ 所有模型下载完成!")
    print(f"  基础模型: {config.models.base_model_path}")
    print(f"  ControlNet: ./models/controlnet/")
    print(f"\n提示: 运行 'anime-gen-webui' 启动 Web 界面")


if __name__ == "__main__":
    main()
