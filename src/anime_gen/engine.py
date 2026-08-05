"""
模型引擎 — 负责加载/管理 Stable Diffusion + ControlNet 模型
针对 4-8GB VRAM 做了大量低配优化
"""

import gc
import logging
from pathlib import Path
from typing import Any, Optional

import torch

from .config import AppConfig, get_config

logger = logging.getLogger(__name__)


class ModelEngine:
    """
    模型引擎: 封装 SD 基础模型 + VAE + ControlNet 的加载与卸载。

    低配优化策略:
      - attention slicing: 切片计算注意力, ~30% VRAM 节省
      - VAE slicing/tiling: 分块编解码, 大图必备
      - CPU offload: 逐模块卸载到 CPU (极低 VRAM)
      - Token Merging (ToMe): 合并冗余 token 加速推理
      - xformers: 高效注意力实现
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_config()
        self._pipe: Any = None
        self._controlnet_models: dict[str, Any] = {}
        self._dtype = self._resolve_dtype()
        self._device = self.config.inference.device

    # ----------------------------------------------------------------
    # 公开 API
    # ----------------------------------------------------------------

    @property
    def pipe(self):
        """延迟加载 pipeline"""
        if self._pipe is None:
            self._load_base_pipeline()
        return self._pipe

    def load_controlnet(self, mode: str) -> Any:
        """
        按需加载单个 ControlNet 模型

        Args:
            mode: 'pose' | 'lineart' | 'scribble' | 'canny' | 'depth' | 'softedge'
        """
        if mode in self._controlnet_models:
            return self._controlnet_models[mode]

        from diffusers import ControlNetModel

        cn_config = self.config.models.controlnet
        model_id = cn_config.get(mode)
        if model_id is None:
            raise ValueError(
                f"ControlNet mode '{mode}' 未配置模型ID, "
                f"请在 configs/default.yaml 中设置 models.controlnet.{mode}"
            )

        logger.info(f"加载 ControlNet [{mode}]: {model_id}")
        cn = ControlNetModel.from_pretrained(
            model_id,
            torch_dtype=self._dtype,
            use_safetensors=True,
        ).to(self._device)

        self._controlnet_models[mode] = cn
        return cn

    def unload_all(self):
        """释放所有模型, 释放 VRAM"""
        logger.info("释放所有模型...")
        self._pipe = None
        self._controlnet_models.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def get_vram_info(self) -> str:
        """获取当前 VRAM 使用情况"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return f"VRAM: {allocated:.1f}G used / {reserved:.1f}G reserved / {total:.1f}G total"
        return "VRAM: N/A (CPU mode)"

    def get_torch_dtype_str(self) -> str:
        return self.config.inference.torch_dtype

    def get_device(self) -> str:
        return self._device

    # ----------------------------------------------------------------
    # 内部实现
    # ----------------------------------------------------------------

    def _resolve_dtype(self) -> torch.dtype:
        dtype_str = self.config.inference.torch_dtype
        mapping = {
            "fp16": torch.float16,
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
        }
        return mapping.get(dtype_str, torch.float16)

    def _load_base_pipeline(self):
        """加载 SD 1.5 基础 pipeline + 低配优化"""
        from diffusers import StableDiffusionPipeline

        model_id = self.config.models.base_model_id
        local_path = Path(self.config.models.base_model_path)

        logger.info(f"加载基础模型: {model_id}")

        # 优先本地, 其次 HF hub
        load_from = str(local_path) if local_path.exists() else model_id

        pipe = StableDiffusionPipeline.from_pretrained(
            load_from,
            torch_dtype=self._dtype,
            safety_checker=None,          # 动漫场景不需要
            requires_safety_checker=False,
            use_safetensors=True,
        )

        # --- 低配优化步骤 ---
        lv = self.config.low_vram

        if lv.enable_attention_slicing:
            pipe.enable_attention_slicing()
            logger.info("  ✓ attention slicing")

        if lv.enable_vae_slicing:
            pipe.enable_vae_slicing()
            logger.info("  ✓ VAE slicing")

        if lv.enable_vae_tiling:
            pipe.vae.enable_tiling()
            logger.info("  ✓ VAE tiling")

        if lv.enable_xformers and self._device.startswith("cuda"):
            try:
                pipe.enable_xformers_memory_efficient_attention()
                logger.info("  ✓ xformers")
            except Exception as e:
                logger.warning(f"  ✗ xformers 不可用: {e}")

        if lv.enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()
            logger.info("  ✓ model CPU offload")

        elif lv.enable_sequential_cpu_offload:
            pipe.enable_sequential_cpu_offload()
            logger.info("  ✓ sequential CPU offload")

        else:
            # 默认移到 GPU
            pipe = pipe.to(self._device)

        # Token Merging (可选)
        if lv.enable_tome:
            self._apply_tome(pipe, lv.tome_ratio)
            logger.info(f"  ✓ ToMe (ratio={lv.tome_ratio})")

        # 可选外部 VAE
        if self.config.models.vae_model_id:
            from diffusers import AutoencoderKL

            logger.info(f"加载外部 VAE: {self.config.models.vae_model_id}")
            vae = AutoencoderKL.from_pretrained(
                self.config.models.vae_model_id,
                torch_dtype=self._dtype,
            )
            pipe.vae = vae.to(self._device)

        self._pipe = pipe
        logger.info(f"模型加载完成 — {self.get_vram_info()}")

    @staticmethod
    def _apply_tome(pipe, ratio: float):
        """应用 Token Merging 加速"""
        try:
            import tomesd  # type: ignore

            tomesd.apply_patched(pipe, ratio=ratio)
        except ImportError:
            logger.warning("tomesd 未安装, 跳过 Token Merging")

    # xformers 由 low_vram.enable_xformers 配置控制

