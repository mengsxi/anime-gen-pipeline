"""
模型引擎 — 负责加载和复用 Stable Diffusion、任务 Pipeline 与 ControlNet。
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, Optional

import torch

from .config import AppConfig, get_config

logger = logging.getLogger(__name__)


class ModelEngine:
    """按需创建任务 Pipeline，并共享基础模型组件。"""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_config()
        self._pipe: Any = None
        self._task_pipes: dict[str, Any] = {}
        self._control_pipe: Any = None
        self._controlnet_model: Any = None
        self._controlnet_mode: Optional[str] = None
        self._device = self._resolve_device()
        self._dtype = self._resolve_dtype()

    @property
    def pipe(self):
        """延迟加载文生图 Pipeline，保留原有公开属性。"""
        if self._pipe is None:
            self._load_base_pipeline()
        return self._pipe

    def get_pipeline(self, mode: str = "txt2img", control_mode: str | None = None):
        """返回与任务匹配的 Pipeline，模型组件不会重复分配。"""
        if control_mode is not None:
            if mode != "txt2img":
                raise ValueError("当前仅支持 txt2img + ControlNet")
            return self._get_controlnet_pipeline(control_mode)

        if mode == "txt2img":
            return self.pipe
        if mode in self._task_pipes:
            return self._task_pipes[mode]

        from diffusers import AutoPipelineForImage2Image, AutoPipelineForInpainting

        factories = {
            "img2img": AutoPipelineForImage2Image,
            "inpaint": AutoPipelineForInpainting,
        }
        try:
            factory = factories[mode]
        except KeyError as exc:
            raise ValueError(f"未知模式: {mode}") from exc

        logger.info("创建任务 Pipeline: %s", mode)
        task_pipe = factory.from_pipe(self.pipe)
        self._reapply_offload(task_pipe)
        self._task_pipes[mode] = task_pipe
        return task_pipe

    def load_controlnet(self, mode: str) -> Any:
        """只保留当前 ControlNet，切换模式时释放上一套权重。"""
        if self._controlnet_mode == mode and self._controlnet_model is not None:
            return self._controlnet_model

        self._release_controlnet()

        from diffusers import ControlNetModel

        model_id = self.config.models.controlnet.get(mode)
        if not model_id:
            raise ValueError(
                f"ControlNet mode '{mode}' 未配置模型 ID，请设置 "
                f"models.controlnet.{mode}"
            )

        local_path = Path(self.config.models.controlnet_dir) / mode
        load_from = (
            str(local_path)
            if self._model_dir_ready(local_path, "config.json")
            else model_id
        )
        logger.info("加载 ControlNet [%s]: %s", mode, load_from)
        controlnet = ControlNetModel.from_pretrained(
            load_from,
            torch_dtype=self._dtype,
            # 部分官方 SD1.5 ControlNet 仅提供 .bin，None 会优先
            # safetensors 并在缺失时安全回退。
            use_safetensors=None,
        )

        lv = self.config.low_vram
        if not (lv.enable_model_cpu_offload or lv.enable_sequential_cpu_offload):
            controlnet = controlnet.to(self._device)

        self._controlnet_model = controlnet
        self._controlnet_mode = mode
        return controlnet

    def unload_all(self):
        """释放所有模型和任务包装器。"""
        logger.info("释放所有模型...")
        self._task_pipes.clear()
        self._release_controlnet(empty_cache=False)
        self._pipe = None
        gc.collect()
        self._empty_cuda_cache()

    def get_vram_info(self) -> str:
        """获取当前 CUDA 设备的显存使用情况。"""
        if self._device.startswith("cuda") and torch.cuda.is_available():
            device = torch.device(self._device)
            index = (
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
            allocated = torch.cuda.memory_allocated(index) / 1024**3
            reserved = torch.cuda.memory_reserved(index) / 1024**3
            total = torch.cuda.get_device_properties(index).total_memory / 1024**3
            return (
                f"VRAM: {allocated:.1f}G used / {reserved:.1f}G reserved / "
                f"{total:.1f}G total"
            )
        return f"VRAM: N/A ({self._device} mode)"

    def get_torch_dtype_str(self) -> str:
        mapping = {
            torch.float16: "fp16",
            torch.float32: "fp32",
            torch.bfloat16: "bf16",
        }
        return mapping[self._dtype]

    def get_device(self) -> str:
        return self._device

    def get_generator_device(self) -> str:
        # MPS 使用 CPU Generator 可获得更稳定的可复现行为。
        return "cpu" if self._device == "mps" else self._device

    def _get_controlnet_pipeline(self, mode: str):
        if self._control_pipe is not None and self._controlnet_mode == mode:
            return self._control_pipe

        from diffusers import StableDiffusionControlNetPipeline

        controlnet = self.load_controlnet(mode)
        control_pipe = StableDiffusionControlNetPipeline.from_pipe(
            self.pipe,
            controlnet=controlnet,
        )

        lv = self.config.low_vram
        self._reapply_offload(control_pipe)

        if lv.enable_xformers and self._device.startswith("cuda"):
            try:
                control_pipe.enable_xformers_memory_efficient_attention()
            except Exception as exc:  # xformers 是可选加速项
                logger.warning("ControlNet xformers 不可用: %s", exc)

        self._control_pipe = control_pipe
        return control_pipe

    def _resolve_device(self) -> str:
        requested = self.config.inference.device.lower()
        if requested == "auto":
            if torch.cuda.is_available():
                return "cuda"
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return "mps"
            return "cpu"

        if requested.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA 不可用，自动回退到 CPU")
            return "cpu"
        if requested == "mps":
            mps = getattr(torch.backends, "mps", None)
            if mps is None or not mps.is_available():
                logger.warning("MPS 不可用，自动回退到 CPU")
                return "cpu"
        if (
            requested != "cpu"
            and requested != "mps"
            and not requested.startswith("cuda")
        ):
            raise ValueError(f"不支持的推理设备: {requested}")
        return requested

    def _resolve_dtype(self) -> torch.dtype:
        dtype_str = self.config.inference.torch_dtype.lower()
        mapping = {
            "fp16": torch.float16,
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
        }
        if dtype_str not in mapping:
            raise ValueError(f"不支持的 torch_dtype: {dtype_str}")

        dtype = mapping[dtype_str]
        if self._device == "cpu" and dtype == torch.float16:
            logger.warning("CPU 不适合 fp16 推理，自动使用 fp32")
            return torch.float32
        if self._device == "mps" and dtype == torch.bfloat16:
            logger.warning("MPS 不支持 bf16，自动使用 fp16")
            return torch.float16
        return dtype

    def _load_base_pipeline(self):
        """加载 SD 基础 Pipeline，再统一应用运行时优化。"""
        from diffusers import StableDiffusionPipeline

        model_id = self.config.models.base_model_id
        local_path = Path(self.config.models.base_model_path)
        load_from = (
            str(local_path)
            if self._model_dir_ready(local_path, "model_index.json")
            else model_id
        )
        logger.info("加载基础模型: %s", load_from)

        load_kwargs = {
            "torch_dtype": self._dtype,
            "use_safetensors": self.config.models.use_safetensors,
        }
        if self.config.models.variant:
            load_kwargs["variant"] = self.config.models.variant
        if not self.config.inference.enable_safety_checker:
            load_kwargs.update(
                safety_checker=None,
                requires_safety_checker=False,
            )
        pipe = StableDiffusionPipeline.from_pretrained(load_from, **load_kwargs)

        if self.config.models.vae_model_id:
            from diffusers import AutoencoderKL

            logger.info("加载外部 VAE: %s", self.config.models.vae_model_id)
            pipe.vae = AutoencoderKL.from_pretrained(
                self.config.models.vae_model_id,
                torch_dtype=self._dtype,
                use_safetensors=self.config.models.use_safetensors,
            )

        self._configure_scheduler(pipe)
        self._apply_runtime_optimizations(pipe)
        self._pipe = pipe
        logger.info("模型加载完成 — %s", self.get_vram_info())

    def _apply_runtime_optimizations(self, pipe):
        lv = self.config.low_vram

        if lv.enable_vae_slicing:
            pipe.enable_vae_slicing()
        if lv.enable_vae_tiling:
            pipe.vae.enable_tiling()

        xformers_enabled = False
        if lv.enable_xformers and self._device.startswith("cuda"):
            try:
                pipe.enable_xformers_memory_efficient_attention()
                xformers_enabled = True
            except Exception as exc:
                logger.warning("xformers 不可用，继续使用 PyTorch attention: %s", exc)
        if lv.enable_attention_slicing and not xformers_enabled:
            pipe.enable_attention_slicing()

        if lv.enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()
        elif lv.enable_sequential_cpu_offload:
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to(self._device)

        if lv.enable_tome:
            self._apply_tome(pipe, lv.tome_ratio)

    def _reapply_offload(self, pipe):
        """from_pipe 后重新安装与新执行顺序匹配的 offload hooks。"""
        lv = self.config.low_vram
        if lv.enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()
        elif lv.enable_sequential_cpu_offload:
            pipe.enable_sequential_cpu_offload()

    def _configure_scheduler(self, pipe):
        scheduler_name = self.config.inference.scheduler.lower()
        if scheduler_name in {"model", "default"}:
            return

        from diffusers import (
            DDIMScheduler,
            DPMSolverMultistepScheduler,
            EulerAncestralDiscreteScheduler,
        )

        scheduler_config = pipe.scheduler.config
        if scheduler_name == "dpmpp_2m_karras":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                scheduler_config,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=True,
            )
        elif scheduler_name == "euler_a":
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                scheduler_config
            )
        elif scheduler_name == "ddim":
            pipe.scheduler = DDIMScheduler.from_config(scheduler_config)
        else:
            raise ValueError(f"不支持的 scheduler: {scheduler_name}")
        logger.info("使用调度器: %s", scheduler_name)

    def _release_controlnet(self, empty_cache: bool = True):
        if self._controlnet_model is not None and self._device.startswith("cuda"):
            try:
                self._controlnet_model.to("cpu")
            except Exception:
                logger.debug("ControlNet 移回 CPU 失败，将直接释放引用", exc_info=True)
        self._control_pipe = None
        self._controlnet_model = None
        self._controlnet_mode = None
        gc.collect()
        if empty_cache:
            self._empty_cuda_cache()

    @staticmethod
    def _apply_tome(pipe, ratio: float):
        try:
            import tomesd  # type: ignore

            tomesd.apply_patch(pipe, ratio=ratio)
        except ImportError:
            logger.warning("tomesd 未安装，跳过 Token Merging")

    @staticmethod
    def _empty_cuda_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _model_dir_ready(path: Path, marker: str) -> bool:
        weights_root = path / "unet" if marker == "model_index.json" else path
        has_weights = any(weights_root.rglob("*.safetensors")) or any(
            weights_root.rglob("*.bin")
        )
        return (path / marker).exists() and has_weights
