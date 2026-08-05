"""
核心生成 Pipeline — 统一入口, 支持 txt2img / img2img / inpaint + ControlNet
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from PIL import Image, PngImagePlugin

from .config import AppConfig, get_config
from .controlnet import ControlMode, ControlNetProcessor
from .engine import ModelEngine

logger = logging.getLogger(__name__)


# ================================================================
# 数据类型
# ================================================================

@dataclass
class GenParams:
    """单次生成参数"""
    prompt: str = ""
    negative_prompt: str = (
        "lowres, bad anatomy, bad hands, text, error, missing fingers, "
        "extra digit, fewer digits, cropped, worst quality, low quality, "
        "normal quality, jpeg artifacts, signature, watermark, username, "
        "blurry, ugly, deformed"
    )

    # 尺寸
    width: int = 512
    height: int = 768

    # 推理
    steps: int = 20
    cfg_scale: float = 7.0
    seed: int = -1
    batch_size: int = 1

    # img2img
    init_image: Optional[Image.Image] = None
    strength: float = 0.75

    # ControlNet
    control_image: Optional[Image.Image] = None
    control_mode: Optional[ControlMode] = None
    control_weight: float = 0.8

    # Inpaint
    mask_image: Optional[Image.Image] = None

    # LoRA
    lora_path: Optional[str] = None
    lora_scale: float = 0.75

    # 输出
    output_dir: str = "./outputs"
    filename_prefix: str = "anime"
    save_metadata: bool = True


@dataclass
class GenResult:
    """生成结果"""
    images: list[Image.Image]
    params: GenParams
    elapsed: float           # 秒
    seed_used: int
    metadata: dict = field(default_factory=dict)


# ================================================================
# Pipeline
# ================================================================

class AnimePipeline:
    """
    动漫图像生成 Pipeline 主类

    使用示例:
        pipe = AnimePipeline()
        result = pipe.txt2img(GenParams(prompt="1girl, sunset, masterpiece"))
        result.images[0].save("output.png")
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_config()
        self.engine = ModelEngine(self.config)
        self.cn_processor: Optional[ControlNetProcessor] = None
        logger.info("AnimePipeline 初始化完成")

    # ----------------------------------------------------------------
    # 生成模式
    # ----------------------------------------------------------------

    def txt2img(self, params: GenParams) -> GenResult:
        """文生图"""
        return self._generate(params, mode="txt2img")

    def img2img(self, params: GenParams) -> GenResult:
        """图生图 (需设置 init_image + strength)"""
        if params.init_image is None:
            raise ValueError("img2img 需要提供 init_image")
        return self._generate(params, mode="img2img")

    def inpaint(self, params: GenParams) -> GenResult:
        """局部重绘 (需设置 init_image + mask_image)"""
        if params.init_image is None or params.mask_image is None:
            raise ValueError("inpaint 需要提供 init_image 和 mask_image")
        return self._generate(params, mode="inpaint")

    def generate_with_control(
        self,
        params: GenParams,
        control_image: Image.Image,
        control_mode: ControlMode | str = ControlMode.LINEART,
    ) -> GenResult:
        """
        使用 ControlNet 条件生成

        Args:
            params:         生成参数
            control_image:  条件输入图 (如照片、草图)
            control_mode:   控制模式

        工作流程:
            输入图 → 预处理器提取条件图 → ControlNet → SD 生成
        """
        if isinstance(control_mode, str):
            control_mode = ControlMode(control_mode)

        # 预处理: 将输入图转为条件图
        processor = self._get_cn_processor()
        condition = processor.process(control_image, control_mode)

        params.control_image = condition
        params.control_mode = control_mode

        return self._generate(params, mode="txt2img")

    # ----------------------------------------------------------------
    # 核心生成逻辑
    # ----------------------------------------------------------------

    def _generate(self, params: GenParams, mode: str) -> GenResult:
        t0 = time.time()

        # 1. 种子
        seed = params.seed if params.seed >= 0 else torch.randint(0, 2**31 - 1, (1,)).item()
        generator = torch.Generator(device=self.engine.get_device()).manual_seed(seed)

        # 2. LoRA (按需加载/卸载)
        lora_loaded = False
        if params.lora_path and Path(params.lora_path).exists():
            self.engine.pipe.load_lora_weights(params.lora_path)
            lora_loaded = True

        # 3. ControlNet 配置
        controlnet_condition = None
        if params.control_image is not None and params.control_mode is not None:
            cn_model = self.engine.load_controlnet(params.control_mode.value)
            controlnet_condition = self._prepare_controlnet(
                params.control_image, params.control_weight
            )

        # 4. 调度器参数
        self._set_scheduler_params(params.steps)

        # 5. 生成
        try:
            images = self._run_inference(params, mode, generator, controlnet_condition)
        finally:
            # 卸载 LoRA (避免影响后续生成)
            if lora_loaded:
                try:
                    self.engine.pipe.unload_lora_weights()
                except Exception:
                    pass

        # 6. 保存
        saved_paths = self._save_images(images, params, seed)

        elapsed = time.time() - t0
        logger.info(
            f"生成完成 — {len(images)} 张, "
            f"{elapsed:.1f}s, seed={seed}, "
            f"{self.engine.get_vram_info()}"
        )

        return GenResult(
            images=images,
            params=params,
            elapsed=elapsed,
            seed_used=seed,
            metadata={"saved_paths": saved_paths},
        )

    def _run_inference(
        self,
        params: GenParams,
        mode: str,
        generator: torch.Generator,
        controlnet_condition: Optional[tuple],
    ) -> list[Image.Image]:
        """执行实际推理"""
        pipe = self.engine.pipe
        common = dict(
            prompt=params.prompt,
            negative_prompt=params.negative_prompt,
            num_inference_steps=params.steps,
            guidance_scale=params.cfg_scale,
            generator=generator,
            num_images_per_prompt=params.batch_size,
            width=params.width,
            height=params.height,
        )

        if mode == "txt2img":
            if controlnet_condition:
                cn_image, cn_scale = controlnet_condition
                result = pipe(image=cn_image, **common)
            else:
                result = pipe(**common)

        elif mode == "img2img":
            result = pipe(
                image=params.init_image,
                strength=params.strength,
                **common,
            )

        elif mode == "inpaint":
            result = pipe(
                image=params.init_image,
                mask_image=params.mask_image,
                strength=params.strength,
                **common,
            )
        else:
            raise ValueError(f"未知模式: {mode}")

        return result.images

    def _prepare_controlnet(
        self, control_image: Image.Image, weight: float
    ) -> tuple:
        """准备 ControlNet 条件输入"""
        # 如果 pipe 还没有设置 controlnet, 需要设置
        cn = self.engine._controlnet_models.get(
            list(self.engine._controlnet_models.keys())[0]
            if self.engine._controlnet_models else None
        )
        return control_image, weight

    def _set_scheduler_params(self, steps: int):
        """设置调度器总步数"""
        pipe = self.engine.pipe
        if hasattr(pipe.scheduler, "set_timesteps"):
            pipe.scheduler.set_timesteps(steps, device=self.engine.get_device())

    # ----------------------------------------------------------------
    # 保存
    # ----------------------------------------------------------------

    def _save_images(
        self,
        images: list[Image.Image],
        params: GenParams,
        seed: int,
    ) -> list[str]:
        """保存图片到磁盘"""
        output_dir = Path(params.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved = []
        fmt = self.config.output.format

        for i, img in enumerate(images):
            filename = f"{params.filename_prefix}_{seed}_{i:02d}.{fmt}"
            filepath = output_dir / filename

            if params.save_metadata and fmt.lower() == "png":
                # 嵌入生成参数到 PNG 元数据
                meta = PngImagePlugin.PngInfo()
                meta.add_text("parameters", self._build_metadata_string(params, seed))
                img.save(filepath, pnginfo=meta)
            else:
                img.save(filepath)

            saved.append(str(filepath))

        return saved

    @staticmethod
    def _build_metadata_string(params: GenParams, seed: int) -> str:
        """构造类似 A1111 的 metadata 字符串"""
        lines = [
            params.prompt,
            f"Negative prompt: {params.negative_prompt}",
            f"Steps: {params.steps}, CFG scale: {params.cfg_scale}",
            f"Seed: {seed}, Size: {params.width}x{params.height}",
        ]
        if params.control_mode:
            lines.append(f"ControlNet: {params.control_mode.value}")
        return "\n".join(lines)

    # ----------------------------------------------------------------
    # 辅助
    # ----------------------------------------------------------------

    def _get_cn_processor(self) -> ControlNetProcessor:
        if self.cn_processor is None:
            self.cn_processor = ControlNetProcessor(device=self.engine.get_device())
        return self.cn_processor

    def cleanup(self):
        """释放所有资源"""
        self.engine.unload_all()
        if self.cn_processor is not None:
            self.cn_processor._preprocessors.clear()
