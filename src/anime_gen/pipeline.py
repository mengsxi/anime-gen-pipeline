"""统一的动漫图像生成入口。"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image, PngImagePlugin

from .config import AppConfig, get_config
from .controlnet import ControlMode, ControlNetProcessor
from .engine import ModelEngine

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[int, int], None]


DEFAULT_NEGATIVE_PROMPT = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, "
    "extra digit, fewer digits, cropped, worst quality, low quality, "
    "normal quality, jpeg artifacts, signature, watermark, username, "
    "blurry, ugly, deformed"
)


@dataclass
class GenParams:
    """单次生成参数；为 ``None`` 的通用参数会从 AppConfig 读取。"""

    prompt: str = ""
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT

    width: Optional[int] = None
    height: Optional[int] = None
    steps: Optional[int] = None
    cfg_scale: Optional[float] = None
    seed: Optional[int] = None
    batch_size: Optional[int] = None
    clip_skip: Optional[int] = None

    init_image: Optional[Image.Image] = None
    strength: float = 0.75

    control_image: Optional[Image.Image] = None
    control_mode: Optional[ControlMode] = None
    control_weight: float = 0.8

    mask_image: Optional[Image.Image] = None

    lora_path: Optional[str] = None
    lora_scale: float = 0.75

    output_dir: Optional[str] = None
    filename_prefix: str = "anime"
    save_metadata: Optional[bool] = None


@dataclass
class GenResult:
    """生成结果。"""

    images: list[Image.Image]
    params: GenParams
    elapsed: float
    seed_used: int
    metadata: dict = field(default_factory=dict)


class AnimePipeline:
    """支持 txt2img、img2img、inpaint 和单 ControlNet 的生成入口。"""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_config()
        self.engine = ModelEngine(self.config)
        self.cn_processor: Optional[ControlNetProcessor] = None
        self._lock = threading.RLock()

    def txt2img(
        self,
        params: GenParams,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> GenResult:
        return self._generate(
            params, mode="txt2img", progress_callback=progress_callback
        )

    def img2img(
        self,
        params: GenParams,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> GenResult:
        if params.init_image is None:
            raise ValueError("img2img 需要提供 init_image")
        return self._generate(
            params, mode="img2img", progress_callback=progress_callback
        )

    def inpaint(
        self,
        params: GenParams,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> GenResult:
        if params.init_image is None or params.mask_image is None:
            raise ValueError("inpaint 需要提供 init_image 和 mask_image")
        return self._generate(
            params, mode="inpaint", progress_callback=progress_callback
        )

    def generate_with_control(
        self,
        params: GenParams,
        control_image: Image.Image,
        control_mode: ControlMode | str = ControlMode.LINEART,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> GenResult:
        with self._lock:
            if isinstance(control_mode, str):
                control_mode = ControlMode(control_mode)

            condition = self._get_cn_processor().process(control_image, control_mode)
            resolved = replace(
                params,
                control_image=condition,
                control_mode=control_mode,
            )
            return self._generate(
                resolved,
                mode="txt2img",
                progress_callback=progress_callback,
            )

    def _generate(
        self,
        params: GenParams,
        mode: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> GenResult:
        with self._lock:
            return self._generate_locked(params, mode, progress_callback)

    def _generate_locked(
        self,
        params: GenParams,
        mode: str,
        progress_callback: Optional[ProgressCallback],
    ) -> GenResult:
        t0 = time.perf_counter()
        params = self._resolve_params(params)
        self._validate_params(params, mode)

        seed = (
            params.seed
            if params.seed is not None and params.seed >= 0
            else torch.randint(0, 2**31 - 1, (1,)).item()
        )
        generator = torch.Generator(
            device=self.engine.get_generator_device()
        ).manual_seed(seed)

        control_mode = params.control_mode.value if params.control_mode else None
        pipe = self.engine.get_pipeline(mode=mode, control_mode=control_mode)

        lora_loaded = False
        lora_scale_via_kwargs = False
        if params.lora_path:
            lora_path = Path(params.lora_path)
            if not lora_path.is_file():
                raise FileNotFoundError(f"LoRA 文件不存在: {lora_path}")
            pipe.load_lora_weights(
                str(lora_path.parent),
                weight_name=lora_path.name,
                adapter_name="request",
            )
            if hasattr(pipe, "set_adapters"):
                pipe.set_adapters("request", adapter_weights=params.lora_scale)
            else:
                lora_scale_via_kwargs = True
            lora_loaded = True

        try:
            images = self._run_inference(
                pipe,
                params,
                mode,
                generator,
                lora_scale_via_kwargs=lora_scale_via_kwargs,
                progress_callback=progress_callback,
            )
        finally:
            if lora_loaded:
                try:
                    pipe.unload_lora_weights()
                except Exception:
                    logger.exception("LoRA 卸载失败，将释放 Pipeline 防止状态污染")
                    self.engine.unload_all()

        saved_paths = self._save_images(images, params, seed)
        elapsed = time.perf_counter() - t0
        logger.info(
            "生成完成 — %s 张, %.1fs, seed=%s, %s",
            len(images),
            elapsed,
            seed,
            self.engine.get_vram_info(),
        )
        return GenResult(
            images=images,
            params=params,
            elapsed=elapsed,
            seed_used=seed,
            metadata={
                "saved_paths": saved_paths,
                "model": self.config.models.base_model_id,
                "scheduler": self.config.inference.scheduler,
                "clip_skip": params.clip_skip,
            },
        )

    @staticmethod
    def _run_inference(
        pipe,
        params: GenParams,
        mode: str,
        generator: torch.Generator,
        *,
        lora_scale_via_kwargs: bool,
        progress_callback: Optional[ProgressCallback],
    ) -> list[Image.Image]:
        common = {
            "prompt": params.prompt,
            "negative_prompt": params.negative_prompt,
            "num_inference_steps": params.steps,
            "guidance_scale": params.cfg_scale,
            "generator": generator,
            "num_images_per_prompt": params.batch_size,
            "width": params.width,
            "height": params.height,
        }
        if params.clip_skip is not None:
            common["clip_skip"] = params.clip_skip
        if progress_callback is not None:

            def on_step_end(_pipe, step_index, _timestep, callback_kwargs):
                progress_callback(step_index + 1, params.steps or 1)
                return callback_kwargs

            common["callback_on_step_end"] = on_step_end
        # 较早版本没有 set_adapters 时，退回 cross_attention_kwargs。
        if lora_scale_via_kwargs:
            common["cross_attention_kwargs"] = {"scale": params.lora_scale}

        if params.control_image is not None and params.control_mode is not None:
            result = pipe(
                image=params.control_image,
                controlnet_conditioning_scale=params.control_weight,
                **common,
            )
        elif mode == "txt2img":
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

    def _resolve_params(self, params: GenParams) -> GenParams:
        inference = self.config.inference
        output = self.config.output
        return replace(
            params,
            width=params.width if params.width is not None else inference.default_width,
            height=(
                params.height if params.height is not None else inference.default_height
            ),
            steps=params.steps if params.steps is not None else inference.default_steps,
            cfg_scale=(
                params.cfg_scale
                if params.cfg_scale is not None
                else inference.default_cfg
            ),
            seed=params.seed if params.seed is not None else inference.default_seed,
            batch_size=(
                params.batch_size
                if params.batch_size is not None
                else inference.default_batch_size
            ),
            clip_skip=(
                params.clip_skip
                if params.clip_skip is not None
                else inference.clip_skip
            ),
            output_dir=(
                params.output_dir if params.output_dir is not None else output.dir
            ),
            save_metadata=(
                params.save_metadata
                if params.save_metadata is not None
                else output.save_metadata
            ),
        )

    def _validate_params(self, params: GenParams, mode: str):
        if mode not in {"txt2img", "img2img", "inpaint"}:
            raise ValueError(f"未知模式: {mode}")
        if not isinstance(params.prompt, str) or not params.prompt.strip():
            raise ValueError("prompt 不能为空")
        if params.width is None or params.height is None:
            raise ValueError("width 和 height 必须已解析")
        if params.width <= 0 or params.height <= 0:
            raise ValueError("width 和 height 必须大于 0")
        if params.width % 8 or params.height % 8:
            raise ValueError("width 和 height 必须是 8 的倍数")
        if params.steps is None or params.steps <= 0:
            raise ValueError("steps 必须大于 0")
        if params.batch_size is None or params.batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        if params.cfg_scale is None or params.cfg_scale < 0:
            raise ValueError("cfg_scale 不能为负数")
        if params.clip_skip is not None and params.clip_skip < 1:
            raise ValueError("clip_skip 必须大于等于 1 或设为 null")
        if not 0 <= params.strength <= 1:
            raise ValueError("strength 必须在 0 到 1 之间")
        if params.control_weight < 0:
            raise ValueError("control_weight 不能为负数")
        if params.lora_scale < 0:
            raise ValueError("lora_scale 不能为负数")
        if params.seed is not None and params.seed < -1:
            raise ValueError("seed 必须为 -1 或非负整数")
        if not params.filename_prefix or params.filename_prefix in {".", ".."}:
            raise ValueError("filename_prefix 不能为空或路径标记")
        if "/" in params.filename_prefix or "\\" in params.filename_prefix:
            raise ValueError("filename_prefix 不能包含路径分隔符")

        fmt = self.config.output.format.lower().lstrip(".")
        if fmt not in {"png", "jpg", "jpeg", "webp"}:
            raise ValueError(f"不支持的输出格式: {fmt}")

    def _save_images(
        self,
        images: list[Image.Image],
        params: GenParams,
        seed: int,
    ) -> list[str]:
        if params.output_dir is None:
            raise ValueError("output_dir 必须已解析")
        output_dir = Path(params.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fmt = self.config.output.format.lower().lstrip(".")

        saved = []
        for index, image in enumerate(images):
            stem = f"{params.filename_prefix}_{seed}_{index:02d}"
            filepath = self._next_available_path(output_dir, stem, fmt)
            if params.save_metadata and fmt == "png":
                metadata = PngImagePlugin.PngInfo()
                metadata.add_text(
                    "parameters", self._build_metadata_string(params, seed)
                )
                image.save(filepath, pnginfo=metadata)
            else:
                image.save(filepath)
            saved.append(str(filepath))
        return saved

    @staticmethod
    def _next_available_path(output_dir: Path, stem: str, suffix: str) -> Path:
        candidate = output_dir / f"{stem}.{suffix}"
        counter = 1
        while candidate.exists():
            candidate = output_dir / f"{stem}_{counter:02d}.{suffix}"
            counter += 1
        return candidate

    def _build_metadata_string(self, params: GenParams, seed: int) -> str:
        lines = [
            params.prompt,
            f"Negative prompt: {params.negative_prompt}",
            f"Steps: {params.steps}, CFG scale: {params.cfg_scale}",
            f"Seed: {seed}, Size: {params.width}x{params.height}",
            f"Model: {self.config.models.base_model_id}",
            f"Scheduler: {self.config.inference.scheduler}",
        ]
        if params.control_mode:
            lines.append(
                f"ControlNet: {params.control_mode.value}, "
                f"weight: {params.control_weight}"
            )
        if params.lora_path:
            lines.append(f"LoRA: {params.lora_path}, scale: {params.lora_scale}")
        if params.clip_skip is not None:
            lines.append(f"Clip skip: {params.clip_skip}")
        return "\n".join(lines)

    def _get_cn_processor(self) -> ControlNetProcessor:
        if self.cn_processor is None:
            self.cn_processor = ControlNetProcessor(device=self.engine.get_device())
        return self.cn_processor

    def cleanup(self):
        with self._lock:
            self.engine.unload_all()
            if self.cn_processor is not None:
                self.cn_processor.cleanup()
