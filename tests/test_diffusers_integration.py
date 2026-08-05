"""真实 Diffusers 冒烟测试；默认跳过，CI 中显式启用。"""

import os
from pathlib import Path

import pytest
from PIL import Image

from anime_gen.config import AppConfig, InferenceConfig, LowVRAMConfig, OutputConfig
from anime_gen.pipeline import AnimePipeline, GenParams

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1",
        reason="set RUN_INTEGRATION=1 to download the tiny test model",
    ),
]


def test_tiny_diffusers_txt2img_img2img_and_inpaint(tmp_path: Path):
    config = AppConfig(
        inference=InferenceConfig(
            device="cpu",
            torch_dtype="fp32",
            default_steps=1,
            default_cfg=1.0,
            default_width=64,
            default_height=64,
            default_batch_size=1,
            default_seed=0,
            clip_skip=None,
            scheduler="model",
            enable_safety_checker=False,
        ),
        low_vram=LowVRAMConfig(
            enable_attention_slicing=False,
            enable_vae_slicing=False,
            enable_vae_tiling=False,
            enable_xformers=False,
        ),
        output=OutputConfig(dir=str(tmp_path / "outputs")),
    )
    config.models.base_model_id = "hf-internal-testing/tiny-stable-diffusion-pipe"
    config.models.base_model_path = str(tmp_path / "missing-local-model")
    config.models.use_safetensors = False
    config.models.variant = None

    pipeline = AnimePipeline(config)
    source = Image.new("RGB", (64, 64), "gray")
    mask = Image.new("L", (64, 64), "white")
    try:
        txt_result = pipeline.txt2img(GenParams(prompt="anime character"))
        img_result = pipeline.img2img(
            GenParams(prompt="anime character", init_image=source, strength=0.5)
        )
        paint_result = pipeline.inpaint(
            GenParams(
                prompt="anime character",
                init_image=source,
                mask_image=mask,
                strength=0.5,
            )
        )
    finally:
        pipeline.cleanup()

    assert txt_result.images[0].size == (64, 64)
    assert img_result.images[0].size == (64, 64)
    assert paint_result.images[0].size == (64, 64)
