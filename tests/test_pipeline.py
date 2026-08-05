from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from anime_gen.config import AppConfig, InferenceConfig, OutputConfig
from anime_gen.controlnet import ControlMode
from anime_gen.pipeline import AnimePipeline, GenParams


class FakePipe:
    def __init__(self):
        self.calls = []
        self.loaded_lora = None
        self.adapter = None
        self.unload_count = 0

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(images=[Image.new("RGB", (8, 8), "white")])

    def load_lora_weights(self, path, weight_name, adapter_name):
        self.loaded_lora = (path, weight_name, adapter_name)

    def set_adapters(self, name, adapter_weights):
        self.adapter = (name, adapter_weights)

    def unload_lora_weights(self):
        self.unload_count += 1


class FakeEngine:
    def __init__(self):
        self.pipeline = FakePipe()
        self.requests = []
        self.unloaded = False

    def get_pipeline(self, mode, control_mode=None):
        self.requests.append((mode, control_mode))
        return self.pipeline

    @staticmethod
    def get_generator_device():
        return "cpu"

    @staticmethod
    def get_device():
        return "cpu"

    @staticmethod
    def get_vram_info():
        return "VRAM: N/A (cpu mode)"

    def unload_all(self):
        self.unloaded = True


class FakeProcessor:
    def __init__(self, result):
        self.result = result
        self.calls = []
        self._preprocessors = {}

    def process(self, image, mode):
        self.calls.append((image, mode))
        return self.result


def make_pipeline(tmp_path: Path) -> tuple[AnimePipeline, FakeEngine]:
    config = AppConfig(
        inference=InferenceConfig(
            device="cpu",
            torch_dtype="fp32",
            default_steps=12,
            default_cfg=6.5,
            default_width=512,
            default_height=512,
            default_batch_size=1,
            default_seed=7,
        ),
        output=OutputConfig(dir=str(tmp_path), format="png", save_metadata=True),
    )
    pipeline = AnimePipeline(config)
    engine = FakeEngine()
    pipeline.engine = engine
    return pipeline, engine


def test_img2img_and_inpaint_use_task_specific_contracts(tmp_path: Path):
    pipeline, engine = make_pipeline(tmp_path)
    image = Image.new("RGB", (32, 32), "black")
    mask = Image.new("L", (32, 32), "white")

    pipeline.img2img(GenParams(prompt="img", init_image=image, strength=0.4))
    assert engine.requests[-1] == ("img2img", None)
    assert engine.pipeline.calls[-1]["image"] is image
    assert engine.pipeline.calls[-1]["strength"] == 0.4

    pipeline.inpaint(
        GenParams(prompt="paint", init_image=image, mask_image=mask, strength=0.6)
    )
    assert engine.requests[-1] == ("inpaint", None)
    assert engine.pipeline.calls[-1]["mask_image"] is mask


def test_controlnet_passes_condition_and_weight(tmp_path: Path):
    pipeline, engine = make_pipeline(tmp_path)
    source = Image.new("RGB", (32, 32), "black")
    condition = Image.new("RGB", (32, 32), "white")
    pipeline.cn_processor = FakeProcessor(condition)

    pipeline.generate_with_control(
        GenParams(prompt="control", control_weight=1.25),
        source,
        ControlMode.CANNY,
    )

    assert engine.requests[-1] == ("txt2img", "canny")
    call = engine.pipeline.calls[-1]
    assert call["image"] is condition
    assert call["controlnet_conditioning_scale"] == 1.25


def test_lora_scale_is_applied_and_unloaded(tmp_path: Path):
    pipeline, engine = make_pipeline(tmp_path)
    lora = tmp_path / "style.safetensors"
    lora.touch()

    pipeline.txt2img(
        GenParams(prompt="lora", lora_path=str(lora), lora_scale=0.35)
    )

    assert engine.pipeline.loaded_lora == (str(lora.parent), lora.name, "request")
    assert engine.pipeline.adapter == ("request", 0.35)
    assert "cross_attention_kwargs" not in engine.pipeline.calls[-1]
    assert engine.pipeline.unload_count == 1


def test_config_defaults_and_existing_outputs_are_preserved(tmp_path: Path):
    pipeline, engine = make_pipeline(tmp_path)
    params = GenParams(prompt="defaults")

    first = pipeline.txt2img(params)
    second = pipeline.txt2img(params)

    call = engine.pipeline.calls[0]
    assert call["num_inference_steps"] == 12
    assert call["guidance_scale"] == 6.5
    assert call["width"] == 512
    assert first.metadata["saved_paths"] != second.metadata["saved_paths"]
    assert all(Path(path).exists() for path in first.metadata["saved_paths"])
    assert all(Path(path).exists() for path in second.metadata["saved_paths"])


@pytest.mark.parametrize(
    ("params", "message"),
    [
        (GenParams(prompt=""), "prompt 不能为空"),
        (GenParams(prompt="x", width=513), "必须是 8 的倍数"),
        (GenParams(prompt="x", strength=1.1), "strength"),
        (GenParams(prompt="x", filename_prefix="../bad"), "路径分隔符"),
    ],
)
def test_invalid_parameters_fail_before_inference(tmp_path: Path, params, message):
    pipeline, engine = make_pipeline(tmp_path)
    with pytest.raises(ValueError, match=message):
        pipeline.txt2img(params)
    assert not engine.pipeline.calls
