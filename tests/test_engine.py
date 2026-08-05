import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from anime_gen.config import AppConfig, InferenceConfig, LowVRAMConfig
from anime_gen.engine import ModelEngine


class FakeTaskFactory:
    calls = []

    @classmethod
    def from_pipe(cls, pipe, **kwargs):
        result = SimpleNamespace(source=pipe, kwargs=kwargs)
        cls.calls.append(result)
        return result


class FakeControlNet:
    sources = []

    @classmethod
    def from_pretrained(cls, source, **kwargs):
        cls.sources.append((source, kwargs))
        return cls()

    def to(self, device):
        self.device = device
        return self


def fake_diffusers_module():
    module = ModuleType("diffusers")
    module.AutoPipelineForImage2Image = FakeTaskFactory
    module.AutoPipelineForInpainting = FakeTaskFactory
    module.StableDiffusionControlNetPipeline = FakeTaskFactory
    module.ControlNetModel = FakeControlNet
    return module


def make_engine(tmp_path: Path) -> ModelEngine:
    config = AppConfig(
        inference=InferenceConfig(device="cpu", torch_dtype="fp32"),
        low_vram=LowVRAMConfig(enable_xformers=False),
    )
    config.models.controlnet_dir = str(tmp_path / "controlnet")
    engine = ModelEngine(config)
    engine._pipe = object()
    return engine


def test_task_pipelines_are_created_from_shared_base(monkeypatch, tmp_path: Path):
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers_module())
    engine = make_engine(tmp_path)

    img_pipe = engine.get_pipeline("img2img")
    inpaint_pipe = engine.get_pipeline("inpaint")

    assert img_pipe.source is engine.pipe
    assert inpaint_pipe.source is engine.pipe
    assert engine.get_pipeline("img2img") is img_pipe


def test_controlnet_prefers_local_cache_and_keeps_one_mode(monkeypatch, tmp_path: Path):
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers_module())
    FakeControlNet.sources.clear()
    engine = make_engine(tmp_path)

    local_canny = tmp_path / "controlnet" / "canny"
    local_canny.mkdir(parents=True)
    (local_canny / "config.json").write_text("{}", encoding="utf-8")
    (local_canny / "model.safetensors").touch()

    canny = engine.get_pipeline("txt2img", control_mode="canny")
    pose = engine.get_pipeline("txt2img", control_mode="pose")

    assert FakeControlNet.sources[0][0] == str(local_canny)
    assert FakeControlNet.sources[1][0] == engine.config.models.controlnet["pose"]
    assert canny is not pose
    assert engine._controlnet_mode == "pose"
    assert engine._control_pipe is pose
