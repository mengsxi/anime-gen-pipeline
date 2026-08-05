from pathlib import Path

from anime_gen.config import AppConfig
from anime_gen.download import download_base_model, download_controlnet_models, main


def test_completed_base_model_is_not_downloaded(monkeypatch, tmp_path: Path):
    config = AppConfig()
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    (base_dir / "model_index.json").write_text("{}", encoding="utf-8")
    (base_dir / "unet").mkdir()
    (base_dir / "unet" / "model.safetensors").touch()
    config.models.base_model_path = str(base_dir)

    def fail_download(**kwargs):
        raise AssertionError(f"unexpected download: {kwargs}")

    monkeypatch.setattr("anime_gen.download.snapshot_download", fail_download)
    download_base_model(config)


def test_controlnet_download_uses_configured_dir_and_endpoint(
    monkeypatch, tmp_path: Path
):
    config = AppConfig()
    config.models.controlnet_dir = str(tmp_path / "controlnet")
    calls = []
    monkeypatch.setenv("HF_ENDPOINT", "https://example.invalid")
    monkeypatch.setattr(
        "anime_gen.download.snapshot_download",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr("anime_gen.download._log_download_size", lambda *args: None)

    download_controlnet_models(config, ["canny"])

    assert calls == [
        {
            "repo_id": config.models.controlnet["canny"],
            "local_dir": str(tmp_path / "controlnet" / "canny"),
            "allow_patterns": ["config.json", "diffusion_pytorch_model.*"],
            "endpoint": "https://example.invalid",
        }
    ]


def test_download_cli_defaults_to_base_only(monkeypatch):
    calls = []
    monkeypatch.setattr("sys.argv", ["anime-gen-download"])
    monkeypatch.setattr("anime_gen.download.load_config", lambda path=None: AppConfig())
    monkeypatch.setattr(
        "anime_gen.download.download_base_model", lambda config: calls.append("base")
    )
    monkeypatch.setattr(
        "anime_gen.download.download_controlnet_models",
        lambda config, modes=None: calls.append(("controlnet", modes)),
    )

    main()

    assert calls == ["base"]
