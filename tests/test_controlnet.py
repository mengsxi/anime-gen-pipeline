from anime_gen.controlnet import ControlNetProcessor


def test_preprocessor_cache_keeps_only_latest_model():
    processor = ControlNetProcessor(device="cpu")
    processor._cache_preprocessor("pose", object())
    processor._cache_preprocessor("depth", object())

    assert list(processor._preprocessors) == ["depth"]

    processor.cleanup()
    assert not processor._preprocessors
