from types import SimpleNamespace

import pytest

from donut_camera.model import set_resolution


class Processor:
    image_processor = SimpleNamespace(size=None)


class Model:
    encoder = SimpleNamespace(config=SimpleNamespace(image_size=None))
    config = SimpleNamespace(encoder=SimpleNamespace(image_size=None))


def test_set_resolution_updates_model_and_processor() -> None:
    model = Model()
    processor = Processor()
    set_resolution(model, processor, 1280, 960)
    assert model.encoder.config.image_size == [1280, 960]
    assert model.config.encoder.image_size == [1280, 960]
    assert processor.image_processor.size == {"height": 1280, "width": 960}


def test_resolution_must_be_divisible_by_40() -> None:
    with pytest.raises(ValueError):
        set_resolution(Model(), Processor(), 1281, 960)
