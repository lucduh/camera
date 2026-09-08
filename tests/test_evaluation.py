import torch

from donut_camera.evaluation import generated_length


def test_generated_length_stops_at_eos() -> None:
    ids = torch.tensor([99, 4, 5, 2, 0, 0])
    assert generated_length(ids, prompt_length=1, eos=2, pad=0) == 3


def test_generated_length_stops_at_padding() -> None:
    ids = torch.tensor([99, 4, 5, 0, 0])
    assert generated_length(ids, prompt_length=1, eos=2, pad=0) == 2
