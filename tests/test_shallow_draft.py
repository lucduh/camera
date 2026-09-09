from copy import deepcopy

import pytest
import torch
from test_attention import tiny_model

from donut_camera.decoding import decode
from donut_camera.shallow_draft import (
    distillation_loss,
    shifted_inputs,
    speculative_decode,
    truncate_decoder,
    visual_memory,
)


def two_layer_model():
    model = tiny_model()
    layer = model.decoder.model.decoder.layers[0]
    model.decoder.model.decoder.layers = torch.nn.ModuleList([layer, deepcopy(layer)])
    model.decoder.config.decoder_layers = 2
    model.decoder.model.decoder.config.decoder_layers = 2
    return model


def test_truncated_draft_has_requested_depth_and_shared_shape():
    model = two_layer_model()
    draft = truncate_decoder(model, 1)
    assert len(draft.model.decoder.layers) == 1
    assert draft.config.vocab_size == model.decoder.config.vocab_size
    assert draft.config.d_model == model.decoder.config.d_model


def test_shifted_inputs_and_distillation_loss():
    labels = torch.tensor([[4, 5, -100]])
    shifted = shifted_inputs(labels, start=2, pad=1)
    assert shifted.tolist() == [[2, 4, 5]]
    loss = distillation_loss(torch.randn(1, 3, 8), torch.randn(1, 3, 8), labels)
    assert torch.isfinite(loss)


@pytest.mark.parametrize("draft_depth", [1])
def test_shared_encoder_speculation_runs_and_is_exact(draft_depth):
    torch.manual_seed(4)
    model = two_layer_model().eval()
    model.config.decoder_start_token_id = 2
    model.generation_config.decoder_start_token_id = 2
    draft = truncate_decoder(model, draft_depth)
    pixels = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        states = model.encoder(pixels, return_dict=True)
        reference = decode(
            model, states, torch.tensor([[2]]), strategy="greedy", max_new_tokens=6
        )
        result, stats = speculative_decode(
            model, draft, states, torch.tensor([[2]]), max_new_tokens=6, max_draft=2
        )
    assert torch.equal(reference.sequences, result.sequences)
    assert stats.calls >= 1
    assert stats.proposed >= stats.accepted
    assert visual_memory(model, states).shape[-1] == model.decoder.config.d_model
