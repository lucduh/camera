import pytest
import torch
from transformers import (
    DonutSwinConfig,
    DonutSwinModel,
    MBartConfig,
    MBartForCausalLM,
    VisionEncoderDecoderModel,
)

from donut_camera.attention import apply_attention, restore_attention


def tiny_model():
    encoder = DonutSwinModel(
        DonutSwinConfig(
            image_size=32,
            patch_size=4,
            embed_dim=8,
            depths=[2, 2],
            num_heads=[2, 4],
            window_size=2,
        )
    )
    decoder = MBartForCausalLM(
        MBartConfig(
            vocab_size=32,
            d_model=16,
            decoder_layers=1,
            decoder_attention_heads=2,
            decoder_ffn_dim=32,
            is_decoder=True,
            add_cross_attention=True,
            dropout=0,
            attention_dropout=0,
        )
    )
    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder).eval()
    model.config.decoder_start_token_id = 2
    model.config.pad_token_id = 1
    return model


@pytest.mark.parametrize("preset", ["eager", "encoder_sdpa", "sdpa", "sdpa_math"])
def test_forward_gradient_and_cache_parity(preset):
    torch.manual_seed(42)
    model = tiny_model()
    pixels = torch.randn(1, 3, 32, 32)
    labels = torch.tensor([[4, 5, 6, 2]])
    apply_attention(model, "baseline")
    reference = model(pixel_values=pixels, labels=labels)
    reference.loss.backward()
    gradient = model.encoder.embeddings.patch_embeddings.projection.weight.grad.clone()
    model.zero_grad()
    apply_attention(model, preset)
    candidate = model(pixel_values=pixels, labels=labels)
    candidate.loss.backward()
    torch.testing.assert_close(reference.logits, candidate.logits, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(
        gradient,
        model.encoder.embeddings.patch_embeddings.projection.weight.grad,
        atol=1e-5,
        rtol=1e-4,
    )
    with torch.no_grad():
        states = model.encoder(pixels)
        full = model(encoder_outputs=states, decoder_input_ids=labels).logits
        cache = None
        for index in range(labels.shape[1]):
            step = model(
                encoder_outputs=states,
                decoder_input_ids=labels[:, index : index + 1],
                past_key_values=cache,
                use_cache=True,
            )
            cache = step.past_key_values
            torch.testing.assert_close(
                full[:, index : index + 1], step.logits, atol=1e-5, rtol=1e-4
            )
    restore_attention(model)
    assert model.decoder.config._attn_implementation == "eager"
    assert not hasattr(model, "_ch05_mask_cache")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("preset", ["sdpa_flash", "sdpa_efficient", "sdpa_cudnn"])
def test_cuda_forced_backend_smoke(preset):
    model = tiny_model().cuda().to(torch.bfloat16)
    apply_attention(model, preset)
    with torch.no_grad():
        output = model(
            pixel_values=torch.randn(1, 3, 32, 32, device="cuda", dtype=torch.bfloat16),
            labels=torch.tensor([[4, 5, 2]], device="cuda"),
        )
    assert torch.isfinite(output.logits).all()


def test_encoder_attention_weights_fall_back_to_eager():
    model = tiny_model()
    module = model.encoder.encoder.layers[0].blocks[0].attention.self
    hidden = torch.randn(2, 4, 8)
    expected = module(hidden, output_attentions=True)
    apply_attention(model, "encoder_sdpa")
    actual = module(hidden, output_attentions=True)
    for reference, candidate in zip(expected, actual, strict=True):
        torch.testing.assert_close(reference, candidate, atol=0, rtol=0)
