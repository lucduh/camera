import pytest
import torch
from test_attention import tiny_model

from donut_camera.decoding import Grammar, decode
from donut_camera.speculation import propose

GRAMMAR = Grammar((4, 6), (5, 7), 8, 2, 2)


@pytest.mark.parametrize("budget", [1, 2, 4, 8])
@pytest.mark.parametrize("seed", [1, 42])
def test_custom_greedy_and_speculative_match_hf(budget, seed):
    torch.manual_seed(seed)
    model = tiny_model().eval()
    model.generation_config.decoder_start_token_id = 2
    model.generation_config.eos_token_id = 2
    model.generation_config.pad_token_id = 1
    with torch.no_grad():
        states = model.encoder(torch.randn(1, 3, 32, 32))
    prompt = torch.tensor([[2]])
    hf = decode(model, states, prompt, strategy="hf", max_new_tokens=budget)
    for strategy in ("greedy", "speculative"):
        result = decode(
            model,
            states,
            prompt,
            strategy=strategy,
            max_new_tokens=budget,
            grammar=GRAMMAR,
        )
        assert torch.equal(hf.sequences, result.sequences)
        assert result.target_calls >= 1
        assert result.sequences.shape[1] <= budget + 1


def test_proposer_does_not_guess_free_text_or_continue_past_eos():
    assert propose([2], GRAMMAR, 8) == [4, 8, 5, 6, 8, 7, 2]
    assert propose([2, 4, 9], GRAMMAR, 8) == []
    assert propose([2, 4], GRAMMAR, 8, missing_chain=False) == []
    assert propose([2, 4, 9, 5], GRAMMAR, 8, missing_chain=False) == [6]


def scripted_model(sequence):
    from types import MethodType, SimpleNamespace

    model = tiny_model()
    model.generation_config.decoder_start_token_id = 2
    model.generation_config.eos_token_id = 2
    model.generation_config.forced_eos_token_id = None

    def forward(
        self, encoder_outputs, decoder_input_ids, past_key_values=None, **kwargs
    ):
        prefix = [] if past_key_values is None else list(past_key_values)
        logits = torch.full((1, decoder_input_ids.shape[1], 32), -100.0)
        for index, token in enumerate(decoder_input_ids[0].tolist()):
            prefix.append(token)
            position = len(prefix) - 1
            # Wrong cached prefixes produce a different continuation; a cache
            # retaining rejected draft tokens cannot accidentally pass this test.
            expected_prefix = [2] + sequence[:position]
            next_token = (
                sequence[position]
                if prefix == expected_prefix and position < len(sequence)
                else 10
            )
            logits[0, index, next_token] = 100
        return SimpleNamespace(logits=logits, past_key_values=prefix)

    model.forward = MethodType(forward, model)
    return model


@pytest.mark.parametrize(
    "sequence", [[4, 8, 5, 6, 8, 7, 2], [4, 9, 5, 6, 11, 7, 2], [4, 9, 2]]
)
def test_verification_acceptance_rejection_and_eos(sequence):
    model = scripted_model(sequence)
    result = decode(
        model,
        None,
        torch.tensor([[2]]),
        strategy="speculative",
        grammar=GRAMMAR,
        max_new_tokens=16,
    )
    assert result.sequences.tolist() == [[2] + sequence]
    assert result.accepted <= result.proposed
    assert result.target_calls >= result.verification_calls + result.rebuild_calls
    if 9 in sequence:
        assert result.rebuild_calls > 0
    else:
        assert result.rebuild_calls == 0
        assert result.target_calls < len(sequence)


@pytest.mark.parametrize("budget", [1, 2, 3, 6, 7, 8])
def test_template_budget_and_forced_tokens(budget):
    sequence = [4, 9, 5, 6, 11, 7, 2]
    result = decode(
        scripted_model(sequence),
        None,
        torch.tensor([[2]]),
        strategy="template",
        grammar=GRAMMAR,
        max_new_tokens=budget,
    )
    assert result.sequences.tolist() == [[2] + sequence[:budget]]
    assert result.forced >= 1
    assert result.target_calls < len(sequence)


def test_template_unexpected_eos_and_missing_close():
    result = decode(
        scripted_model([4, 2]),
        None,
        torch.tensor([[2]]),
        strategy="template",
        grammar=GRAMMAR,
    )
    assert result.termination == "unexpected_eos"
    result = decode(
        scripted_model([4, 9, 9, 9]),
        None,
        torch.tensor([[2]]),
        strategy="template",
        grammar=GRAMMAR,
        max_new_tokens=4,
    )
    assert result.termination == "missing_close_at_limit"


def test_batch_and_budget_validation():
    model = tiny_model()
    with pytest.raises(NotImplementedError):
        decode(model, None, torch.tensor([[2], [2]]))
    with pytest.raises(ValueError):
        decode(model, None, torch.tensor([[2]]), max_new_tokens=0)


def test_zero_proposal_budget_is_greedy_and_rebuild_profile_is_separate():
    model = scripted_model([4, 9, 5, 6, 8, 7, 2])
    result = decode(
        model,
        None,
        torch.tensor([[2]]),
        strategy="speculative",
        grammar=GRAMMAR,
        max_draft=0,
    )
    assert result.proposed == result.accepted == result.rebuild_calls == 0
    assert result.target_calls == result.sequences.shape[1] - 1
    profiled = decode(
        model,
        None,
        torch.tensor([[2]]),
        strategy="speculative",
        grammar=GRAMMAR,
        profile_rebuilds=True,
    )
    assert torch.equal(result.sequences, profiled.sequences)
    assert profiled.rebuild_calls > 0
    assert profiled.rebuild_ms >= 0


@pytest.mark.parametrize("budget", [1, 3, 8])
def test_forced_bos_eos_shared_with_hf_without_mutating_config(budget):
    model = tiny_model()
    model.generation_config.decoder_start_token_id = 2
    model.generation_config.forced_bos_token_id = 4
    model.generation_config.forced_eos_token_id = 2
    original = model.generation_config.to_dict()
    with torch.no_grad():
        states = model.encoder(torch.randn(1, 3, 32, 32))
    outputs = [
        decode(
            model,
            states,
            torch.tensor([[2]]),
            strategy=strategy,
            max_new_tokens=budget,
            grammar=GRAMMAR,
        )
        for strategy in ("hf", "greedy", "speculative")
    ]
    assert all(
        torch.equal(outputs[0].sequences, result.sequences) for result in outputs
    )
    assert all(outputs[0].forced == result.forced for result in outputs)
    assert model.generation_config.to_dict() == original
