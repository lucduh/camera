"""Shared-encoder shallow drafts. Deliberately small research implementation."""

from copy import deepcopy
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers import MBartForCausalLM

from donut_camera.decoding import Session


def truncate_decoder(target, depth):
    """Copy the first N decoder layers; retain embeddings, norm and tied head."""
    layers = target.decoder.model.decoder.layers
    if not 1 <= depth < len(layers):
        raise ValueError("Draft depth must be positive and smaller than target depth")
    draft = deepcopy(target.decoder)
    draft.model.decoder.layers = torch.nn.ModuleList(
        list(draft.model.decoder.layers)[:depth]
    )
    draft.config.decoder_layers = depth
    draft.model.decoder.config.decoder_layers = depth
    draft.eval()
    return draft


def load_draft(target, path=None, depth=1):
    if path is None:
        return truncate_decoder(target, depth)
    parameter = next(target.decoder.parameters())
    draft = (
        MBartForCausalLM.from_pretrained(
            path, dtype=parameter.dtype, attn_implementation="eager"
        )
        .to(parameter.device)
        .eval()
    )
    if (
        draft.config.vocab_size != target.decoder.config.vocab_size
        or draft.config.d_model != target.decoder.config.d_model
    ):
        raise ValueError("Draft and target must share vocabulary and hidden dimensions")
    return draft


def visual_memory(target, states):
    memory = states.last_hidden_state
    if hasattr(target, "enc_to_dec_proj"):
        memory = target.enc_to_dec_proj(memory)
    return memory


def shifted_inputs(labels, start, pad):
    inputs = labels.new_full(labels.shape, pad)
    inputs[:, 0] = start
    inputs[:, 1:] = labels[:, :-1].masked_fill(labels[:, :-1] == -100, pad)
    return inputs


def distillation_loss(student, teacher, labels, temperature=1.0, ce_weight=0.5):
    """Masked teacher KL + ground-truth CE; teacher logits are detached."""
    valid = labels != -100
    if not valid.any():
        raise ValueError("Batch contains no target tokens")
    student, teacher = student[valid].float(), teacher[valid].detach().float()
    kl = (
        F.kl_div(
            F.log_softmax(student / temperature, dim=-1),
            F.softmax(teacher / temperature, dim=-1),
            reduction="batchmean",
        )
        * temperature**2
    )
    ce = F.cross_entropy(student, labels[valid])
    return ce_weight * ce + (1 - ce_weight) * kl


@dataclass
class DraftStats:
    calls: int = 0
    proposed: int = 0
    accepted: int = 0
    rounds: list[dict] = field(default_factory=list)


def crop(cache, length):
    # EncoderDecoderCache.crop only crops self-attention, preserving visual KV.
    if cache is not None:
        cache.crop(min(length, cache.get_seq_length()))


@torch.inference_mode()
def speculative_decode(
    target,
    draft,
    states,
    prompt,
    *,
    max_new_tokens=80,
    max_draft=4,
    confidence=0.0,
):
    if max_draft < 0 or not 0 <= confidence <= 1:
        raise ValueError("Invalid proposal budget or confidence threshold")
    session = Session(target, states, prompt, max_new_tokens)
    memory = visual_memory(target, states)
    draft_cache = None
    stats = DraftStats()
    while session.remaining:
        prefix = session.result.sequences
        candidates = prefix
        proposal = []
        budget = min(max_draft, session.remaining - 1)
        for _ in range(budget):
            cached = draft_cache.get_seq_length() if draft_cache is not None else 0
            output = draft(
                input_ids=candidates[:, cached:],
                encoder_hidden_states=memory,
                past_key_values=draft_cache,
                use_cache=True,
                return_dict=True,
            )
            stats.calls += 1
            draft_cache = output.past_key_values
            scores = session.processors(candidates, output.logits[:, -1].float())
            token = scores.argmax(dim=-1, keepdim=True)
            if confidence and scores.softmax(-1).max().item() < confidence:
                break
            proposal.append(token.item())
            candidates = torch.cat((candidates, token), dim=1)
            if token.item() in session.eos:
                break
        if not proposal:
            done = session.greedy_step()
            stats.rounds.append({"proposed": 0, "accepted": 0})
        else:
            logits = session.forward(
                torch.cat((prefix[:, -1:], prefix.new_tensor([proposal])), dim=1)
            )
            session.result.verification_calls += 1
            stats.proposed += len(proposal)
            matched, done = 0, False
            for index, proposed in enumerate(proposal):
                token = session.choose(session.result.sequences, logits[:, index])
                if token.item() != proposed:
                    break
                matched += 1
                done = session.append(token)
                if done:
                    break
            else:
                token = session.choose(
                    session.result.sequences, logits[:, len(proposal)]
                )
            if not done:
                done = session.append(token)
            stats.accepted += matched
            stats.rounds.append({"proposed": len(proposal), "accepted": matched})
            crop(session.cache, session.result.sequences.shape[1] - 1)
        crop(draft_cache, session.result.sequences.shape[1] - 1)
        if done:
            break
    session.result.proposed = stats.proposed
    session.result.accepted = stats.accepted
    return session.result, stats
