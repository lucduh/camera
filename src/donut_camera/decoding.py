"""Batch-one decoding with a shared, explicit greedy generation policy."""

import time
from dataclasses import dataclass

import torch
from transformers import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList

STRATEGIES = ("hf", "greedy", "template", "speculative")


@dataclass(frozen=True)
class Grammar:
    opens: tuple[int, ...]
    closes: tuple[int, ...]
    missing: int
    eos: int
    start: int

    @classmethod
    def from_bundle(cls, bundle):
        tokenizer, task = bundle.processor.tokenizer, bundle.task
        names = [task.task_token, task.missing_token]
        names += [
            token
            for field in task.fields
            for token in (task.open_token(field), task.close_token(field))
        ]
        ids = [tokenizer.convert_tokens_to_ids(name) for name in names]
        if any(token == tokenizer.unk_token_id for token in ids):
            raise ValueError("Checkpoint is missing schema tokens")
        if any(
            tokenizer.encode(name, add_special_tokens=False) != [token]
            for name, token in zip(names, ids, strict=True)
        ):
            raise ValueError("Schema tokens must each encode as one token")
        return cls(
            tuple(ids[2::2]), tuple(ids[3::2]), ids[1], tokenizer.eos_token_id, ids[0]
        )


@dataclass
class DecodeResult:
    sequences: torch.Tensor
    target_calls: int = 0
    forced: int = 0
    proposed: int = 0
    accepted: int = 0
    verification_calls: int = 0
    rebuild_calls: int = 0
    termination: str = "length_limit"
    rebuild_ms: float | None = None

    def counters(self):
        emitted = self.sequences.shape[1] - 1
        return {
            "emitted_tokens": emitted,
            "target_calls": self.target_calls,
            "forced_tokens": self.forced,
            "proposed_tokens": self.proposed,
            "accepted_tokens": self.accepted,
            "verification_calls": self.verification_calls,
            "rebuild_calls": self.rebuild_calls,
            "termination": self.termination,
            "acceptance_rate": self.accepted / self.proposed if self.proposed else None,
            "tokens_per_target_call": emitted / self.target_calls
            if self.target_calls
            else None,
        }


def generation_policy(model, max_new_tokens):
    """Deliberately restricted policy, shared with HF; no inherited sampling rules.

    Preserve checkpoint start/pad/EOS and forced BOS/EOS. All other generation
    processors are disabled explicitly rather than approximated in a custom loop.
    """
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    original = model.generation_config
    return GenerationConfig(
        do_sample=False,
        num_beams=1,
        use_cache=True,
        decoder_start_token_id=original.decoder_start_token_id,
        pad_token_id=original.pad_token_id,
        eos_token_id=original.eos_token_id,
        forced_bos_token_id=original.forced_bos_token_id,
        forced_eos_token_id=original.forced_eos_token_id,
        max_new_tokens=max_new_tokens,
        max_length=max_new_tokens + 1,
    )


class Session:
    def __init__(self, model, states, prompt, max_new_tokens, profile_rebuilds=False):
        if prompt.shape != (1, 1):
            raise NotImplementedError(
                "Research decoding requires batch one and a one-token prompt"
            )
        self.model, self.states = model, states
        self.policy = generation_policy(model, max_new_tokens)
        # Pinned Transformers integration, also exercised by the tiny-model tests.
        model._prepare_special_tokens(self.policy, device=prompt.device)
        self.processors = model._get_logits_processor(
            self.policy,
            input_ids_seq_length=1,
            device=prompt.device,
            logits_processor=LogitsProcessorList(),
        )
        self.result = DecodeResult(
            prompt.clone(), rebuild_ms=0.0 if profile_rebuilds else None
        )
        self.cache = None
        eos = self.policy.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])
        self.limit = max_new_tokens

    @property
    def remaining(self):
        return self.limit - (self.result.sequences.shape[1] - 1)

    def forward(self, inputs, *, rebuild=False):
        self.result.target_calls += 1
        if rebuild:
            self.result.rebuild_calls += 1
        instrument = rebuild and self.result.rebuild_ms is not None
        if instrument:
            if inputs.device.type == "cuda":
                torch.cuda.synchronize(inputs.device)
            started = time.perf_counter()
        output = self.model(
            encoder_outputs=self.states,
            decoder_input_ids=inputs,
            past_key_values=None if rebuild else self.cache,
            use_cache=True,
            return_dict=True,
        )
        if instrument:
            if inputs.device.type == "cuda":
                torch.cuda.synchronize(inputs.device)
            self.result.rebuild_ms += (time.perf_counter() - started) * 1000
        self.cache = output.past_key_values
        return output.logits

    def choose(self, prefix, scores):
        return self.processors(prefix, scores.float()).argmax(dim=-1, keepdim=True)

    def append(self, token, *, forced=False):
        rule_forced = (
            self.result.sequences.shape[1] == 1
            and self.policy.forced_bos_token_id is not None
        ) or (self.remaining == 1 and self.policy.forced_eos_token_id is not None)
        self.result.sequences = torch.cat((self.result.sequences, token), dim=1)
        self.result.forced += int(forced or rule_forced)
        if token.item() in self.eos:
            self.result.termination = "eos"
            return True
        return not self.remaining

    def greedy_step(self):
        logits = self.forward(self.result.sequences[:, -1:])
        return self.append(self.choose(self.result.sequences, logits[:, -1]))


@torch.inference_mode()
def decode(
    model,
    states,
    prompt,
    *,
    strategy="greedy",
    max_new_tokens=80,
    grammar=None,
    max_draft=8,
    missing_chain=True,
    profile_rebuilds=False,
):
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    if prompt.shape != (1, 1):
        raise NotImplementedError(
            "Research decoding requires batch one and a one-token prompt"
        )
    if strategy == "hf":
        policy = generation_policy(model, max_new_tokens)
        eos = policy.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        calls = 0

        def count(_module, _inputs):
            nonlocal calls
            calls += 1

        handle = model.decoder.register_forward_pre_hook(count)
        try:
            sequences = model.generate(
                encoder_outputs=states,
                decoder_input_ids=prompt,
                generation_config=policy,
            )
        finally:
            handle.remove()
        forced_positions = set()
        if policy.forced_bos_token_id is not None and sequences.shape[1] > 1:
            forced_positions.add(1)
        if (
            policy.forced_eos_token_id is not None
            and sequences.shape[1] == max_new_tokens + 1
        ):
            forced_positions.add(max_new_tokens)
        return DecodeResult(
            sequences,
            target_calls=calls,
            forced=len(forced_positions),
            termination="eos" if sequences[0, -1].item() in eos_ids else "length_limit",
        )
    session = Session(model, states, prompt, max_new_tokens, profile_rebuilds)
    if strategy == "greedy":
        while session.remaining:
            if session.greedy_step():
                break
    elif strategy == "template":
        if grammar is None:
            raise ValueError("Template decoding requires a grammar")
        template(session, grammar)
    else:
        from donut_camera.speculation import speculate

        if grammar is None:
            raise ValueError("Speculative decoding requires a grammar")
        speculate(session, grammar, max_draft=max_draft, missing_chain=missing_chain)
    return session.result


def template(session, grammar):
    for opening, closing in zip(grammar.opens, grammar.closes, strict=True):
        if not session.remaining:
            return
        prefix = session.result.sequences
        forced = prefix.new_tensor([[opening]])
        if session.append(forced, forced=True):
            return
        # Both positions must enter the cache, but only the second predicts a value.
        logits = session.forward(torch.cat((prefix[:, -1:], forced), dim=1))
        token = session.choose(session.result.sequences, logits[:, -1])
        while True:
            done = session.append(token)
            if token.item() in session.eos:
                session.result.termination = "unexpected_eos"
                return
            if token.item() == closing:
                if done:
                    return
                break
            if done:
                session.result.termination = "missing_close_at_limit"
                return
            logits = session.forward(session.result.sequences[:, -1:])
            token = session.choose(session.result.sequences, logits[:, -1])
    if session.remaining:
        session.append(
            session.result.sequences.new_tensor([[grammar.eos]]), forced=True
        )
