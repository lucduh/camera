"""Schema proposals with greedy verification and rebuild-on-rejection caching."""

import torch


def propose(prefix, grammar, max_draft, missing_chain=True):
    """Re-derive grammar state from the authoritative prefix on every call."""
    after_close = dict(
        zip(grammar.closes, (*grammar.opens[1:], grammar.eos), strict=True)
    )
    field = None
    for token in prefix:
        if token in grammar.opens:
            field = grammar.opens.index(token)
        elif token in grammar.closes:
            field = None
    chain = []
    last = prefix[-1]
    while len(chain) < max_draft:
        if last == grammar.start:
            token = grammar.opens[0]
        elif last in after_close:
            token = after_close[last]
        elif last in grammar.opens and missing_chain:
            field = grammar.opens.index(last)
            token = grammar.missing
        elif last == grammar.missing and field is not None:
            token = grammar.closes[field]
        else:
            break
        chain.append(token)
        last = token
        if token == grammar.eos:
            break
    return chain


def speculate(session, grammar, *, max_draft, missing_chain):
    if max_draft < 0:
        raise ValueError("max_draft must be nonnegative")
    while session.remaining:
        draft = propose(
            session.result.sequences[0].tolist(),
            grammar,
            min(max_draft, session.remaining - 1),
            missing_chain,
        )
        if not draft:
            if session.greedy_step():
                return
            continue
        prefix = session.result.sequences
        tokens = prefix.new_tensor([draft])
        logits = session.forward(torch.cat((prefix[:, -1:], tokens), dim=1))
        session.result.verification_calls += 1
        session.result.proposed += len(draft)
        matched = 0
        for index, proposed in enumerate(draft):
            token = session.choose(session.result.sequences, logits[:, index])
            if token.item() != proposed:
                break
            matched += 1
            session.result.accepted += 1
            if session.append(token):
                return
        else:
            token = session.choose(session.result.sequences, logits[:, len(draft)])
        if session.append(token):
            return
        if matched != len(draft):
            # Verification mutated the cache through rejected tokens. Recompute
            # exactly the accepted prefix excluding its still-uncached last token.
            session.forward(session.result.sequences[:, :-1], rebuild=True)
