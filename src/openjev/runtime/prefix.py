"""Request-local state -> question -> option execution through the adapter API."""

from collections import defaultdict

import torch

from openjev.compute import compute_context, model_compute_contract


@torch.inference_mode()
def predict_shared_prefix(model, decisions, *, device, branch_microbatch_size, pad_token_id,
                          temperature=1.0):
    """Prefill one shared state and score independent questions in input order.

    Question batches have equal lengths because their cache will be forked.
    Leaf options may have causal tail padding: pool before the padding and discard
    their caches. Only one bounded question batch and option batch are live.
    Caches belong to this call; no cache survives a request or a weight update.
    Return (outputs in input order, token positions actually processed).
    """
    if model.training:
        raise ValueError("shared-prefix execution requires eval mode")
    if branch_microbatch_size < 1:
        raise ValueError("branch_microbatch_size must be positive")
    if not decisions or any(item.state_ids != decisions[0].state_ids for item in decisions):
        raise ValueError("shared-prefix decisions must have one identical nonempty state prefix")
    adapter = model.adapter
    contract = model_compute_contract(model)

    def tokens(rows):
        return torch.tensor(rows, dtype=torch.long, device=device)

    with compute_context(contract, "backbone", device):
        state_prefix = adapter.prefill(tokens([decisions[0].state_ids]))
    executed_tokens = len(decisions[0].state_ids)
    outputs = [None] * len(decisions)
    question_groups = defaultdict(list)
    for index, decision in enumerate(decisions):
        question_groups[len(decision.question_ids)].append(index)
    for group in question_groups.values():
        for offset in range(0, len(group), branch_microbatch_size):
            question_indices = group[offset:offset + branch_microbatch_size]
            question_ids = tokens([decisions[i].question_ids for i in question_indices])
            # Continuation mutates its cache: always fork before touching a parent.
            with compute_context(contract, "backbone", device):
                question_hidden, question_prefix = adapter.continue_from(
                    adapter.fork_state(state_prefix, [0] * len(question_indices)), question_ids,
                )
            executed_tokens += question_ids.numel()
            question_features = question_hidden[:, -1].clone()
            del question_hidden
            option_features = [question_features.new_empty(
                (len(decisions[i].option_token_ids), question_features.shape[-1])
            ) for i in question_indices]
            leaves = sorted((
                (row, option, ids)
                for row, index in enumerate(question_indices)
                for option, ids in enumerate(decisions[index].option_token_ids)
            ), key=lambda leaf: len(leaf[2]))
            for start in range(0, len(leaves), branch_microbatch_size):
                batch = leaves[start:start + branch_microbatch_size]
                width = max(len(ids) for _, _, ids in batch)
                suffixes = tokens([list(ids) + [pad_token_id] * (width - len(ids)) for _, _, ids in batch])
                with compute_context(contract, "backbone", device):
                    hidden, child = adapter.continue_from(
                        adapter.fork_state(question_prefix, [row for row, _, _ in batch]), suffixes,
                    )
                executed_tokens += suffixes.numel()
                for branch, (row, option, ids) in enumerate(batch):
                    option_features[row][option] = hidden[branch, len(ids) - 1]
                del hidden, child
            del question_prefix
            for row, index in enumerate(question_indices):
                features = option_features[row]
                mask = torch.ones((1, len(features)), dtype=torch.bool, device=device)
                outputs[index] = model.score_features(
                    question_features[row:row+1], features.unsqueeze(0), mask, temperature=temperature,
                )
    return outputs, executed_tokens
