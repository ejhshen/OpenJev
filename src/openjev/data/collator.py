"""Pad whole decisions after their rank/microbatch assignment."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .schema import DecisionContext, DecisionExample


class DecisionCollator:
    """Flatten complete decision branches, never split or truncate options.

    Call this after selecting full decisions for a rank/microbatch. ``decision_ptr``
    reconstructs each decision's full K-way distribution from the flattened rows.
    """

    def __init__(self, serializer: Any, pad_token_id: int, *,
                 max_branch_length: int | None = None, pad_to_multiple_of: int = 1):
        if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int) or pad_token_id < 0:
            raise ValueError("pad_token_id must be a nonnegative integer")
        if pad_to_multiple_of < 1 or (max_branch_length is not None and max_branch_length < 1):
            raise ValueError("padding multiple and length limit must be positive")
        self.serializer = serializer
        self.pad_token_id = pad_token_id
        self.max_branch_length = max_branch_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, rows: Sequence[DecisionExample | Mapping[str, Any]]) -> dict[str, Any]:
        import torch

        if not rows:
            raise ValueError("cannot collate an empty decision batch")
        examples = [row if isinstance(row, DecisionExample) else DecisionExample.from_dict(row) for row in rows]
        serialized = [self.serializer.serialize(example.decision) for example in examples]
        if any(tuple(encoded.option_ids) != tuple(option.id for option in example.decision.options)
               for encoded, example in zip(serialized, examples)):
            raise ValueError("serializer changed option order")
        result = collate_serialized(serialized, self.pad_token_id,
                                    max_branch_length=self.max_branch_length,
                                    pad_to_multiple_of=self.pad_to_multiple_of)
        batch, max_k = len(examples), result["option_mask"].shape[1]
        targets = torch.zeros((batch, max_k), dtype=torch.float32)
        for index, example in enumerate(examples):
            targets[index, :len(example.decision.options)] = torch.tensor(example.target.vector(example.decision), dtype=torch.float32)
        result.update(targets=targets,
                      sample_weight=torch.tensor([ex.sample_weight for ex in examples], dtype=torch.float32),
                      group_ids=[ex.group_id for ex in examples])
        return result

    def collate_contexts(self, rows: Sequence[DecisionContext | Mapping[str, Any]]) -> dict[str, Any]:
        """Use the shared tensor layout without reading or inventing labels."""
        import torch

        if not rows:
            raise ValueError("cannot collate an empty context batch")
        contexts = [row if isinstance(row, DecisionContext) else DecisionContext.from_dict(row) for row in rows]
        serialized = [self.serializer.serialize(context.decision) for context in contexts]
        if any(tuple(encoded.option_ids) != tuple(option.id for option in context.decision.options)
               for encoded, context in zip(serialized, contexts)):
            raise ValueError("serializer changed option order")
        result = collate_serialized(serialized, self.pad_token_id,
                                    max_branch_length=self.max_branch_length,
                                    pad_to_multiple_of=self.pad_to_multiple_of)
        result.update(sample_weight=torch.tensor([row.sample_weight for row in contexts], dtype=torch.float32),
                      group_ids=[row.group_id for row in contexts])
        return result


def collate_serialized(serialized, pad_token_id, *, max_branch_length=None, pad_to_multiple_of=1):
        """Shared train/inference tensor layout; inference never invents labels."""
        import torch

        if not serialized:
            raise ValueError("cannot collate an empty decision batch")
        branches = []
        ptr = [0]
        for encoded in serialized:
            if len(encoded.branches) != len(encoded.option_ids):
                raise ValueError("serializer must return one branch for every option")
            branches.extend(encoded.branches)
            ptr.append(len(branches))
        raw_length = max(len(branch.input_ids) for branch in branches)
        if max_branch_length is not None and raw_length > max_branch_length:
            raise ValueError(f"branch length {raw_length} exceeds {max_branch_length}; truncation is disabled")
        length = ((raw_length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        if max_branch_length is not None and length > max_branch_length:
            raise ValueError("padding would exceed max_branch_length")
        n, batch, max_k = len(branches), len(serialized), max(len(ex.option_ids) for ex in serialized)
        input_ids = torch.full((n, length), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((n, length), dtype=torch.bool)
        position_ids = torch.zeros((n, length), dtype=torch.long)
        question_end, option_end = [], []
        for index, branch in enumerate(branches):
            size = len(branch.input_ids)
            if not (0 <= branch.question_end <= branch.option_end < size):
                raise ValueError("serializer returned invalid pooling spans")
            input_ids[index, :size] = torch.tensor(branch.input_ids, dtype=torch.long)
            attention_mask[index, :size] = True
            position_ids[index, :size] = torch.arange(size, dtype=torch.long)
            question_end.append(branch.question_end)
            option_end.append(branch.option_end)
        option_mask = torch.zeros((batch, max_k), dtype=torch.bool)
        for index, encoded in enumerate(serialized):
            option_mask[index, :len(encoded.option_ids)] = True
        return {
            "input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids,
            "decision_ptr": torch.tensor(ptr, dtype=torch.long),
            "question_end": torch.tensor(question_end, dtype=torch.long),
            "option_end": torch.tensor(option_end, dtype=torch.long),
            "option_mask": option_mask,
            "decision_ids": [ex.decision_id for ex in serialized],
            "option_ids": [ex.option_ids for ex in serialized],
        }


def pack_decisions(entries, max_decisions, max_tokens, *, pad_multiple=1):
    """Entries are (role, row, serialized); never split a decision or mix targets with primary."""
    groups = []
    for primary in (False, True):
        ordered = sorted((x for x in entries if (x[0] == 'primary') == primary),
                         key=lambda x: max(len(b.input_ids) for b in x[2].branches))
        group = []; length = branches = 0
        for entry in ordered:
            raw = max(length, max(len(b.input_ids) for b in entry[2].branches))
            padded = (raw + pad_multiple - 1) // pad_multiple * pad_multiple
            count = branches + len(entry[2].branches)
            if group and (len(group) >= max_decisions or (max_tokens and count*padded > max_tokens)):
                groups.append(group); group = []; length = branches = 0
            group.append(entry)
            length = max(length, max(len(b.input_ids) for b in entry[2].branches))
            branches += len(entry[2].branches)
        if group:
            groups.append(group)
    return groups


def split_to_microsteps(groups, count):
    """Equalize FSDP call counts by splitting real batches, not duplicating examples."""
    result = [list(x) for x in groups]
    if count < len(result) or count > sum(map(len, result)):
        raise ValueError('cannot create requested number of nonempty microbatches')
    while len(result) < count:
        i = max(range(len(result)), key=lambda j: len(result[j]))
        group = result.pop(i); mid = len(group)//2
        result[i:i] = [group[:mid], group[mid:]]
    return result
