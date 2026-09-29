"""Differentiable span pooling and grouping of complete decisions."""

import torch
from torch import Tensor

from .schema import DecisionFeatures


def pool_branch_endpoints(
    hidden: Tensor, question_end: Tensor, option_end: Tensor, attention_mask: Tensor
) -> tuple[Tensor, Tensor]:
    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
        raise ValueError("hidden/mask must have shapes [N,L,H] and [N,L]")
    n, length, _ = hidden.shape
    if question_end.shape != (n,) or option_end.shape != (n,):
        raise ValueError("endpoint tensors must contain one index per branch")
    if question_end.dtype != torch.long or option_end.dtype != torch.long:
        raise TypeError("endpoint tensors must use torch.long")
    q_end = question_end.to(hidden.device)
    o_end = option_end.to(hidden.device)
    if bool(((q_end < 0) | (o_end <= q_end) | (o_end >= length)).any()):
        raise ValueError("endpoints must satisfy 0 <= question_end < option_end < L")
    rows = torch.arange(n, device=hidden.device)
    mask = attention_mask.to(hidden.device).bool()
    if not bool((mask[rows, q_end] & mask[rows, o_end]).all()):
        raise ValueError("pooling endpoints must refer to non-padding tokens")
    return hidden[rows, q_end], hidden[rows, o_end]


def group_branch_features(
    question_features: Tensor,
    option_features: Tensor,
    decision_ptr: Tensor,
    option_mask: Tensor,
) -> DecisionFeatures:
    if question_features.ndim != 2 or option_features.shape != question_features.shape:
        raise ValueError("branch question and option features must have shape [N,H]")
    if decision_ptr.ndim != 1 or decision_ptr.dtype != torch.long:
        raise TypeError("decision_ptr must be a one-dimensional torch.long tensor")
    if option_mask.ndim != 2 or option_mask.dtype != torch.bool:
        raise TypeError("option_mask must be a boolean [B,K] tensor")
    ptr = decision_ptr.detach().cpu().tolist()
    if len(ptr) != option_mask.shape[0] + 1 or len(ptr) < 2:
        raise ValueError("decision_ptr must delimit each non-empty decision")
    if ptr[0] != 0 or ptr[-1] != len(question_features):
        raise ValueError("decision_ptr must cover all branches exactly")
    mask = option_mask.to(option_features.device)
    grouped_questions, grouped_options = [], []
    for row, (start, end) in enumerate(zip(ptr, ptr[1:])):
        indices = mask[row].nonzero(as_tuple=False).flatten()
        if end <= start or end - start != len(indices):
            raise ValueError("branch counts must equal the valid options per decision")
        grouped_questions.append(question_features[start:end].mean(dim=0))
        padded = option_features.new_zeros((mask.shape[1], option_features.shape[1]))
        grouped_options.append(padded.index_copy(0, indices, option_features[start:end]))
    return DecisionFeatures(
        question_features=torch.stack(grouped_questions),
        option_features=torch.stack(grouped_options),
        option_mask=mask,
    )
