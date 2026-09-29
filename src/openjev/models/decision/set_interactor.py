"""Permutation-equivariant pre-norm attention over a complete option set."""

import torch
from torch import Tensor, nn


class SetBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor, option_mask: Tensor) -> Tensor:
        normalized = self.attention_norm(values)
        update, _ = self.attention(
            normalized, normalized, normalized,
            key_padding_mask=~option_mask, need_weights=False,
        )
        values = values + self.dropout(update)
        values = values + self.dropout(self.ffn(self.ffn_norm(values)))
        return values.masked_fill(~option_mask.unsqueeze(-1), 0)


class OptionSetInteractor(nn.Module):
    def __init__(
        self, dim: int, *, layers: int = 2, heads: int = 8,
        ffn_dim: int = 2048, dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim <= 0 or heads <= 0 or dim % heads or layers < 0 or ffn_dim <= 0:
            raise ValueError("invalid set dimensions, head count or layer count")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.blocks = nn.ModuleList(SetBlock(dim, heads, ffn_dim, dropout) for _ in range(layers))

    def forward(self, values: Tensor, option_mask: Tensor) -> Tensor:
        if values.ndim != 3 or option_mask.shape != values.shape[:2]:
            raise ValueError("set values and mask must have shapes [B,K,D] and [B,K]")
        if option_mask.dtype != torch.bool or not bool(option_mask.any(dim=1).all()):
            raise ValueError("each option set must have a boolean mask with a valid item")
        values = values.masked_fill(~option_mask.unsqueeze(-1), 0)
        for block in self.blocks:
            values = block(values, option_mask)
        return values
