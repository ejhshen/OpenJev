"""Shared dynamic-option head, initialized independently of its backbone."""

import math

import torch
from torch import Tensor, nn

from .schema import DecisionOutput
from .set_interactor import OptionSetInteractor


class DecisionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        set_dim: int = 512,
        set_layers: int = 2,
        set_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.0,
        init_seed: int = 0,
        score_init_gain: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if (isinstance(score_init_gain, bool) or not isinstance(score_init_gain, (int, float))
                or not math.isfinite(score_init_gain) or score_init_gain <= 0):
            raise ValueError("score_init_gain must be finite and positive")
        self.input_dim = input_dim
        self.set_dim = set_dim
        self.score_init_gain = float(score_init_gain)
        self.config = dict(
            input_dim=input_dim, set_dim=set_dim, set_layers=set_layers,
            set_heads=set_heads, ffn_dim=ffn_dim, dropout=dropout, init_seed=init_seed,
            score_init_gain=self.score_init_gain,
        )
        # Preserve the caller's RNG and never traverse the backbone during init.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator(device="cpu").manual_seed(init_seed).get_state())
            self.question_projection = nn.Linear(input_dim, set_dim)
            self.option_projection = nn.Linear(input_dim, set_dim)
            self.interactor = OptionSetInteractor(
                set_dim, layers=set_layers, heads=set_heads, ffn_dim=ffn_dim, dropout=dropout
            )
            self.question_compatibility = nn.Linear(set_dim, set_dim, bias=False)
            self.option_compatibility = nn.Linear(set_dim, set_dim, bias=False)
            self._initialize_head()

    def _initialize_head(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Both terminal bilinear maps stay nonzero: full-model
                # gradients exist from the first update. This changes only
                # initialization, not forward temperature or trained logits.
                gain = self.score_init_gain if module in (
                    self.question_compatibility, self.option_compatibility
                ) else 1.0
                nn.init.xavier_uniform_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        question_features: Tensor,
        option_features: Tensor,
        option_mask: Tensor,
        *,
        temperature: float = 1.0,
    ) -> DecisionOutput:
        if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be a finite positive scalar")
        if question_features.ndim != 2 or option_features.ndim != 3:
            raise ValueError("question/option features must have shapes [B,H] and [B,K,H]")
        if question_features.shape != (option_features.shape[0], self.input_dim):
            raise ValueError("question features do not match batch size or input_dim")
        if option_features.shape[-1] != self.input_dim or option_mask.shape != option_features.shape[:2]:
            raise ValueError("option features or mask have incompatible shape")
        if option_mask.dtype != torch.bool or not bool(option_mask.any(dim=1).all()):
            raise ValueError("every decision must have at least one valid option")
        question = self.question_projection(
            question_features.to(dtype=self.question_projection.weight.dtype)
        )
        options = self.option_projection(
            option_features.masked_fill(~option_mask.unsqueeze(-1), 0).to(
                dtype=self.option_projection.weight.dtype
            )
        )
        options = self.interactor(options + question.unsqueeze(1), option_mask)
        q = self.question_compatibility(question).float()
        e = self.option_compatibility(options).float()
        # Elementwise FP32 reduction avoids autocast lowering an einsum/matmul
        # back to BF16 after the explicit casts.
        logits = (q.unsqueeze(1) * e).sum(dim=-1, dtype=torch.float32) / math.sqrt(self.set_dim)
        logits = logits.masked_fill(~option_mask, -torch.inf)
        log_probs = torch.log_softmax(logits / temperature, dim=-1, dtype=torch.float32)
        return DecisionOutput(
            logits=logits, log_probs=log_probs,
            probabilities=log_probs.exp(), option_mask=option_mask,
        )
