"""Context-Conditional Capacity Allocation (CCCA) transformer.

The Mnemosyne thesis applied to representation *width*: instead of every token
using the full embedding dimension, a small causal gate decides — per position,
conditioned on context — how many dimensions the model spends on predicting the
next token. A budget penalty makes capacity scarce, so the model learns to spend
width only where the context is hard.

Distinct from the explored neighbours:
  - Adaptive Input / Adaptive Softmax: width by *static token frequency*.
  - Matryoshka (MRL): nested dims, but *uniform* truncation chosen by the user.
  - Mixture-of-Depths: adapts *compute/depth*, not representation *width*.

Mechanism:
  - Run the standard transformer stack to the final hidden state h_t (causal,
    so h_t already summarises context up to t).
  - alpha_t = sigmoid(gate(h_t)) in (0, 1); active width = alpha_t * D.
  - Soft nested mask m_t[j] = sigmoid((alpha_t*D - j) / tau) zeros dims beyond the
    active width; the next-token logits see only the active prefix.
  - Loss = CE(masked) + lambda * mean(alpha). BPC is read from the pure CE only.

`uniform_width` (constant alpha, gate disabled) reproduces the Matryoshka-uniform
baseline; `force_full` (mask == 1) reproduces the dense transformer exactly.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import Transformer, ModelConfig


@dataclass
class AdaptiveConfig:
    # ── mirror of ModelConfig (so Block/CausalSelfAttention duck-type cleanly) ──
    vocab_size: int = 256
    block_size: int = 1024
    n_layer: int = 6
    n_head: int = 6
    n_kv_head: int | None = None
    n_embd: int = 384
    dropout: float = 0.1
    rope_theta: float = 10000.0
    bias: bool = False
    grad_checkpoint: bool = False
    # ── CCCA-specific ──
    capacity_lambda: float = 0.0      # budget weight on mean(alpha)
    mask_tau: float = 2.0             # softness of the nested mask ramp (in dims)
    matryoshka_prob: float = 0.5      # prob. a train step uses a random prefix width
    uniform_width: float | None = None  # if set, constant alpha (gate disabled) → MRL-uniform baseline
    force_full: bool = False          # if True, mask == 1 (exact dense baseline / identity check)
    use_freq_width: bool = False      # if True, per-token width = freq_alpha[token] (Adaptive-Input-style baseline)
    gate_hidden_frac: float = 0.25    # gate MLP hidden size as a fraction of n_embd


class AdaptiveCapacityTransformer(Transformer):
    """Transformer with a learned, context-conditional capacity gate.

    Inherits the embedding, blocks, norm, tied head, optimizer config, parameter
    counts and generate() from Transformer; overrides only construction (to add
    the gate) and forward (to apply the mask + budget loss).
    """

    def __init__(self, config: AdaptiveConfig):
        # AdaptiveConfig carries every field ModelConfig needs, so the parent
        # constructor builds the standard stack unchanged.
        super().__init__(config)  # type: ignore[arg-type]
        self.config = config

        hidden = max(8, int(config.n_embd * config.gate_hidden_frac))
        self.gate = nn.Sequential(
            nn.Linear(config.n_embd, hidden, bias=True),
            nn.SiLU(),
            nn.Linear(hidden, 1, bias=True),
        )
        self.gate.apply(self._init_weights)
        # Start near full width (alpha ≈ 0.95) so the model first learns to model,
        # then the budget penalty squeezes capacity down.
        with torch.no_grad():
            self.gate[-1].bias.fill_(3.0)

        # per-token width fractions for the static-frequency baseline (set externally
        # from corpus byte frequencies); default ones = dense unless use_freq_width.
        self.register_buffer("freq_alpha", torch.ones(config.vocab_size))

        # diagnostics, populated each forward
        self.last_alpha: torch.Tensor | None = None
        self.last_avg_width: float | None = None
        self.last_aux_loss: torch.Tensor = torch.zeros(())

    def _capacity_mask(self, alpha: torch.Tensor, dim: int) -> torch.Tensor:
        """Soft nested prefix mask. alpha: [b, t] → mask: [b, t, dim]."""
        j = torch.arange(dim, device=alpha.device, dtype=alpha.dtype)  # [dim]
        ramp = (alpha.unsqueeze(-1) * dim - j) / self.config.mask_tau
        return torch.sigmoid(ramp)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        _, t = idx.shape
        if t > self.config.block_size:
            raise ValueError(f"sequence length {t} > block_size {self.config.block_size}")

        x = self.drop(self.token_embedding(idx))
        if self.config.grad_checkpoint and self.training:
            for block in self.blocks:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
        else:
            for block in self.blocks:
                x = block(x)
        x = self.norm(x)                       # [b, t, D]
        d = x.shape[-1]

        # ── capacity gate ──
        apply_budget = False
        if self.config.force_full:
            alpha = torch.ones(x.shape[:2], device=x.device, dtype=x.dtype)
            mask = torch.ones_like(x)
        elif self.config.use_freq_width:
            # static, context-free allocation: width set by the input token's
            # corpus frequency (Adaptive-Input / Adaptive-Softmax style baseline).
            alpha = self.freq_alpha[idx].to(x.dtype)          # [b, t]
            mask = self._capacity_mask(alpha, d)
        elif self.config.uniform_width is not None:
            alpha = x.new_full(x.shape[:2], float(self.config.uniform_width))
            mask = self._capacity_mask(alpha, d)
        else:
            alpha = torch.sigmoid(self.gate(x)).squeeze(-1)   # [b, t], the gate's own output
            apply_budget = True
            mask_alpha = alpha
            # Matryoshka prefix-robustness: some steps train the head on a random
            # prefix width so every prefix stays a usable representation.
            if self.training and self.config.matryoshka_prob > 0 \
                    and float(torch.rand(())) < self.config.matryoshka_prob:
                mask_alpha = torch.rand_like(alpha)
            mask = self._capacity_mask(mask_alpha, d)

        h = x * mask
        logits = self.lm_head(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        # ── diagnostics + budget aux loss (regularises the gate's own output) ──
        self.last_alpha = alpha.detach()
        self.last_avg_width = float(alpha.mean().item() * d)
        if apply_budget and self.config.capacity_lambda > 0 and targets is not None:
            self.last_aux_loss = self.config.capacity_lambda * alpha.mean()
        else:
            self.last_aux_loss = x.new_zeros(())

        return logits, loss
