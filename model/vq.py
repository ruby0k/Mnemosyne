"""Vector-quantized embedding substrate.

The discrete arm of the representation rate-distortion study. We hold the
tokenizer (byte) and transformer body fixed and replace the *continuous* final
hidden state with its **product-quantized** reconstruction before the prediction
head. The representation rate is then a controllable knob:

    rate = n_groups * log2(codebook_size)   bits / token

This is the discrete counterpart to CCCA / Matryoshka, which reduce the rate by
dropping real-valued dimensions. Comparing the two answers: is it cheaper (in
BPC) to spend representation bits as fewer dimensions or as coarser quantization?

Product quantization splits the D-dim hidden into `n_groups` subvectors and
quantizes each against its own codebook of `codebook_size` entries, with a
straight-through estimator so gradients still reach the encoder.

Codebooks are trained with **EMA updates + data-dependent init + dead-code
revival** — the standard anti-collapse recipe (a naive gradient-updated codebook
collapses to a single code per group). `vq_enabled=False` is the exact dense
baseline (identity), used as the BPC floor / identity check.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import Transformer, ModelConfig


@dataclass
class VQConfig:
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
    # ── VQ-specific ──
    vq_enabled: bool = True
    n_groups: int = 8            # split D into this many subvectors (product quantization)
    codebook_size: int = 256     # codes per group
    commitment_beta: float = 0.25
    ema_decay: float = 0.99
    revive_threshold: float = 1.0  # codes with EMA count below this are reinitialized


class VQTransformer(Transformer):
    """Transformer with a product-VQ bottleneck on the final hidden state.

    Inherits embedding, blocks, norm, tied head, optimizer config, parameter
    counts and generate() from Transformer; overrides construction (codebooks +
    EMA buffers) and forward (quantize before the head + commitment loss).
    """

    def __init__(self, config: VQConfig):
        super().__init__(config)  # type: ignore[arg-type]
        self.config = config
        assert config.n_embd % config.n_groups == 0, "n_embd must be divisible by n_groups"
        G, K, sub = config.n_groups, config.codebook_size, config.n_embd // config.n_groups
        self.sub = sub

        # Codebooks are EMA buffers (not gradient parameters).
        self.register_buffer("codebook", torch.randn(G, K, sub) * 0.02)
        self.register_buffer("ema_count", torch.zeros(G, K))
        self.register_buffer("ema_sum", self.codebook.clone())
        self.register_buffer("initted", torch.zeros((), dtype=torch.bool))

        self.rate_bits = G * math.log2(K)
        self.last_aux_loss: torch.Tensor = torch.zeros(())
        self.last_code_perplexity: float | None = None

    @torch.no_grad()
    def _data_init(self, hg_flat):
        """Initialize each group's codebook from random encoder vectors."""
        G, K = self.config.n_groups, self.config.codebook_size
        for g in range(G):
            pool = hg_flat[g]                                   # [N, sub]
            n = pool.shape[0]
            sel = torch.randint(0, n, (K,), device=pool.device)
            self.codebook[g] = pool[sel]
        self.ema_sum.copy_(self.codebook)
        self.ema_count.fill_(1.0)
        self.initted.fill_(True)

    @torch.no_grad()
    def _ema_update(self, hg_flat, idx_flat):
        """EMA codebook update + dead-code revival (per group)."""
        G, K = self.config.n_groups, self.config.codebook_size
        decay, eps, thr = self.config.ema_decay, 1e-5, self.config.revive_threshold
        for g in range(G):
            x = hg_flat[g]                                      # [N, sub]
            onehot = F.one_hot(idx_flat[g], K).type_as(x)       # [N, K]
            counts = onehot.sum(0)                              # [K]
            sums = onehot.t() @ x                               # [K, sub]
            self.ema_count[g].mul_(decay).add_(counts, alpha=1 - decay)
            self.ema_sum[g].mul_(decay).add_(sums, alpha=1 - decay)
            n = self.ema_count[g].sum()
            cluster = (self.ema_count[g] + eps) / (n + K * eps) * n
            self.codebook[g] = self.ema_sum[g] / cluster.unsqueeze(1)
            # revive dead codes with random current encoder vectors
            dead = self.ema_count[g] < thr
            n_dead = int(dead.sum())
            if n_dead > 0:
                sel = torch.randint(0, x.shape[0], (n_dead,), device=x.device)
                self.codebook[g][dead] = x[sel]
                self.ema_sum[g][dead] = x[sel]
                self.ema_count[g][dead] = 1.0

    def _quantize(self, h: torch.Tensor):
        """Product-quantize h: [b, t, D]. Returns (h_st, vq_loss, perplexity)."""
        b, t, d = h.shape
        G, K, sub = self.config.n_groups, self.config.codebook_size, self.sub
        # per-group encoder vectors, in fp32 for stable codebook math
        hg = h.view(b, t, G, sub).permute(2, 0, 1, 3).reshape(G, b * t, sub).float()  # [G, N, sub]
        cb = self.codebook                                              # [G, K, sub] fp32

        if self.training and not bool(self.initted):
            self._data_init(hg)

        # nearest code per group: dist [G, N, K]
        dist = (hg.unsqueeze(2) - cb.unsqueeze(1)).pow(2).sum(-1)
        idx = dist.argmin(-1)                                           # [G, N]
        zq = torch.stack([cb[g][idx[g]] for g in range(G)], dim=0)      # [G, N, sub]

        if self.training:
            self._ema_update(hg, idx)

        # reshape zq back to [b, t, D]
        zq_btd = zq.reshape(G, b, t, sub).permute(1, 2, 0, 3).reshape(b, t, d).to(h.dtype)

        # commitment loss only (codebook learns via EMA)
        commit_loss = (h - zq_btd.detach()).pow(2).mean()
        vq_loss = self.config.commitment_beta * commit_loss

        # straight-through
        h_st = h + (zq_btd - h).detach()

        # codebook usage perplexity, averaged over groups (collapse diagnostic)
        with torch.no_grad():
            perp = 0.0
            for g in range(G):
                p = torch.bincount(idx[g], minlength=K).float()
                p = p / p.sum().clamp_min(1)
                perp += float((-(p * p.clamp_min(1e-9).log()).sum()).exp())
            perp /= G

        return h_st, vq_loss, perp

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
        x = self.norm(x)                          # [b, t, D]

        if self.config.vq_enabled:
            h, vq_loss, perp = self._quantize(x)
            self.last_aux_loss = vq_loss
            self.last_code_perplexity = perp
        else:
            h = x
            self.last_aux_loss = x.new_zeros(())
            self.last_code_perplexity = None

        logits = self.lm_head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
