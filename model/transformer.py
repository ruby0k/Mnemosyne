"""Small GPT-style transformer for representation experiments.

Configurable: works with any vocab size (256 bytes, 100 chars, 50K BPE).
Designed to be ~3-30M params depending on config, for fast training on 8GB VRAM.

Features ported from Calliope:
  - GQA (Grouped Query Attention) via n_kv_head
  - Gradient checkpointing (toggle per config)
  - Better generate() with top-p, repetition penalty, no-repeat-ngram
  - configure_optimizer() with separate weight-decay groups
  - Fused AdamW on CUDA
"""

import inspect
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    vocab_size: int = 256
    block_size: int = 1024
    n_layer: int = 6
    n_head: int = 6
    n_kv_head: int | None = None       # GQA: if None, defaults to n_head
    n_embd: int = 384
    dropout: float = 0.1
    rope_theta: float = 10000.0
    bias: bool = False
    grad_checkpoint: bool = False       # gradient checkpointing for memory savings


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x


def apply_rope(x: torch.Tensor, theta: float) -> torch.Tensor:
    _, _, t, d = x.shape
    half = d // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, device=x.device).float() / half))
    angles = torch.outer(torch.arange(t, device=x.device).float(), freqs).to(x.dtype)
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        n_kv_head = config.n_kv_head or config.n_head
        assert config.n_head % n_kv_head == 0
        self.n_head = config.n_head
        self.n_kv_head = n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.rope_theta = config.rope_theta
        self.dropout = config.dropout
        self.c_attn = nn.Linear(config.n_embd, config.n_embd + 2 * n_kv_head * self.head_dim, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.c_attn(x).split((c, self.n_kv_head * self.head_dim, self.n_kv_head * self.head_dim), dim=2)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, self.rope_theta), apply_rope(k, self.rope_theta)
        if self.n_kv_head != self.n_head:
            repeat = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.c_proj(y))


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden = int(8 * config.n_embd / 3)
        self.w1 = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.w2 = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.w3 = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.rms_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.rms_2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.rms_1(x))
        x = x + self.mlp(self.rms_2(x))
        return x


# ── Generation helpers (ported from Calliope) ──────────────────────────────────

def apply_repetition_penalty(logits: torch.Tensor, generated_ids: list[int], penalty: float) -> torch.Tensor:
    if penalty == 1.0:
        return logits
    for token_id in set(generated_ids):
        logits[token_id] = logits[token_id] / penalty if logits[token_id] > 0 else logits[token_id] * penalty
    return logits


def banned_ngram_tokens(generated: list[int], ngram_size: int) -> set[int]:
    if ngram_size <= 0 or len(generated) < ngram_size - 1:
        return set()
    prefix = tuple(generated[-(ngram_size - 1):])
    banned = set()
    for i in range(len(generated) - ngram_size + 1):
        ngram = tuple(generated[i:i + ngram_size])
        if ngram[:-1] == prefix:
            banned.add(ngram[-1])
    return banned


def top_p_filter(logits: torch.Tensor, top_p: float | None) -> torch.Tensor:
    if top_p is None or top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    remove = torch.cumsum(probs, dim=-1) > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    logits[sorted_idx[remove]] = -float("inf")
    return logits


class Transformer(nn.Module):
    """Standard autoregressive transformer LM."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, t = idx.shape
        if t > self.config.block_size:
            raise ValueError(f"sequence length {t} > block_size {self.config.block_size}")
        x = self.drop(self.token_embedding(idx))

        if self.config.grad_checkpoint and self.training:
            # Gradient checkpointing: recompute activations during backward to save memory
            for block in self.blocks:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
        else:
            for block in self.blocks:
                x = block(x)

        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 50,
        top_p: float | None = 0.92,
        repetition_penalty: float = 1.15,
        no_repeat_ngram_size: int = 3,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        generated_ids = idx[0].tolist()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[0, -1, :] / temperature
            logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)
            for token_id in banned_ngram_tokens(generated_ids, no_repeat_ngram_size):
                logits[token_id] = -float("inf")
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[-1]] = -float("inf")
            logits = top_p_filter(logits, top_p)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            token = int(next_id)
            generated_ids.append(token)
            idx = torch.cat((idx, next_id.view(1, 1)), dim=1)
            if eos_token_id is not None and token == eos_token_id:
                break
        return idx

    def configure_optimizer(self, lr: float, weight_decay: float = 0.1,
                            beta1: float = 0.9, beta2: float = 0.95,
                            embed_lr_scale: float = 1.0) -> torch.optim.Optimizer:
        """Build AdamW with separate param groups: embeddings (maybe different LR), 2D (decay), 1D (no decay)."""
        decay, nodecay = [], []
        seen = set()
        for name, p in self.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            is_embed = "token_embedding" in name or "lm_head" in name
            if p.ndim < 2:
                nodecay.append(p)
            elif is_embed:
                decay.append(p)
            else:
                decay.append(p)

        adamw_kwargs = dict(betas=(beta1, beta2))
        if "fused" in inspect.signature(torch.optim.AdamW).parameters and torch.cuda.is_available():
            adamw_kwargs["fused"] = True

        param_groups = [
            {"params": decay, "weight_decay": weight_decay, "lr": lr * embed_lr_scale if embed_lr_scale != 1.0 else lr},
            {"params": nodecay, "weight_decay": 0.0, "lr": lr},
        ]
        # If embed_lr_scale is 1.0, just use single LR
        if embed_lr_scale == 1.0:
            for g in param_groups:
                g["lr"] = lr

        optimizer = torch.optim.AdamW(param_groups, **adamw_kwargs)
        for g in optimizer.param_groups:
            g["initial_lr"] = g["lr"]
        return optimizer

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_embedding_parameters(self) -> int:
        return self.token_embedding.weight.numel()

    def num_modeling_parameters(self) -> int:
        """Non-embedding parameters — the actual 'modeling' capacity."""
        return self.num_parameters() - self.num_embedding_parameters()

    def config_dict(self) -> dict:
        return asdict(self.config)