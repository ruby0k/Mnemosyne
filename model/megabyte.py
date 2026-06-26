"""Megabyte-style hierarchical model for patch-based byte representation.

Two-level architecture:
  1. Global model: processes patch-level embeddings (one vector per patch)
  2. Local model: predicts individual bytes within each patch, conditioned on
     the global representation of that patch.

BUG FIX (v2): The target shifting is now done at the byte level BEFORE reshaping
into patches. This ensures that:
  - The target for the last byte of patch i is the first byte of patch i+1
  - The local model receives the global representation of the CURRENT patch
    (which includes information from all previous patches via the global transformer)
  - Cross-patch context flows through the global model, not through local targets

Architecture improvements:
  - GQA support for global and local models
  - Gradient checkpointing option
  - Better generation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict

from .transformer import RMSNorm, apply_rope, Block, ModelConfig


@dataclass
class MegabyteConfig:
    vocab_size: int = 256           # byte vocab
    patch_size: int = 16            # bytes per patch
    global_seq_len: int = 64        # patches per sequence (= block_size / patch_size)
    # Global model (processes patches)
    global_n_layer: int = 6
    global_n_head: int = 6
    global_n_embd: int = 384
    global_n_kv_head: int | None = None  # GQA
    # Local model (predicts bytes within patches)
    local_n_layer: int = 2
    local_n_head: int = 4
    local_n_embd: int = 192
    local_n_kv_head: int | None = None   # GQA
    dropout: float = 0.1
    rope_theta: float = 10000.0
    grad_checkpoint: bool = False


class PatchEmbed(nn.Module):
    """Embeds a patch of bytes into a single vector via a small linear layer."""
    def __init__(self, vocab_size: int, patch_size: int, n_embd: int):
        super().__init__()
        # Each byte gets a small embedding, then we concatenate and project
        self.byte_embed = nn.Embedding(vocab_size, n_embd // patch_size)
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        # patch: [batch, n_patches, patch_size]
        b, p, s = patch.shape
        emb = self.byte_embed(patch)  # [batch, n_patches, patch_size, n_embd//patch_size]
        emb = emb.reshape(b, p, s * (emb.shape[-1]))  # [batch, n_patches, n_embd]
        return self.proj(emb)


class LocalModel(nn.Module):
    """Predicts bytes within a patch given the global context.

    Takes the global representation of a patch + the bytes seen so far in the patch,
    predicts the next byte. This is a small transformer operating within patches.

    Cross-patch context flows through the global model's representation — the local
    model only needs to attend within its own patch, conditioned on the global vector.
    """
    def __init__(self, config: MegabyteConfig):
        super().__init__()
        self.config = config
        self.byte_embed = nn.Embedding(config.vocab_size, config.local_n_embd)
        self.global_proj = nn.Linear(config.global_n_embd, config.local_n_embd)

        local_cfg = ModelConfig(
            vocab_size=config.vocab_size,
            block_size=config.patch_size,
            n_layer=config.local_n_layer,
            n_head=config.local_n_head,
            n_kv_head=config.local_n_kv_head,
            n_embd=config.local_n_embd,
            dropout=config.dropout,
            rope_theta=config.rope_theta,
        )
        self.blocks = nn.ModuleList(Block(local_cfg) for _ in range(config.local_n_layer))
        self.norm = RMSNorm(config.local_n_embd)
        self.lm_head = nn.Linear(config.local_n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.byte_embed.weight  # tie

    def forward(self, global_repr: torch.Tensor, patch_bytes: torch.Tensor) -> torch.Tensor:
        """
        global_repr: [batch, n_patches, global_n_embd] — from global model
        patch_bytes: [batch, n_patches, patch_size] — input bytes for each patch
        Returns: logits [batch, n_patches, patch_size, vocab_size]
        """
        b, p, s = patch_bytes.shape
        # Embed bytes
        byte_emb = self.byte_embed(patch_bytes)  # [batch, n_patches, patch_size, local_n_embd]
        # Add global context (broadcast across patch dimension)
        global_ctx = self.global_proj(global_repr).unsqueeze(2)  # [batch, n_patches, 1, local_n_embd]
        x = byte_emb + global_ctx  # [batch, n_patches, patch_size, local_n_embd]

        # Reshape to process all patches in parallel as a single batch
        x = x.reshape(b * p, s, self.config.local_n_embd)

        # Apply causal attention within each patch
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)  # [batch*n_patches, patch_size, vocab_size]

        return logits.reshape(b, p, s, self.config.vocab_size)


class MegabyteModel(nn.Module):
    """Hierarchical byte-level model: global transformer + local decoder."""

    def __init__(self, config: MegabyteConfig):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(config.vocab_size, config.patch_size, config.global_n_embd)

        # Global model: standard transformer over patch embeddings
        global_cfg = ModelConfig(
            vocab_size=config.vocab_size,  # not used (we feed pre-embedded patches)
            block_size=config.global_seq_len,
            n_layer=config.global_n_layer,
            n_head=config.global_n_head,
            n_kv_head=config.global_n_kv_head,
            n_embd=config.global_n_embd,
            dropout=config.dropout,
            rope_theta=config.rope_theta,
            grad_checkpoint=config.grad_checkpoint,
        )
        self.global_blocks = nn.ModuleList(Block(global_cfg) for _ in range(config.global_n_layer))
        self.global_norm = RMSNorm(config.global_n_embd)

        # Local model: predicts bytes within patches
        self.local_model = LocalModel(config)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        x: [batch, n_patches, patch_size] — input byte patches
        targets: [batch, n_patches, patch_size] — target byte patches (shifted by 1 byte)
        """
        b, p, s = x.shape

        # Global model: embed patches → transformer → global representations
        patch_emb = self.patch_embed(x)  # [batch, n_patches, global_n_embd]
        global_x = patch_emb
        if self.config.grad_checkpoint and self.training:
            for block in self.global_blocks:
                global_x = torch.utils.checkpoint.checkpoint(block, global_x, use_reentrant=False)
        else:
            for block in self.global_blocks:
                global_x = block(global_x)
        global_x = self.global_norm(global_x)  # [batch, n_patches, global_n_embd]

        # Local model: predict bytes within each patch
        logits = self.local_model(global_x, x)  # [batch, n_patches, patch_size, vocab_size]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.config.vocab_size), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, x: torch.Tensor, max_new_bytes: int, temperature: float = 0.8, top_k: int = 50) -> torch.Tensor:
        """Generate bytes. x: [1, n_patches, patch_size] → appends new bytes."""
        for _ in range(max_new_bytes):
            logits, _ = self(x)  # [1, n_patches, patch_size, vocab_size]
            # Take last patch, last position
            next_logits = logits[0, -1, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < values[-1]] = -float("inf")
            probs = F.softmax(next_logits, dim=-1)
            next_byte = torch.multinomial(probs, num_samples=1)
            # Append to last patch (simplified — full implementation would manage patch boundaries)
            next_byte = next_byte.view(1, 1, 1)
            x = torch.cat([x, next_byte], dim=2)  # grow patch
        return x

    def configure_optimizer(self, lr: float, weight_decay: float = 0.1,
                            beta1: float = 0.9, beta2: float = 0.95,
                            embed_lr_scale: float = 1.0) -> torch.optim.Optimizer:
        """Build AdamW with separate param groups."""
        import inspect
        decay, nodecay = [], []
        seen = set()
        for name, p in self.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            if p.ndim < 2:
                nodecay.append(p)
            else:
                decay.append(p)

        adamw_kwargs = dict(betas=(beta1, beta2))
        if "fused" in inspect.signature(torch.optim.AdamW).parameters and torch.cuda.is_available():
            adamw_kwargs["fused"] = True

        param_groups = [
            {"params": decay, "weight_decay": weight_decay, "lr": lr},
            {"params": nodecay, "weight_decay": 0.0, "lr": lr},
        ]
        optimizer = torch.optim.AdamW(param_groups, **adamw_kwargs)
        for g in optimizer.param_groups:
            g["initial_lr"] = g["lr"]
        return optimizer

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_embedding_parameters(self) -> int:
        return self.patch_embed.byte_embed.weight.numel() + self.local_model.byte_embed.weight.numel()

    def num_modeling_parameters(self) -> int:
        return self.num_parameters() - self.num_embedding_parameters()

    def config_dict(self) -> dict:
        return asdict(self.config)