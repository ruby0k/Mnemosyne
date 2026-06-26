"""Mamba (State Space Model) for byte-level language modeling.

Mamba processes sequences in linear time O(N) instead of O(N²) for attention.
This makes it ideal for byte-level representation where sequences are 4-6x longer
than BPE. A byte-level Mamba can handle 4096+ byte contexts efficiently.

This is a simplified Mamba implementation using the selective state space model
from Gu & Dao (2023) — arXiv: 2312.00752.

Note: This uses a non-fused Python scan loop. For production speed,
install mamba-ssm which provides fused CUDA kernels. The fallback
works correctly but is slower for long sequences.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict

from .transformer import RMSNorm


@dataclass
class MambaConfig:
    vocab_size: int = 256
    d_model: int = 256          # model dimension
    n_layer: int = 4
    d_state: int = 16           # SSM state dimension
    d_conv: int = 4             # local convolution width
    expand: int = 2             # inner dimension expansion factor
    block_size: int = 1024
    dropout: float = 0.0


class MambaBlock(nn.Module):
    """A single Mamba block: input → conv → SSM → output.

    Architecture (simplified from Mamba paper):
      1. Linear in_proj: d_model → 2*d_inner (gate + main)
      2. Conv1d on main branch (causal, depthwise)
      3. SiLU activation on main
      4. x_proj: d_inner → d_inner + 2*d_state (delta_raw, B, C)
      5. dt_proj: scalar → d_inner (projects delta_raw to per-channel timestep)
      6. Selective SSM scan
      7. Gate with SiLU(x_gate)
      8. Linear out_proj: d_inner → d_model
    """

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        d_inner = config.d_model * config.expand
        self.d_inner = d_inner
        d_state = config.d_state

        # Input projection: d_model → 2*d_inner (for gated branch)
        self.in_proj = nn.Linear(config.d_model, d_inner * 2, bias=False)

        # Depthwise causal convolution on main branch
        self.conv = nn.Conv1d(
            d_inner, d_inner,
            kernel_size=config.d_conv,
            padding=config.d_conv - 1,
            groups=d_inner,
            bias=True,
        )

        # x_proj: produces delta_raw (1 scalar) + B (d_state) + C (d_state)
        self.x_proj = nn.Linear(d_inner, 1 + 2 * d_state, bias=False)

        # dt_proj: maps delta scalar → d_inner-dimensional timestep
        self.dt_proj = nn.Linear(1, d_inner, bias=True)

        # Initialize dt_proj so initial dt is small (stable)
        self.dt_proj.weight.data.zero_()
        self.dt_proj.bias.data.fill_(0.5)

        # A parameter (log space for stability) — shape [d_inner, d_state]
        A = torch.repeat_interleave(
            torch.arange(1, d_state + 1, dtype=torch.float32), d_inner
        ).reshape(d_state, d_inner).t().contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        # Output projection
        self.out_proj = nn.Linear(d_inner, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq, d_model]
        Returns: [batch, seq, d_model]
        """
        b, t, d = x.shape

        # 1. Input projection → gated branches
        x_proj = self.in_proj(x)  # [b, t, 2*d_inner]
        x_main, x_gate = x_proj.split(self.d_inner, dim=-1)

        # 2. Causal depthwise convolution on main branch
        x_main = x_main.transpose(1, 2)  # [b, d_inner, t]
        x_main = self.conv(x_main)[:, :, :t]  # causal: trim to [b, d_inner, t]
        x_main = x_main.transpose(1, 2)  # [b, t, d_inner]
        x_main = F.silu(x_main)

        # 3. Compute SSM parameters from x_main
        ssm_in = self.x_proj(x_main)  # [b, t, 1 + 2*d_state]
        delta_raw, B, C = ssm_in.split([1, self.config.d_state, self.config.d_state], dim=-1)

        # 4. Project delta scalar → per-channel timestep
        # delta_raw: [b, t, 1] → dt_proj → [b, t, d_inner]
        delta = F.softplus(self.dt_proj(delta_raw))  # [b, t, d_inner]

        # 5. A matrix
        A = -torch.exp(self.A_log)  # [d_inner, d_state]

        # 6. Selective scan
        y = self._selective_scan(x_main, delta, A, B, C)  # [b, t, d_inner]

        # 7. Gate with x_gate
        y = y * F.silu(x_gate)
        y = self.dropout(y)

        # 8. Output projection
        return self.out_proj(y)

    def _selective_scan(self, x, delta, A, B, C):
        """Simplified selective scan (non-fused Python loop).

        Args:
            x: [b, t, d_inner] — input sequence
            delta: [b, t, d_inner] — per-channel timesteps
            A: [d_inner, d_state] — state transition matrix
            B: [b, t, d_state] — input matrices (input-dependent)
            C: [b, t, d_state] — output matrices (input-dependent)

        Returns: [b, t, d_inner]
        """
        b, t, d_inner = x.shape
        d_state = self.config.d_state

        # Discretize: dA = exp(delta * A), dB = delta * B
        # delta: [b, t, d_inner] → [b, t, d_inner, 1]
        # A: [d_inner, d_state] → [1, 1, d_inner, d_state]
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # [b, t, d_inner, d_state]

        # delta: [b, t, d_inner, 1] * B: [b, t, 1, d_state] → [b, t, d_inner, d_state]
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # [b, t, d_inner, d_state]

        # Sequential scan
        h = torch.zeros(b, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []
        for i in range(t):
            # h_{t+1} = dA_t * h_t + dB_t * x_t
            h = deltaA[:, i] * h + deltaB[:, i] * x[:, i].unsqueeze(-1)
            # y_t = C_t @ h_t + D * x_t
            y_i = (h * C[:, i].unsqueeze(1)).sum(dim=-1)  # [b, d_inner]
            ys.append(y_i)

        y = torch.stack(ys, dim=1)  # [b, t, d_inner]
        y = y + x * self.D  # skip connection via D
        return y


class MambaModel(nn.Module):
    """Mamba-based language model for byte/char-level text."""

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(MambaBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # tie
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, nonlinearity='linear')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, t = idx.shape
        if t > self.config.block_size:
            idx = idx[:, -self.config.block_size:]
            t = self.config.block_size
        x = self.drop(self.token_embedding(idx))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, **kwargs):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[0, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[-1]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id.view(1, 1)), dim=1)
        return idx

    def configure_optimizer(self, lr: float, weight_decay: float = 0.1,
                            beta1: float = 0.9, beta2: float = 0.95,
                            embed_lr_scale: float = 1.0) -> torch.optim.Optimizer:
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
        return self.token_embedding.weight.numel()

    def num_modeling_parameters(self) -> int:
        return self.num_parameters() - self.num_embedding_parameters()

    def config_dict(self) -> dict:
        return asdict(self.config)