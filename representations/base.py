"""Abstract base class for data representations.

Every representation implements this interface so the training loop stays identical
across all experiments. The only thing that changes is how text → tensors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class RepConfig:
    """Configuration for a data representation."""
    name: str
    vocab_size: int           # number of distinct tokens/bytes/chars
    block_size: int           # sequence length the model sees
    embed_dim: int            # embedding dimension (can be >1 for continuous reps)


class Representation(ABC):
    """A data representation converts raw text to/from integer sequences (or tensors)."""

    def __init__(self, config: RepConfig):
        self.config = config

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    @property
    def block_size(self) -> int:
        return self.config.block_size

    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """Encode a string to a 1D numpy array of token/byte/char IDs."""
        ...

    @abstractmethod
    def decode(self, ids: np.ndarray) -> str:
        """Decode an array of IDs back to a string."""
        ...

    @abstractmethod
    def prepare_data(self, raw_dir: Path, out_dir: Path, max_docs: int = -1) -> dict:
        """Read raw text, encode it, write train.bin/val.bin/meta.json.
        Returns metadata dict."""
        ...

    def get_batch(self, data: np.memmap, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Standard random-slice batching. Override for patch-based or hierarchical."""
        starts = np.random.randint(0, len(data) - self.block_size - 1, size=batch_size)
        offsets = starts[:, None] + np.arange(self.block_size + 1)[None, :]
        seq = torch.from_numpy(data[offsets].astype(np.int64))
        x = seq[:, :-1].contiguous().to(device)
        y = seq[:, 1:].contiguous().to(device)
        return x, y

    def bpc_from_loss(self, loss: float) -> float:
        """Convert cross-entropy loss (nats) to bits-per-character.
        Override if the representation's token-to-char ratio differs from 1:1.
        """
        # Default: assume 1 token = 1 character (byte-level, char-level)
        return loss / np.log(2)

    @property
    def chars_per_token(self) -> float:
        """Average characters per token — used for BPC conversion."""
        return 1.0