"""Patch-based byte representation (Megabyte-inspired).

Instead of treating each byte as a separate token, group bytes into patches.
This reduces sequence length by `patch_size` while keeping the tiny 256 vocab.

Architecture: two-level model
  - Global model: processes patch-level embeddings (N_patches tokens)
  - Local model: predicts bytes within each patch conditioned on global context

FIX (v2): get_batch now shifts at the byte level BEFORE reshaping into patches.
The input and targets are both [batch, n_patches, patch_size], where targets
are the input shifted by 1 byte. This means:
  - The last byte of patch i predicts the first byte of patch i+1
  - The global model processes all patches, so cross-patch context flows
    through the global representation
  - The local model only predicts within-patch, conditioned on global context

Example: 1024 bytes → 64 patches of 16 bytes each.
  Global model sees 64 tokens. Local model sees 16 bytes per patch.
"""

from pathlib import Path
import json
import numpy as np
import torch

from .base import Representation, RepConfig


class PatchRepresentation(Representation):
    """Byte patches — groups of `patch_size` bytes treated as a unit."""

    def __init__(self, block_size: int = 1024, patch_size: int = 16):
        self.patch_size = patch_size
        super().__init__(RepConfig(
            name="patch",
            vocab_size=256,
            block_size=block_size,
            embed_dim=patch_size,  # each "token" is a patch of bytes
        ))
        # Global sequence length = block_size / patch_size
        self.global_seq_len = block_size // patch_size

    def encode(self, text: str) -> np.ndarray:
        raw = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int64)
        # Pad to multiple of patch_size
        pad_len = self.patch_size - (len(raw) % self.patch_size) if len(raw) % self.patch_size else 0
        if pad_len:
            raw = np.concatenate([raw, np.zeros(pad_len, dtype=np.int64)])
        return raw  # stored as flat byte array, reshaped during batching

    def decode(self, ids: np.ndarray) -> str:
        return bytes(ids.astype(np.uint8).tolist()).decode("utf-8", errors="replace")

    def prepare_data(self, raw_dir: Path, out_dir: Path, max_docs: int = -1) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(raw_dir.glob("*.txt"))
        if max_docs > 0:
            files = files[:max_docs]
        split = max(1, len(files) // 10)

        train_ids, val_ids = [], []
        for i, f in enumerate(files):
            data = self.encode(f.read_text(encoding="utf-8"))
            if i < split:
                val_ids.append(data)
            else:
                train_ids.append(data)

        train = np.concatenate(train_ids) if train_ids else np.array([], dtype=np.uint8)
        val = np.concatenate(val_ids) if val_ids else np.array([], dtype=np.uint8)

        train.astype(np.uint8).tofile(out_dir / "train.bin")
        val.astype(np.uint8).tofile(out_dir / "val.bin")

        meta = {"name": "patch", "vocab_size": 256, "block_size": self.block_size,
                "patch_size": self.patch_size, "global_seq_len": self.global_seq_len,
                "train_tokens": len(train), "val_tokens": len(val),
                "dtype": "uint8"}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    @staticmethod
    def load_split(data_dir: str | Path, split: str) -> np.memmap:
        return np.memmap(Path(data_dir) / f"{split}.bin", dtype=np.uint8, mode="r")

    def get_batch(self, data: np.memmap, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of byte patches.

        Returns (x, y) where:
          x: [batch, global_seq_len, patch_size] — input byte patches
          y: [batch, global_seq_len, patch_size] — target byte patches (shifted by 1 byte)

        The shift is done at the byte level BEFORE reshaping into patches.
        This ensures the target for byte i is byte i+1, even across patch boundaries.
        The global model handles cross-patch context; the local model predicts
        within-patch bytes conditioned on the global representation.
        """
        total_bytes = self.block_size  # total bytes per sample
        # We need total_bytes + 1 to create shifted targets
        starts = np.random.randint(0, len(data) - total_bytes - 1, size=batch_size)
        # Gather [batch, total_bytes + 1]
        offsets = starts[:, None] + np.arange(total_bytes + 1)[None, :]
        seq = torch.from_numpy(data[offsets].astype(np.int64))  # [batch, total_bytes+1]

        # Shift at byte level: input = bytes 0..total_bytes-1, target = bytes 1..total_bytes
        x_flat = seq[:, :-1]  # [batch, total_bytes]
        y_flat = seq[:, 1:]   # [batch, total_bytes]

        # Reshape into patches
        x = x_flat.reshape(batch_size, self.global_seq_len, self.patch_size)
        y = y_flat.reshape(batch_size, self.global_seq_len, self.patch_size)

        if device == "cuda":
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)

    @property
    def chars_per_token(self) -> float:
        return 1.0  # still byte-level, just grouped