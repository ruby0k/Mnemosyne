"""Byte-level representation — train directly on raw UTF-8 bytes.

Vocab size: 256 (one per byte value)
No tokenizer needed. Every text file is already a byte sequence.

Key insight: vocab=256 means embedding table is tiny (256 × dim),
freeing almost all parameters for actual modeling. But sequences are
~4-5x longer than BPE for the same text, so you need longer context
or hierarchical processing.

Pros: no tokenizer, language-agnostic, works on any binary data.
Cons: long sequences, harder to learn word-level patterns.
"""

from pathlib import Path
import json
import numpy as np

from .base import Representation, RepConfig


class ByteRepresentation(Representation):
    """Raw UTF-8 byte-level representation."""

    def __init__(self, block_size: int = 1024):
        super().__init__(RepConfig(
            name="byte",
            vocab_size=256,
            block_size=block_size,
            embed_dim=1,
        ))

    def encode(self, text: str) -> np.ndarray:
        return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int64)

    def decode(self, ids: np.ndarray) -> str:
        return bytes(ids.astype(np.uint8).tolist()).decode("utf-8", errors="replace")

    def prepare_data(self, raw_dir: Path, out_dir: Path, max_docs: int = -1) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        train_ids, val_ids = [], []

        files = sorted(raw_dir.glob("*.txt"))
        if max_docs > 0:
            files = files[:max_docs]
        split = max(1, len(files) // 10)  # 90/10 split

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

        meta = {"name": "byte", "vocab_size": 256, "block_size": self.block_size,
                "train_tokens": len(train), "val_tokens": len(val),
                "dtype": "uint8"}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        return meta

    @staticmethod
    def load_split(data_dir: str | Path, split: str) -> np.memmap:
        return np.memmap(Path(data_dir) / f"{split}.bin", dtype=np.uint8, mode="r")

    @property
    def chars_per_token(self) -> float:
        return 1.0  # 1 byte ≈ 1 char for ASCII; ~1 for UTF-8 English