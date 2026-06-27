"""BPE representation — standard subword tokenization (baseline).

Uses GPT-2 BPE tokenizer (50257 vocab) as the baseline to compare against.
This is what every standard LLM uses.

Key property: ~4 chars per token for English, so sequences are 4x shorter
than byte/char level. But vocab is 200x larger (50K vs 256).
"""

from pathlib import Path
import json
import numpy as np

from .base import Representation, RepConfig


class BPERepresentation(Representation):
    """Standard BPE tokenization using GPT-2 tokenizer."""

    def __init__(self, block_size: int = 256, vocab_size: int = 50257):
        super().__init__(RepConfig(
            name="bpe",
            vocab_size=vocab_size,
            block_size=block_size,
            embed_dim=1,
        ))
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import GPT2TokenizerFast
            self._tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        return self._tokenizer

    def encode(self, text: str) -> np.ndarray:
        return np.array(self.tokenizer.encode(text), dtype=np.int64)

    def decode(self, ids: np.ndarray) -> str:
        return self.tokenizer.decode(ids.tolist())

    def prepare_data(self, raw_dir: Path, out_dir: Path, max_docs: int = -1) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(raw_dir.glob("*.txt"))
        if max_docs > 0:
            files = files[:max_docs]
        split = max(1, len(files) // 10)

        train_ids, val_ids = [], []
        for i, f in enumerate(files):
            ids = self.encode(f.read_text(encoding="utf-8"))
            if i < split:
                val_ids.append(ids)
            else:
                train_ids.append(ids)

        train = np.concatenate(train_ids) if train_ids else np.array([], dtype=np.uint16)
        val = np.concatenate(val_ids) if val_ids else np.array([], dtype=np.uint16)

        # GPT-2 vocab fits in uint16 (max 50256 < 65535)
        train.astype(np.uint16).tofile(out_dir / "train.bin")
        val.astype(np.uint16).tofile(out_dir / "val.bin")

        meta = {"name": "bpe", "vocab_size": 50257, "block_size": self.block_size,
                "train_tokens": len(train), "val_tokens": len(val),
                "dtype": "uint16", "chars_per_token": 4.0}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    @staticmethod
    def load_split(data_dir: str | Path, split: str) -> np.memmap:
        return np.memmap(Path(data_dir) / f"{split}.bin", dtype=np.uint16, mode="r")

    @property
    def chars_per_token(self) -> float:
        return 4.0  # GPT-2 BPE averages ~4 chars per token for English