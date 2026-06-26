"""Character-level representation.

Vocab: ~100-256 depending on which characters appear in the data.
Slightly shorter sequences than byte-level for non-ASCII, but same order.
Maps each unique character to an integer ID.

Key difference from byte-level: works at the character level, so multi-byte
UTF-8 sequences (é, 漢) become single tokens rather than 2-4 bytes.
"""

from pathlib import Path
import json
import numpy as np

from .base import Representation, RepConfig


class CharRepresentation(Representation):
    """Character-level representation with a learned vocabulary."""

    def __init__(self, block_size: int = 1024):
        super().__init__(RepConfig(
            name="char",
            vocab_size=256,  # will be updated after build_vocab
            block_size=block_size,
            embed_dim=1,
        ))
        self.char_to_id: dict[str, int] = {}
        self.id_to_char: dict[int, str] = {}

    def build_vocab(self, texts: list[str]) -> None:
        chars = set()
        for text in texts:
            chars.update(text)
        sorted_chars = sorted(chars)
        self.char_to_id = {c: i for i, c in enumerate(sorted_chars)}
        self.id_to_char = {i: c for i, c in enumerate(sorted_chars)}
        self.config.vocab_size = len(sorted_chars)

    def encode(self, text: str) -> np.ndarray:
        return np.array([self.char_to_id.get(c, 0) for c in text], dtype=np.int64)

    def decode(self, ids: np.ndarray) -> str:
        return "".join(self.id_to_char.get(int(i), "") for i in ids)

    def prepare_data(self, raw_dir: Path, out_dir: Path, max_docs: int = -1) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(raw_dir.glob("*.txt"))
        if max_docs > 0:
            files = files[:max_docs]
        texts = [f.read_text(encoding="utf-8") for f in files]
        self.build_vocab(texts)

        split = max(1, len(texts) // 10)
        train_ids = np.concatenate([self.encode(t) for t in texts[split:]])
        val_ids = np.concatenate([self.encode(t) for t in texts[:split]])

        train_ids.astype(np.uint16).tofile(out_dir / "train.bin")
        val_ids.astype(np.uint16).tofile(out_dir / "val.bin")

        meta = {"name": "char", "vocab_size": self.config.vocab_size,
                "block_size": self.block_size, "train_tokens": len(train_ids),
                "val_tokens": len(val_ids), "dtype": "uint16",
                "vocab": self.char_to_id}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        return meta

    @staticmethod
    def load_split(data_dir: str | Path, split: str) -> np.memmap:
        return np.memmap(Path(data_dir) / f"{split}.bin", dtype=np.uint16, mode="r")

    @property
    def chars_per_token(self) -> float:
        return 1.0