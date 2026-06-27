"""Word-level representation — tokenize on whitespace.

The simplest tokenization: split on spaces. Each word is a token.
Vocabulary is built from the training data.

Key property: ~5 chars per token (longest sequences compressed most),
but vocab can be very large (thousands of unique words even in TinyStories).
Unknown words at val time get a special <unk> token.

This is the historical baseline — what NLP used before subword tokenization.
"""

from pathlib import Path
import json
import numpy as np

from .base import Representation, RepConfig


class WordRepresentation(Representation):
    """Whitespace tokenization with a learned vocabulary."""

    def __init__(self, block_size: int = 256, max_vocab: int = 10000):
        super().__init__(RepConfig(
            name="word",
            vocab_size=max_vocab,
            block_size=block_size,
            embed_dim=1,
        ))
        self._max_vocab = max_vocab
        self._word2id = None
        self._id2word = None
        self._chars_per_token = 5.0  # default, updated after prepare

    def _build_vocab(self, texts: list[str]) -> None:
        """Build vocabulary from training texts, keeping most frequent words."""
        from collections import Counter
        word_counts = Counter()
        for text in texts:
            # Split on whitespace, keep whitespace attached to following word
            words = text.split()
            word_counts.update(words)

        # Sort by frequency, take top max_vocab - 2 (reserve 0=<pad>, 1=<unk>)
        most_common = word_counts.most_common(self._max_vocab - 2)
        self._word2id = {"<pad>": 0, "<unk>": 1}
        self._id2word = {0: "<pad>", 1: "<unk>"}
        for word, _ in most_common:
            idx = len(self._word2id)
            self._word2id[word] = idx
            self._id2word[idx] = word

        actual_vocab = len(self._word2id)
        self.config.vocab_size = actual_vocab

    def encode(self, text: str) -> np.ndarray:
        if self._word2id is None:
            raise RuntimeError("Vocabulary not built. Call prepare_data first.")
        words = text.split()
        ids = [self._word2id.get(w, 1) for w in words]  # 1 = <unk>
        return np.array(ids, dtype=np.int64)

    def decode(self, ids: np.ndarray) -> str:
        if self._id2word is None:
            raise RuntimeError("Vocabulary not built.")
        words = [self._id2word.get(int(i), "<unk>") for i in ids]
        return " ".join(words)

    def prepare_data(self, raw_dir: Path, out_dir: Path, max_docs: int = -1) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(raw_dir.glob("*.txt"))
        if max_docs > 0:
            files = files[:max_docs]
        split = max(1, len(files) // 10)

        # Build vocab from training texts
        print(f"  Building word vocabulary (max {self._max_vocab})...")
        train_texts = [f.read_text(encoding="utf-8") for i, f in enumerate(files) if i >= split]
        self._build_vocab(train_texts)
        actual_vocab = len(self._word2id)
        print(f"  Vocab size: {actual_vocab}")

        # Encode all data
        train_ids, val_ids = [], []
        for i, f in enumerate(files):
            ids = self.encode(f.read_text(encoding="utf-8"))
            if i < split:
                val_ids.append(ids)
            else:
                train_ids.append(ids)

        train = np.concatenate(train_ids) if train_ids else np.array([], dtype=np.uint16)
        val = np.concatenate(val_ids) if val_ids else np.array([], dtype=np.uint16)

        dtype = np.uint16 if actual_vocab > 256 else np.uint8
        train.astype(dtype).tofile(out_dir / "train.bin")
        val.astype(dtype).tofile(out_dir / "val.bin")

        # Compute chars-per-token
        total_chars = sum(len(f.read_text(encoding="utf-8")) for f in files)
        total_tokens = len(train) + len(val)
        cpt = total_chars / max(total_tokens, 1)
        self._chars_per_token = cpt

        meta = {"name": "word", "vocab_size": actual_vocab, "block_size": self.block_size,
                "train_tokens": len(train), "val_tokens": len(val),
                "dtype": "uint16" if dtype == np.uint16 else "uint8",
                "chars_per_token": cpt}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Save vocab
        (out_dir / "vocab.json").write_text(json.dumps(self._word2id), encoding="utf-8")

        return meta

    def load_vocab(self, data_dir: str | Path):
        """Load vocabulary from a prepared data directory."""
        vocab_path = Path(data_dir) / "vocab.json"
        if vocab_path.exists():
            self._word2id = json.loads(vocab_path.read_text(encoding="utf-8"))
            self._id2word = {v: k for k, v in self._word2id.items()}

    def bpc_from_loss(self, loss: float) -> float:
        return loss * self.chars_per_token / np.log(2)

    @property
    def chars_per_token(self) -> float:
        return self._chars_per_token