"""N-gram byte representation — group N consecutive bytes into a single token.

Instead of a hierarchical approach like Megabyte (global+local models),
this uses a FLAT model where each "position" is a group of N bytes.

Vocab = 256^N (but in practice much smaller due to byte distribution).
For N=2: vocab=65536 (but only ~10K unique pairs in practice)
For N=3: vocab=16M (but only ~50K unique triples in practice)

We limit vocab with a frequency cutoff. This trades sequence length
for vocab size — the opposite direction from byte-level.

Key question: does grouping bytes help the model learn patterns,
or does it just recreate BPE's problems (large vocab) without BPE's
linguistic structure?
"""

from pathlib import Path
import json
import numpy as np

from .base import Representation, RepConfig


class NgramByteRepresentation(Representation):
    """N-gram byte grouping with a frequency-limited vocabulary."""

    def __init__(self, block_size: int = 512, n: int = 2, max_vocab: int = 8000):
        super().__init__(RepConfig(
            name=f"ngram{n}",
            vocab_size=max_vocab,
            block_size=block_size,
            embed_dim=1,
        ))
        self.n = n
        self._max_vocab = max_vocab
        self._ngram2id = None
        self._id2ngram = None
        self._chars_per_token = float(n)

    def _build_vocab(self, train_bytes: np.ndarray) -> None:
        """Build n-gram vocabulary from training byte data."""
        from collections import Counter

        # Count n-grams in training data
        ngram_counts = Counter()
        for i in range(0, len(train_bytes) - self.n + 1, self.n):
            ngram = tuple(train_bytes[i:i + self.n])
            ngram_counts[ngram] += 1

        # Keep top max_vocab - 2 (reserve 0=<pad>, 1=<unk_ngram>)
        most_common = ngram_counts.most_common(self._max_vocab - 2)
        self._ngram2id = {tuple([0] * self.n): 0, tuple([255] * self.n): 1}  # pad, unk
        self._id2ngram = {0: tuple([0] * self.n), 1: tuple([255] * self.n)}

        for ngram, _ in most_common:
            idx = len(self._ngram2id)
            self._ngram2id[ngram] = idx
            self._id2ngram[idx] = ngram

        actual_vocab = len(self._ngram2id)
        self.config.vocab_size = actual_vocab

    def encode_bytes(self, data: np.ndarray) -> np.ndarray:
        """Encode a byte array to n-gram token IDs."""
        if self._ngram2id is None:
            raise RuntimeError("Vocabulary not built.")

        ids = []
        for i in range(0, len(data) - self.n + 1, self.n):
            ngram = tuple(int(b) for b in data[i:i + self.n])
            ids.append(self._ngram2id.get(ngram, 1))  # 1 = unk

        return np.array(ids, dtype=np.int64)

    def encode(self, text: str) -> np.ndarray:
        raw_bytes = np.frombuffer(text.encode('utf-8'), dtype=np.uint8)
        return self.encode_bytes(raw_bytes)

    def decode(self, ids: np.ndarray) -> str:
        if self._id2ngram is None:
            raise RuntimeError("Vocabulary not built.")
        byte_list = []
        for i in ids:
            ngram = self._id2ngram.get(int(i), tuple([0] * self.n))
            byte_list.extend(ngram)
        return bytes(byte_list).decode('utf-8', errors='replace')

    def prepare_data(self, raw_dir: Path, out_dir: Path, max_docs: int = -1) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(raw_dir.glob("*.txt"))
        if max_docs > 0:
            files = files[:max_docs]
        split = max(1, len(files) // 10)

        # Read all files as bytes
        all_bytes = []
        file_bytes = []
        for f in files:
            raw = np.frombuffer(f.read_text(encoding="utf-8").encode("utf-8"), dtype=np.uint8)
            file_bytes.append(raw)
            all_bytes.append(raw)

        # Build vocab from training portion
        train_bytes = np.concatenate([b for i, b in enumerate(file_bytes) if i >= split])
        print(f"  Building {self.n}-gram vocabulary (max {self._max_vocab})...")
        self._build_vocab(train_bytes)
        actual_vocab = len(self._ngram2id)
        print(f"  Vocab size: {actual_vocab}")

        # Encode
        train_ids, val_ids = [], []
        for i, raw in enumerate(file_bytes):
            ids = self.encode_bytes(raw)
            if i < split:
                val_ids.append(ids)
            else:
                train_ids.append(ids)

        train = np.concatenate(train_ids) if train_ids else np.array([], dtype=np.uint16)
        val = np.concatenate(val_ids) if val_ids else np.array([], dtype=np.uint16)

        dtype = np.uint16 if actual_vocab > 256 else np.uint8
        train.astype(dtype).tofile(out_dir / "train.bin")
        val.astype(dtype).tofile(out_dir / "val.bin")

        total_chars = sum(len(f.read_text(encoding="utf-8")) for f in files)
        total_tokens = len(train) + len(val)
        cpt = total_chars / max(total_tokens, 1)
        self._chars_per_token = cpt

        meta = {"name": f"ngram{self.n}", "vocab_size": actual_vocab,
                "n": self.n, "block_size": self.block_size,
                "train_tokens": len(train), "val_tokens": len(val),
                "dtype": "uint16" if dtype == np.uint16 else "uint8",
                "chars_per_token": cpt}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    def load_vocab(self, data_dir: str | Path):
        """Load vocabulary from a prepared data directory."""
        vocab_path = Path(data_dir) / "vocab.json"
        if vocab_path.exists():
            raw = json.loads(vocab_path.read_text(encoding="utf-8"))
            self._ngram2id = {tuple(k): v for k, v in raw["ngram2id"]}
            self._id2ngram = {int(k): tuple(v) for k, v in raw["id2ngram"].items()}

    def bpc_from_loss(self, loss: float) -> float:
        return loss * self.chars_per_token / np.log(2)

    @property
    def chars_per_token(self) -> float:
        return self._chars_per_token