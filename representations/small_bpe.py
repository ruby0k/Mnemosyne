"""Small-vocab BPE — train a custom BPE tokenizer with configurable vocab size.

This lets us sweep vocab size: 256 (byte) → 1K → 4K → 8K → 50K (GPT-2).
The key question: is there a sweet spot where you get enough sequence compression
to speed up training, without wasting too many params on embeddings?

For a 3M param model at 256 dim:
  vocab=256:   embed = 65K   (2.0%)
  vocab=1000:  embed = 256K  (7.7%)
  vocab=4000:  embed = 1M    (25%)
  vocab=8000:  embed = 2M    (40%)
  vocab=50000: embed = 12.8M (80%)
"""

from pathlib import Path
import json
import numpy as np

from .base import Representation, RepConfig


class SmallBPERepresentation(Representation):
    """BPE with a custom-trained tokenizer at a specified vocab size."""

    def __init__(self, block_size: int = 512, vocab_size: int = 1000):
        super().__init__(RepConfig(
            name=f"bpe{vocab_size}",
            vocab_size=vocab_size,
            block_size=block_size,
            embed_dim=1,
        ))
        self._target_vocab = vocab_size
        self._merges = None  # list of (token_a, token_b, new_id)
        self._merge_dict = None  # (token_a, token_b) -> new_id
        self._chars_per_token = 1.0

    def _train_bpe(self, texts: list[str]) -> None:
        """Train BPE using a fast pair-counting approach."""
        # Pretokenize: split into words, each word is a list of byte IDs
        word_freqs = {}
        for text in texts:
            for j, w in enumerate(text.split(' ')):
                key = w.encode('utf-8') if j == 0 else b' ' + w.encode('utf-8')
                word_freqs[key] = word_freqs.get(key, 0) + 1

        # Convert to token lists
        word_tokens = {w: list(w) for w in word_freqs}
        # Cache: for each pair, which words contain it and how many times
        pair_counts = {}
        pair_words = {}  # pair -> set of words containing it

        def count_pairs(word, tokens, freq):
            for i in range(len(tokens) - 1):
                p = (tokens[i], tokens[i + 1])
                pair_counts[p] = pair_counts.get(p, 0) + freq
                if p not in pair_words:
                    pair_words[p] = set()
                pair_words[p].add(word)

        for word, freq in word_freqs.items():
            count_pairs(word, word_tokens[word], freq)

        merges = []
        num_merges = self._target_vocab - 256

        for merge_idx in range(num_merges):
            if not pair_counts:
                break

            # Find best pair
            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < 2:
                break  # Not worth merging rare pairs
            new_id = 256 + merge_idx

            # Merge in affected words only
            affected_words = list(pair_words[best_pair])
            for word in affected_words:
                tokens = word_tokens[word]
                freq = word_freqs[word]

                # Remove old pair counts for this word
                for i in range(len(tokens) - 1):
                    p = (tokens[i], tokens[i + 1])
                    pair_counts[p] -= freq
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                        if p in pair_words:
                            del pair_words[p]

                # Do the merge
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                        new_tokens.append(new_id)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                word_tokens[word] = new_tokens

                # Add new pair counts
                for i in range(len(new_tokens) - 1):
                    p = (new_tokens[i], new_tokens[i + 1])
                    pair_counts[p] = pair_counts.get(p, 0) + freq
                    if p not in pair_words:
                        pair_words[p] = set()
                    pair_words[p].add(word)

            merges.append((best_pair[0], best_pair[1], new_id))

        self._merges = merges
        self._merge_dict = {(a, b): mid for a, b, mid in merges}

    def encode(self, text: str) -> np.ndarray:
        """Encode text using trained BPE merges (greedy, lowest merge index first)."""
        if self._merges is None:
            raise RuntimeError("Tokenizer not trained.")

        all_ids = []
        for j, w in enumerate(text.split(' ')):
            raw = list(w.encode('utf-8')) if j == 0 else [32] + list(w.encode('utf-8'))

            # Greedy BPE: repeatedly apply the highest-priority merge
            while len(raw) > 1:
                best_rank = len(self._merges)
                best_idx = -1
                for i in range(len(raw) - 1):
                    rank = self._merge_dict.get((raw[i], raw[i + 1]), len(self._merges))
                    if rank < best_rank:
                        best_rank = rank
                        best_idx = i

                if best_idx == -1:
                    break

                a, b, mid = self._merges[best_rank]
                raw = raw[:best_idx] + [mid] + raw[best_idx + 2:]

            all_ids.extend(raw)

        return np.array(all_ids, dtype=np.int64)

    def decode(self, ids: np.ndarray) -> str:
        """Decode by expanding merge IDs back to bytes."""
        if self._merges is None:
            raise RuntimeError("Tokenizer not trained.")

        merge_map = {mid: (a, b) for a, b, mid in self._merges}

        def expand(token):
            if token < 256:
                return bytes([token])
            if token in merge_map:
                a, b = merge_map[token]
                return expand(a) + expand(b)
            return b''

        return b''.join(expand(int(t)) for t in ids).decode('utf-8', errors='replace')

    def prepare_data(self, raw_dir: Path, out_dir: Path, max_docs: int = -1) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(raw_dir.glob("*.txt"))
        if max_docs > 0:
            files = files[:max_docs]
        split = max(1, len(files) // 10)

        # Train BPE on training texts
        print(f"  Training BPE tokenizer (vocab={self._target_vocab})...")
        train_texts = [f.read_text(encoding="utf-8") for i, f in enumerate(files) if i >= split]
        self._train_bpe(train_texts)
        actual_vocab = 256 + len(self._merges)
        print(f"  Actual vocab: {actual_vocab} ({len(self._merges)} merges)")

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

        total_chars = sum(len(f.read_text(encoding="utf-8")) for f in files)
        total_tokens = len(train) + len(val)
        cpt = total_chars / max(total_tokens, 1)
        self._chars_per_token = cpt

        meta = {"name": f"bpe{self._target_vocab}", "vocab_size": actual_vocab,
                "target_vocab": self._target_vocab,
                "block_size": self.block_size,
                "train_tokens": len(train), "val_tokens": len(val),
                "dtype": "uint16" if dtype == np.uint16 else "uint8",
                "chars_per_token": cpt,
                "num_merges": len(self._merges)}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (out_dir / "merges.json").write_text(json.dumps(
            [[a, b, mid] for a, b, mid in self._merges]), encoding="utf-8")

        return meta

    def load_tokenizer(self, data_dir: str | Path):
        """Load trained merges from a prepared data directory."""
        merges_path = Path(data_dir) / "merges.json"
        if merges_path.exists():
            self._merges = [tuple(m) for m in json.loads(merges_path.read_text(encoding="utf-8"))]
            self._merge_dict = {(a, b): mid for a, b, mid in self._merges}

    def bpc_from_loss(self, loss: float) -> float:
        return loss * self.chars_per_token / np.log(2)

    @property
    def chars_per_token(self) -> float:
        return self._chars_per_token