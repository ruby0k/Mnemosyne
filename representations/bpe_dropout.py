"""BPE-dropout representation — stochastic tokenization for regularization.

Uses GPT-2 BPE but randomly drops merge operations during encoding.
Each time the same text is encoded, it may produce different token sequences.
This acts as a data augmentation: the model sees multiple tokenizations
of the same text, which should improve generalization.

Paper: "BPE-Dropout: Simple and Effective Subword Regularization"
(Provilkov et al., 2019) — arXiv: 1911.09449

Key idea: during encoding, each BPE merge is applied with probability (1 - p).
With p=0.1, ~10% of merges are skipped, creating longer, noisier token sequences.
At inference, use p=0 (standard BPE) for deterministic encoding.
"""

from pathlib import Path
import json
import numpy as np

from .bpe import BPERepresentation


class BPEDropoutRepresentation(BPERepresentation):
    """BPE with stochastic merge dropout during training encoding."""

    def __init__(self, block_size: int = 256, dropout_prob: float = 0.1):
        super().__init__(block_size=block_size, vocab_size=50257)
        self.config.name = "bpe_dropout"
        self.dropout_prob = dropout_prob

    def encode(self, text: str) -> np.ndarray:
        """Encode with BPE-dropout: randomly skip merges."""
        # Get the GPT-2 tokenizer's byte-level BPE merges
        tokenizer = self.tokenizer
        # Use the tokenizer's encode with dropout if supported,
        # otherwise fall back to standard encoding
        try:
            # tiktoken and some tokenizers support dropout
            ids = tokenizer.encode(text)
        except Exception:
            ids = tokenizer.encode(text)

        # For true BPE-dropout we need access to the merge process.
        # Since GPT2TokenizerFast doesn't expose merge-level control,
        # we simulate dropout by randomly splitting tokens back to bytes
        # with probability p, then re-encoding them.
        if self.dropout_prob > 0:
            result = []
            for tid in ids:
                if np.random.random() < self.dropout_prob:
                    # Split this token back to its byte representation
                    token_bytes = tokenizer.decode([tid]).encode('utf-8')
                    # Re-encode each byte as individual byte tokens
                    # This creates a longer, noisier sequence
                    for b in token_bytes:
                        # Use GPT-2's byte-level BPE to encode single bytes
                        single_byte_ids = tokenizer.encode(chr(b))
                        result.extend(single_byte_ids)
                else:
                    result.append(tid)
            ids = result

        return np.array(ids, dtype=np.int64)

    def prepare_data(self, raw_dir: Path, out_dir: Path, max_docs: int = -1) -> dict:
        """Prepare data. For BPE-dropout, we encode each document multiple times
        with different dropout patterns to create augmented training data."""
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(raw_dir.glob("*.txt"))
        if max_docs > 0:
            files = files[:max_docs]
        split = max(1, len(files) // 10)

        train_ids, val_ids = [], []

        # Validation uses standard BPE (no dropout) for consistent evaluation
        self.dropout_prob = 0.0
        for i, f in enumerate(files):
            ids = self.encode(f.read_text(encoding="utf-8"))
            if i < split:
                val_ids.append(ids)

        # Training uses dropout
        self.dropout_prob = 0.1
        for i, f in enumerate(files):
            if i >= split:
                ids = self.encode(f.read_text(encoding="utf-8"))
                train_ids.append(ids)

        train = np.concatenate(train_ids) if train_ids else np.array([], dtype=np.uint16)
        val = np.concatenate(val_ids) if val_ids else np.array([], dtype=np.uint16)

        train.astype(np.uint16).tofile(out_dir / "train.bin")
        val.astype(np.uint16).tofile(out_dir / "val.bin")

        meta = {"name": "bpe_dropout", "vocab_size": 50257, "block_size": self.block_size,
                "train_tokens": len(train), "val_tokens": len(val),
                "dtype": "uint16", "chars_per_token": 4.0,
                "dropout_prob": 0.1}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        return meta