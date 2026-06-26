"""Data preparation — prepares TinyStories in all representations.

Downloads TinyStories, writes .txt files, then encodes into each representation.
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from datasets import load_dataset


def download_tinystories(raw_dir: Path, max_docs: int = 5000) -> None:
    """Download TinyStories and save as individual .txt files."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("roneneldan/TinyStories", split="train")
    n = min(max_docs, len(ds)) if max_docs > 0 else len(ds)
    print(f"Downloading {n} TinyStories documents...")
    for i in range(n):
        (raw_dir / f"{i:06d}.txt").write_text(ds[i]["text"], encoding="utf-8")
        if i % 1000 == 0 and i > 0:
            print(f"  {i}/{n}")
    print(f"  Saved {n} documents to {raw_dir}")


def prepare_all_representations(raw_dir: Path, data_root: Path, max_docs: int = -1) -> None:
    """Prepare data for all representations (or a filtered subset)."""
    from representations.byte import ByteRepresentation
    from representations.char import CharRepresentation
    from representations.bpe import BPERepresentation
    from representations.patch import PatchRepresentation
    from representations.small_bpe import SmallBPERepresentation
    from representations.word import WordRepresentation
    from representations.bpe_dropout import BPEDropoutRepresentation
    from representations.ngram import NgramByteRepresentation

    reps = [
        # Original 4
        ("byte", ByteRepresentation(block_size=1024)),
        ("char", CharRepresentation(block_size=1024)),
        ("bpe", BPERepresentation(block_size=256)),
        ("patch", PatchRepresentation(block_size=1024, patch_size=16)),
        # New representations
        ("bpe1000", SmallBPERepresentation(block_size=512, vocab_size=1000)),
        ("bpe4000", SmallBPERepresentation(block_size=384, vocab_size=4000)),
        ("bpe8000", SmallBPERepresentation(block_size=320, vocab_size=8000)),
        ("word", WordRepresentation(block_size=256, max_vocab=8000)),
        ("bpe_dropout", BPEDropoutRepresentation(block_size=256, dropout_prob=0.1)),
        ("ngram2", NgramByteRepresentation(block_size=512, n=2, max_vocab=8000)),
        ("ngram3", NgramByteRepresentation(block_size=384, n=3, max_vocab=8000)),
        # Mamba uses byte-level data (same as byte) but with longer context
        ("mamba", ByteRepresentation(block_size=2048)),
    ]

    # Filter reps if _reps_filter is set
    reps_filter = globals().get("_reps_filter", None)
    if reps_filter:
        reps = [(name, rep) for name, rep in reps if name in reps_filter]

    for name, rep in reps:
        out_dir = data_root / name
        print(f"\nPreparing {name} → {out_dir}")
        meta = rep.prepare_data(raw_dir, out_dir, max_docs=max_docs)
        print(f"  vocab={meta['vocab_size']}, train={meta['train_tokens']:,}, val={meta['val_tokens']:,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw", help="Where to save raw .txt files")
    parser.add_argument("--data-root", default="data", help="Root for representation data")
    parser.add_argument("--max-docs", type=int, default=5000, help="Number of documents (0=all)")
    parser.add_argument("--skip-download", action="store_true", help="Skip download if already have raw data")
    parser.add_argument("--reps", type=str, default=None,
                        help="Comma-separated list of reps to prepare (default: all)")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    data_root = Path(args.data_root)

    if not args.skip_download:
        download_tinystories(raw_dir, max_docs=args.max_docs)

    # Filter reps if specified
    global _reps_filter
    _reps_filter = args.reps.split(",") if args.reps else None

    prepare_all_representations(raw_dir, data_root, max_docs=args.max_docs)
    print("\n✓ All representations prepared.")


if __name__ == "__main__":
    main()