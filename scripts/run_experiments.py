"""Run all representation experiments back-to-back and compare results.

Each experiment trains a small model on the same data with the same budget.
The only variable is the data representation.
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train import train_one_representation


REPS = ["byte", "char", "bpe", "patch",
        "bpe1000", "bpe4000", "bpe8000",
        "word", "bpe_dropout", "ngram2", "ngram3"]


def run_all(data_root: str, out_dir: str, max_iters: int, batch_size: int,
            n_layer: int, n_head: int, n_embd: int, lr: float, seed: int,
            eval_iters: int = 20, eval_interval: int = 200):
    all_metrics = {}

    for rep_name in REPS:
        data_dir = str(Path(data_root) / rep_name)
        if not (Path(data_dir) / "train.bin").exists():
            print(f"⚠ Skipping {rep_name} — data not found at {data_dir}")
            continue

        metrics = train_one_representation(
            rep_name=rep_name,
            data_dir=data_dir,
            out_dir=out_dir,
            max_iters=max_iters,
            batch_size=batch_size,
            eval_iters=eval_iters,
            eval_interval=eval_interval,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            learning_rate=lr,
            seed=seed,
        )
        all_metrics[rep_name] = metrics

    # Save combined results
    out_path = Path(out_dir) / "comparison.json"
    out_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"\n{'='*60}")
    print(f"COMPARISON COMPLETE — saved to {out_path}")
    print(f"{'='*60}")

    # Print comparison table
    print(f"\n{'Representation':<15} {'Vocab':>6} {'Params':>10} {'Embed%':>7} {'Val Loss':>10} {'BPC':>8}")
    print("-" * 60)
    for name, m in all_metrics.items():
        embed_pct = 100 * m["embed_params"] / m["total_params"]
        print(f"{name:<15} {m['vocab_size']:>6} {m['total_params']:>10,} {embed_pct:>6.1f}% {m['best_val_loss']:>10.4f} {m['best_bpc']:>8.4f}")

    print(f"\nKey insight: lower BPC = better, regardless of representation.")
    print(f"Embed% shows how much of the model is 'wasted' on vocabulary.")


def main():
    parser = argparse.ArgumentParser(description="Run all representation experiments")
    parser.add_argument("--data-root", default="data", help="Root directory with per-rep data")
    parser.add_argument("--out-dir", default="experiments", help="Where to save results")
    parser.add_argument("--max-iters", type=int, default=3000, help="Training iters per representation")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--eval-iters", type=int, default=20, help="Forward passes per eval")
    parser.add_argument("--eval-interval", type=int, default=200, help="Iters between evals")
    args = parser.parse_args()

    run_all(
        data_root=args.data_root,
        out_dir=args.out_dir,
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        lr=args.lr,
        seed=args.seed,
        eval_iters=args.eval_iters,
        eval_interval=args.eval_interval,
    )


if __name__ == "__main__":
    main()