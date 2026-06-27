"""Run all representation experiments back-to-back and compare results.

Each experiment trains a small model on the same data with the same compute budget.
The only variable is the data representation.

Improvements (v2):
  - Fair comparison: equalize chars-seen per representation
  - Data scaling: support different dataset sizes
  - Architecture sweep: test multiple architectures per representation
  - Better comparison table with chars-seen column
  - Per-rep LR defaults
  - Text generation samples
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Force UTF-8 console output: Windows consoles default to a locale codec
# (e.g. cp1250) that can't encode the → / ✓ / ⚠ characters in progress messages.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from train import train_one_representation, DEFAULT_LR


REPS = ["byte", "char", "bpe", "patch",
        "bpe1000", "bpe4000", "bpe8000",
        "word", "bpe_dropout", "ngram2", "ngram3",
        "mamba"]


def run_all(data_root: str, out_dir: str, max_iters: int, batch_size: int,
            n_layer: int, n_head: int, n_embd: int, lr: float, seed: int,
            eval_iters: int = 20, eval_interval: int = 500,
            fair_iters: bool = True, target_chars: int = 0,
            n_kv_head: int | None = None, embed_lr_scale: float = 1.0,
            reps: list[str] | None = None, grad_checkpoint: bool = False):
    all_metrics = {}
    rep_list = reps or REPS

    for rep_name in rep_list:
        data_dir = str(Path(data_root) / rep_name)
        if not (Path(data_dir) / "train.bin").exists():
            print(f"⚠ Skipping {rep_name} — data not found at {data_dir}", flush=True)
            continue

        # Use per-rep LR default if lr is None or 0
        rep_lr = lr if lr and lr > 0 else DEFAULT_LR.get(rep_name, 3e-4)

        metrics = train_one_representation(
            rep_name=rep_name,
            data_dir=data_dir,
            out_dir=out_dir,
            max_iters=max_iters,
            batch_size=batch_size,
            eval_iters=eval_iters,
            eval_interval=eval_interval,
            learning_rate=rep_lr,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            n_kv_head=n_kv_head,
            grad_checkpoint=grad_checkpoint,
            seed=seed,
            fair_iters=fair_iters,
            target_chars=target_chars,
            embed_lr_scale=embed_lr_scale,
            generate_after=True,
        )
        all_metrics[rep_name] = metrics

    # Save combined results
    out_path = Path(out_dir) / "comparison.json"
    out_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"\n{'='*60}", flush=True)
    print(f"COMPARISON COMPLETE — saved to {out_path}", flush=True)
    print(f"{'='*60}", flush=True)

    # Print comparison table
    print(f"\n{'Representation':<15} {'Vocab':>6} {'Params':>10} {'Embed%':>7} {'CPT':>5} {'Iters':>6} {'Val Loss':>10} {'BPC':>8}", flush=True)
    print("-" * 75, flush=True)
    for name, m in all_metrics.items():
        embed_pct = 100 * m["embed_params"] / m["total_params"]
        cpt = m.get("chars_per_token", 1.0)
        iters = m.get("max_iters", "?")
        print(f"{name:<15} {m['vocab_size']:>6} {m['total_params']:>10,} {embed_pct:>6.1f}% {cpt:>5.1f} {iters:>6} {m['best_val_loss']:>10.4f} {m['best_bpc']:>8.4f}", flush=True)

    print(f"\nKey insight: lower BPC = better, regardless of representation.", flush=True)
    print(f"Embed% shows how much of the model is 'wasted' on vocabulary.", flush=True)
    print(f"CPT = chars per token (higher = more text compression).", flush=True)
    print(f"Iters adjusted for fair comparison (same total chars seen).", flush=True)

    # Print generation samples
    print(f"\n{'='*60}", flush=True)
    print(f"TEXT SAMPLES", flush=True)
    print(f"{'='*60}", flush=True)
    for name, m in all_metrics.items():
        samples = m.get("samples", [])
        if samples:
            print(f"\n--- {name} ---", flush=True)
            for s in samples[:2]:
                preview = s["text"][:300].replace("\n", " ")
                print(f"  [{s['prompt'] or '<empty>'}] → {preview}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run all representation experiments")
    parser.add_argument("--data-root", default="data", help="Root directory with per-rep data")
    parser.add_argument("--out-dir", default="experiments", help="Where to save results")
    parser.add_argument("--max-iters", type=int, default=5000, help="Training iters per representation")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-kv-head", type=int, default=None, help="GQA KV heads (None = n_head)")
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0, help="Learning rate (0 = per-rep default)")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--eval-iters", type=int, default=20, help="Forward passes per eval")
    parser.add_argument("--eval-interval", type=int, default=500, help="Iters between evals")
    parser.add_argument("--no-fair-iters", action="store_true", help="Disable fair iteration adjustment")
    parser.add_argument("--target-chars", type=int, default=0, help="Override target chars for fair comparison")
    parser.add_argument("--embed-lr-scale", type=float, default=1.0, help="Scale embedding LR")
    parser.add_argument("--reps", type=str, default=None, help="Comma-separated list of reps to run (default: all)")
    parser.add_argument("--grad-checkpoint", action="store_true", help="Force gradient checkpointing")
    args = parser.parse_args()

    reps = args.reps.split(",") if args.reps else None

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
        fair_iters=not args.no_fair_iters,
        target_chars=args.target_chars,
        n_kv_head=args.n_kv_head,
        embed_lr_scale=args.embed_lr_scale,
        reps=reps,
        grad_checkpoint=args.grad_checkpoint,
    )


if __name__ == "__main__":
    main()