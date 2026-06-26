"""Plot experiment results — training curves, BPC vs vocab, embed% vs BPC.

Generates static PNG plots from saved metrics JSON files.
Does NOT require a running training process.

Usage:
    uv run python scripts/plot_results.py
    uv run python scripts/plot_results.py --exp-dir experiments --out-dir experiments/plots
    uv run python scripts/plot_results.py --exp-dir experiments --out-dir experiments/plots --dpi 150
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def load_metrics(exp_dir: Path) -> dict[str, dict]:
    """Load all *_metrics.json files from an experiment directory."""
    metrics = {}
    for f in sorted(exp_dir.glob("*_metrics.json")):
        name = f.stem.replace("_metrics", "")
        try:
            metrics[name] = json.loads(f.read_text())
        except json.JSONDecodeError:
            pass
    return metrics


def plot_training_curves(metrics: dict, out_dir: Path, dpi: int = 150):
    """Plot training and validation loss curves for all representations."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Training loss
    ax = axes[0]
    for name, m in metrics.items():
        iters = [r["iter"] for r in m.get("iters", [])]
        val_losses = [r["val_loss"] for r in m.get("iters", [])]
        if iters:
            ax.plot(iters, val_losses, "-", label=name, linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Validation Loss (nats)")
    ax.set_title("Validation Loss Curves")
    ax.legend(fontsize=8, ncol=2)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # BPC curves
    ax = axes[1]
    for name, m in metrics.items():
        iters = [r["iter"] for r in m.get("iters", [])]
        bpcs = [r["bpc"] for r in m.get("iters", [])]
        if iters:
            ax.plot(iters, bpcs, "-", label=name, linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Bits Per Character (BPC)")
    ax.set_title("BPC Curves (lower = better)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = out_dir / "training_curves.png"
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def plot_bpc_vs_vocab(metrics: dict, out_dir: Path, dpi: int = 150):
    """Scatter plot: BPC vs vocab size, colored by representation type."""
    fig, ax = plt.subplots(figsize=(10, 7))

    names = []
    vocabs = []
    bpcs = []
    embed_pcts = []
    total_params = []

    for name, m in metrics.items():
        names.append(name)
        vocabs.append(m["vocab_size"])
        bpcs.append(m["best_bpc"])
        embed_pcts.append(100 * m["embed_params"] / m["total_params"])
        total_params.append(m["total_params"])

    # Color by embed%
    scatter = ax.scatter(vocabs, bpcs, c=embed_pcts, s=[p / 5000 for p in total_params],
                         cmap="RdYlGn_r", edgecolors="black", linewidth=0.5, zorder=5)
    plt.colorbar(scatter, label="Embedding % of params")

    # Annotate each point
    for i, name in enumerate(names):
        ax.annotate(name, (vocabs[i], bpcs[i]), fontsize=8,
                    xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Vocabulary Size")
    ax.set_ylabel("Bits Per Character (BPC)")
    ax.set_title("BPC vs Vocabulary Size (color = embed%, size = total params)")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = out_dir / "bpc_vs_vocab.png"
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def plot_embed_vs_bpc(metrics: dict, out_dir: Path, dpi: int = 150):
    """Plot: embedding % of params vs BPC. Shows the tradeoff."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for name, m in metrics.items():
        embed_pct = 100 * m["embed_params"] / m["total_params"]
        bpc = m["best_bpc"]
        ax.scatter(embed_pct, bpc, s=100, zorder=5)
        ax.annotate(name, (embed_pct, bpc), fontsize=9,
                    xytext=(8, 5), textcoords="offset points")

    # Trend line
    embed_pcts = [100 * m["embed_params"] / m["total_params"] for m in metrics.values()]
    bpcs = [m["best_bpc"] for m in metrics.values()]
    if len(embed_pcts) > 2:
        z = np.polyfit(embed_pcts, bpcs, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(embed_pcts), max(embed_pcts), 100)
        ax.plot(x_line, p(x_line), "--", color="gray", alpha=0.5, label=f"trend: y={z[0]:.3f}x+{z[1]:.2f}")
        ax.legend()

    ax.set_xlabel("Embedding % of Total Parameters")
    ax.set_ylabel("Bits Per Character (BPC)")
    ax.set_title("Parameter Waste vs Model Quality\n(more embedding % = worse BPC)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = out_dir / "embed_vs_bpc.png"
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def print_comparison_table(metrics: dict):
    """Print a formatted comparison table."""
    ranked = sorted(metrics.items(), key=lambda x: x[1]["best_bpc"])

    print(f"\n{'Rep':<15} {'Vocab':>6} {'Params':>10} {'Embed%':>7} {'CPT':>5} {'Iters':>6} {'BPC':>8}")
    print("-" * 65)
    for name, m in ranked:
        ep = 100 * m["embed_params"] / m["total_params"]
        cpt = m.get("chars_per_token", 1.0)
        iters = m.get("max_iters", "?")
        print(f"{name:<15} {m['vocab_size']:>6} {m['total_params']:>10,} {ep:>6.1f}% {cpt:>5.1f} {iters:>6} {m['best_bpc']:>8.4f}")


def main():
    parser = argparse.ArgumentParser(description="Plot experiment results")
    parser.add_argument("--exp-dir", default="experiments", help="Directory with metrics JSONs")
    parser.add_argument("--out-dir", default="experiments/plots", help="Where to save plots")
    parser.add_argument("--dpi", type=int, default=150, help="Plot resolution")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(exp_dir)
    if not metrics:
        print(f"No metrics found in {exp_dir}")
        sys.exit(1)

    print(f"Loaded {len(metrics)} representations: {', '.join(sorted(metrics.keys()))}\n")

    print_comparison_table(metrics)

    print(f"\nGenerating plots in {out_dir}...")
    plot_training_curves(metrics, out_dir, args.dpi)
    plot_bpc_vs_vocab(metrics, out_dir, args.dpi)
    plot_embed_vs_bpc(metrics, out_dir, args.dpi)

    print(f"\n✓ All plots saved to {out_dir}")


if __name__ == "__main__":
    main()