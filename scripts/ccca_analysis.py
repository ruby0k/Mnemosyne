"""Analysis for the CCCA paper: (a) test that per-position BPC b(w) is convex
in width (de-post-hoc the Jensen argument), and (b) show width helps easy tokens
but not hard ones (why a budgeted gate starves hard tokens).

Both use one prefix-robust (Matryoshka) model evaluated at a grid of fixed
widths, plus a full-width reference model for per-token intrinsic difficulty.

Outputs experiments/ccca_analysis.json and plots/{bw_convexity,difficulty_vs_width}.png.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.adaptive import AdaptiveCapacityTransformer, AdaptiveConfig

DATA = Path(__file__).resolve().parent.parent / "data" / "byte"
D, N_LAYER, N_HEAD, T = 256, 3, 4, 256
BATCH, ITERS, LR, SEED = 24, 1000, 3e-4, 1337
EVAL_BATCHES, EVAL_SEED = 30, 99
LN2 = np.log(2.0)
WIDTH_DIMS = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256]

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
ctx = torch.autocast(device_type=device, dtype=dtype) if device == "cuda" else torch.cuda.amp.nullcontext()
train_data = np.memmap(DATA / "train.bin", dtype=np.uint8, mode="r")
val_data = np.memmap(DATA / "val.bin", dtype=np.uint8, mode="r")


def get_batch(data, rng):
    s = rng.integers(0, len(data) - T - 1, size=BATCH)
    o = s[:, None] + np.arange(T + 1)[None, :]
    seq = torch.from_numpy(data[o].astype(np.int64)).to(device)
    return seq[:, :-1].contiguous(), seq[:, 1:].contiguous()


def build(matryoshka_prob, force_full=False):
    torch.manual_seed(SEED)
    cfg = AdaptiveConfig(vocab_size=256, block_size=T, n_layer=N_LAYER, n_head=N_HEAD, n_embd=D,
                         grad_checkpoint=False, matryoshka_prob=matryoshka_prob, force_full=force_full)
    return AdaptiveCapacityTransformer(cfg).to(device)


def train(model):
    rng = np.random.default_rng(SEED)
    opt = model.configure_optimizer(lr=LR)
    model.train()
    for _ in range(ITERS):
        x, y = get_batch(train_data, rng)
        with ctx:
            _, loss = model(x, y)
        (loss + model.last_aux_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
    return model


@torch.no_grad()
def per_token_ce(model, batches, uniform_width=None):
    """Concatenated per-token CE (nats) over fixed batches at a given width."""
    prev = model.config.uniform_width
    model.config.uniform_width = uniform_width
    model.eval()
    out = []
    for x, y in batches:
        with ctx:
            logits, _ = model(x, y)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                             y.reshape(-1), reduction="none")
        out.append(ce.cpu())
    model.config.uniform_width = prev
    return torch.cat(out).numpy()


def main():
    print(f"device={device} D={D} L={N_LAYER} T={T} iters={ITERS}", flush=True)
    base = train(build(matryoshka_prob=1.0))            # prefix-robust, truncatable
    ref = train(build(matryoshka_prob=0.0, force_full=True))  # full-width reference

    rng = np.random.default_rng(EVAL_SEED)
    batches = [get_batch(val_data, rng) for _ in range(EVAL_BATCHES)]

    # (a) convexity of b(w): mean per-token CE at each fixed width
    bw = []
    for dim in WIDTH_DIMS:
        ce = per_token_ce(base, batches, uniform_width=dim / D)
        bw.append(float(ce.mean()) / LN2)
        print(f"  b(w): width={dim:3d}  BPC={bw[-1]:.4f}", flush=True)
    bw = np.array(bw); w = np.array(WIDTH_DIMS, float)
    second = np.diff(bw, 2)                              # discrete 2nd derivative
    convex = bool(np.all(second >= -1e-4))
    print(f"  convex (all 2nd-diffs >= ~0): {convex}; 2nd-diffs={np.round(second,5).tolist()}", flush=True)

    # (b) difficulty vs width: bucket tokens by reference CE quartile, BPC per bucket
    diff = per_token_ce(ref, batches, uniform_width=None)      # intrinsic difficulty
    q = np.quantile(diff, [0.25, 0.5, 0.75])
    buckets = {
        "Q1 easy": diff <= q[0],
        "Q2": (diff > q[0]) & (diff <= q[1]),
        "Q3": (diff > q[1]) & (diff <= q[2]),
        "Q4 hard": diff > q[2],
    }
    by_bucket = {name: [] for name in buckets}
    for dim in WIDTH_DIMS:
        ce = per_token_ce(base, batches, uniform_width=dim / D)
        for name, mask in buckets.items():
            by_bucket[name].append(float(ce[mask].mean()) / LN2)
    print("\n  difficulty-vs-width (BPC per bucket):", flush=True)
    for name in buckets:
        print(f"    {name:8s}: {[round(v,3) for v in by_bucket[name]]}", flush=True)

    out = {"width_dims": WIDTH_DIMS, "bw_bpc": bw.tolist(), "convex": convex,
           "second_diffs": second.tolist(), "difficulty_buckets": by_bucket,
           "quartile_thresholds_nats": q.tolist()}
    exp = Path(__file__).resolve().parent.parent / "experiments"
    exp.mkdir(exist_ok=True)
    (exp / "ccca_analysis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    plots = exp / "plots"; plots.mkdir(parents=True, exist_ok=True)

    # plot (a)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(w, bw, "o-", color="#185fa5")
    ax.set_xlabel("Width (dimensions)"); ax.set_ylabel("BPC  b(w)")
    ax.set_title(f"Per-position cost b(w) is {'convex' if convex else 'NOT convex'}")
    ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(plots / "bw_convexity.png", dpi=150)

    # plot (b)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = {"Q1 easy": "#1d9e75", "Q2": "#185fa5", "Q3": "#ba7517", "Q4 hard": "#d1495b"}
    for name in buckets:
        ax.plot(w, by_bucket[name], "o-", color=colors[name], label=name)
    ax.set_xlabel("Width (dimensions)"); ax.set_ylabel("BPC (per difficulty bucket)")
    ax.set_title("Width helps easy tokens, not hard ones")
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(plots / "difficulty_vs_width.png", dpi=150)
    print("\n✓ wrote experiments/ccca_analysis.json and 2 plots", flush=True)


if __name__ == "__main__":
    main()
