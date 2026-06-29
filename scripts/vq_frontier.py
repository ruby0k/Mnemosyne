"""VQ rate-distortion frontier: BPC vs representation bit-rate.

The discrete arm of the embedding-substrate study. We product-quantize the final
hidden state at a range of bit-rates (rate = n_groups * log2(codebook_size)) and
measure validation BPC, holding the byte tokenizer and transformer body fixed.

Runs every configuration over multiple seeds and reports mean +/- std.

Outputs experiments/vq_frontier.json and experiments/plots/vq_frontier.png.
Self-contained training loop (short context), same data/byte as the harness.
"""

import sys
import json
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.vq import VQTransformer, VQConfig

DATA = Path(__file__).resolve().parent.parent / "data" / "byte"
D, N_LAYER, N_HEAD, T = 256, 3, 4, 256
BATCH, ITERS, LR = 24, 1000, 3e-4
EVAL_BATCHES = 30
SEEDS = [1337, 1338, 1339]
EVAL_SEED = 99
LN2 = math.log(2.0)

# (groups, K): rate = groups*log2(K)
CONFIGS = [
    (2, 256),   # 16 bits
    (4, 256),   # 32 bits
    (8, 64),    # 48 bits
    (8, 256),   # 64 bits
    (16, 256),  # 128 bits
    (32, 256),  # 256 bits
    (8, 16),    # 32 bits  (same rate as (4,256), different shape)
]

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


def build(vq_enabled, seed, groups=8, K=256):
    torch.manual_seed(seed)
    cfg = VQConfig(vocab_size=256, block_size=T, n_layer=N_LAYER, n_head=N_HEAD, n_embd=D,
                   grad_checkpoint=False, vq_enabled=vq_enabled, n_groups=groups, codebook_size=K)
    return VQTransformer(cfg).to(device)


def train(model, seed):
    rng = np.random.default_rng(seed)
    opt = model.configure_optimizer(lr=LR)
    model.train()
    for _ in range(ITERS):
        x, y = get_batch(train_data, rng)
        with ctx:
            _, loss = model(x, y)
        (loss + model.last_aux_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    return model


@torch.no_grad()
def evaluate(model):
    model.eval()
    rng = np.random.default_rng(EVAL_SEED)
    ces, perps = [], []
    for _ in range(EVAL_BATCHES):
        x, y = get_batch(val_data, rng)
        with ctx:
            _, loss = model(x, y)
        ces.append(loss.item())
        if model.last_code_perplexity is not None:
            perps.append(model.last_code_perplexity)
    bpc = float(np.mean(ces)) / LN2
    perp = float(np.mean(perps)) if perps else None
    return bpc, perp


def agg(vals):
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=0))


def main():
    print(f"device={device}  D={D} L={N_LAYER} T={T} batch={BATCH} iters={ITERS} seeds={SEEDS}", flush=True)

    floor_raw = []
    cfg_raw = {(g, k): {"bpc": [], "perp": []} for (g, k) in CONFIGS}

    for seed in SEEDS:
        print(f"\n=== seed {seed} ===", flush=True)
        fb, _ = evaluate(train(build(False, seed), seed))
        floor_raw.append(fb)
        print(f"  [floor] dense BPC={fb:.4f}", flush=True)
        for (g, k) in CONFIGS:
            bpc, perp = evaluate(train(build(True, seed, groups=g, K=k), seed))
            cfg_raw[(g, k)]["bpc"].append(bpc)
            if perp is not None:
                cfg_raw[(g, k)]["perp"].append(perp)
            print(f"  [vq] G={g:<2} K={k:<3} rate={g*math.log2(k):5.0f}b BPC={bpc:.4f} perp={perp:.1f}/{k}", flush=True)

    fm, fs = agg(floor_raw)
    points = []
    for (g, k) in CONFIGS:
        bm, bs = agg(cfg_raw[(g, k)]["bpc"])
        pm = float(np.mean(cfg_raw[(g, k)]["perp"])) if cfg_raw[(g, k)]["perp"] else None
        points.append({"groups": g, "codebook": k, "rate_bits": g * math.log2(k),
                       "bpc_mean": bm, "bpc_std": bs, "code_perplexity": pm, "max_perplexity": k,
                       "bpc_seeds": cfg_raw[(g, k)]["bpc"]})

    print(f"\n  dense floor: {fm:.4f}±{fs:.4f}", flush=True)

    out = {"config": {"D": D, "n_layer": N_LAYER, "T": T, "batch": BATCH, "iters": ITERS, "seeds": SEEDS},
           "dense_floor_bpc_mean": fm, "dense_floor_bpc_std": fs, "points": points}
    exp = Path(__file__).resolve().parent.parent / "experiments"
    exp.mkdir(exist_ok=True)
    (exp / "vq_frontier.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # plot: BPC vs rate with error bars (K=256 main series)
    main_series = sorted([p for p in points if p["codebook"] == 256], key=lambda p: p["rate_bits"])
    plots = exp / "plots"; plots.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(fm, color="#888", ls="--", lw=1.5, label=f"dense floor ({fm:.3f})")
    ax.fill_between([0, 260], fm - fs, fm + fs, color="#888", alpha=0.15)
    ax.errorbar([p["rate_bits"] for p in main_series], [p["bpc_mean"] for p in main_series],
                yerr=[p["bpc_std"] for p in main_series], fmt="s-", color="#1d9e75", capsize=3,
                label="VQ (K=256, vary groups)")
    for p in points:
        if p["codebook"] != 256:
            ax.errorbar(p["rate_bits"], p["bpc_mean"], yerr=p["bpc_std"], fmt="o", color="#d1495b", capsize=3)
            ax.annotate(f"G{p['groups']}×K{p['codebook']}", (p["rate_bits"], p["bpc_mean"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8, color="#d1495b")
    ax.set_xlabel("Representation rate (bits / token)")
    ax.set_ylabel("Validation BPC  (lower = better)")
    ax.set_title(f"VQ embedding substrate: rate-distortion (mean±std, {len(SEEDS)} seeds)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / "vq_frontier.png", dpi=150)
    print(f"\n✓ wrote experiments/vq_frontier.json and experiments/plots/vq_frontier.png", flush=True)


if __name__ == "__main__":
    main()
