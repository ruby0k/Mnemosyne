"""CCCA frontier sweep: three allocation policies at matched average width.

Compares how to spend a fixed average representation-width budget on a byte-level
LM, holding tokenizer and transformer body fixed:
  - Uniform (Matryoshka): every token gets the same width (one prefix-robust model,
    evaluated at fixed truncation widths).
  - Adaptive-frequency: per-token width set by the input byte's corpus frequency
    (Adaptive-Input / Adaptive-Softmax style), context-free; trained at allocation.
  - CCCA (learned): a gate picks per-token width from context; trained under a
    budget lambda.

Prediction (convexity / Jensen): uniform allocation is optimal at a fixed mean
width, so both non-uniform policies (frequency, learned) should lose to uniform.

Runs every configuration over multiple seeds, reports mean +/- std with error
bars. `--big` runs a reduced protocol at a larger config to confirm the ordering
holds at scale.

Outputs experiments/ccca_frontier.json and experiments/plots/ccca_frontier.png.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.adaptive import AdaptiveCapacityTransformer, AdaptiveConfig

BIG = "--big" in sys.argv

DATA = Path(__file__).resolve().parent.parent / "data" / "byte"
if BIG:
    D, N_LAYER, N_HEAD, T = 384, 6, 6, 512
    BATCH, ITERS, LR = 16, 5000, 3e-4
    SEEDS = [1337, 1338]
    LAMBDAS = [0.3]                       # one CCCA point
    UNIFORM_WIDTHS = [0.10, 1.0]          # ~38 dims + full
    FREQ_TARGET_WIDTHS = [0.10 * D]       # ~one matched freq point
else:
    D, N_LAYER, N_HEAD, T = 256, 3, 4, 256
    BATCH, ITERS, LR = 24, 1000, 3e-4
    SEEDS = [1337, 1338, 1339]
    LAMBDAS = [0.1, 0.3, 0.5, 1.0, 2.0]
    UNIFORM_WIDTHS = [0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0]
    FREQ_TARGET_WIDTHS = [60.0, 38.0, 25.0, 16.0]

EVAL_BATCHES = 30
EVAL_SEED = 99
GRAD_CKPT = BIG          # 512-context big runs need checkpointing on 8GB
LN2 = np.log(2.0)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
ctx = torch.autocast(device_type=device, dtype=dtype) if device == "cuda" else torch.cuda.amp.nullcontext()

train_data = np.memmap(DATA / "train.bin", dtype=np.uint8, mode="r")
val_data = np.memmap(DATA / "val.bin", dtype=np.uint8, mode="r")

# ── byte-frequency shape for the adaptive-frequency baseline ──
_counts = np.bincount(np.asarray(train_data[: 2_000_000]), minlength=256).astype(np.float64)
_p = _counts / _counts.sum()                                  # token distribution
_base = np.log(_counts + 1.0)
_base = (_base - _base.min()) / (_base.max() - _base.min() + 1e-9)   # [0,1], frequent→1


def freq_alpha_for_width(target_width, floor_dims=2):
    """Scale the log-freq shape so the token-weighted mean active width hits target."""
    floor = floor_dims / D
    lo, hi = 0.0, 50.0
    for _ in range(40):
        s = (lo + hi) / 2
        a = np.clip(s * _base, floor, 1.0)
        if float((_p * a).sum()) * D < target_width:
            lo = s
        else:
            hi = s
    a = np.clip(((lo + hi) / 2) * _base, floor, 1.0)
    return torch.tensor(a, dtype=torch.float32)


def get_batch(data, rng):
    s = rng.integers(0, len(data) - T - 1, size=BATCH)
    o = s[:, None] + np.arange(T + 1)[None, :]
    seq = torch.from_numpy(data[o].astype(np.int64)).to(device)
    return seq[:, :-1].contiguous(), seq[:, 1:].contiguous()


def build(seed, capacity_lambda=0.0, matryoshka_prob=0.5, use_freq_width=False, freq_alpha=None):
    torch.manual_seed(seed)
    cfg = AdaptiveConfig(vocab_size=256, block_size=T, n_layer=N_LAYER, n_head=N_HEAD,
                         n_embd=D, grad_checkpoint=GRAD_CKPT,
                         capacity_lambda=capacity_lambda, matryoshka_prob=matryoshka_prob,
                         use_freq_width=use_freq_width)
    m = AdaptiveCapacityTransformer(cfg).to(device)
    if freq_alpha is not None:
        m.freq_alpha.copy_(freq_alpha.to(device))
    return m


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
def evaluate(model, uniform_width=None):
    prev = model.config.uniform_width
    model.config.uniform_width = uniform_width
    model.eval()
    rng = np.random.default_rng(EVAL_SEED)
    ces, widths = [], []
    for _ in range(EVAL_BATCHES):
        x, y = get_batch(val_data, rng)
        with ctx:
            _, loss = model(x, y)
        ces.append(loss.item())
        widths.append(model.last_avg_width)
    model.config.uniform_width = prev
    return float(np.mean(ces)) / LN2, float(np.mean(widths))


def agg(v):
    a = np.array(v, float)
    return float(a.mean()), float(a.std(ddof=0))


def main():
    print(f"device={device} BIG={BIG} D={D} L={N_LAYER} T={T} batch={BATCH} iters={ITERS} seeds={SEEDS}", flush=True)

    ours_raw = {lam: {"bpc": [], "w": []} for lam in LAMBDAS}
    uni_raw = {f: {"bpc": [], "w": []} for f in UNIFORM_WIDTHS}
    freq_raw = {tw: {"bpc": [], "w": []} for tw in FREQ_TARGET_WIDTHS}

    for seed in SEEDS:
        print(f"\n=== seed {seed} ===", flush=True)
        for lam in LAMBDAS:
            b, w = evaluate(train(build(seed, capacity_lambda=lam, matryoshka_prob=0.5), seed))
            ours_raw[lam]["bpc"].append(b); ours_raw[lam]["w"].append(w)
            print(f"  [ours]  λ={lam:<4} w={w:6.1f} BPC={b:.4f}", flush=True)
        base = train(build(seed, capacity_lambda=0.0, matryoshka_prob=1.0), seed)
        for f in UNIFORM_WIDTHS:
            b, w = evaluate(base, uniform_width=f)
            uni_raw[f]["bpc"].append(b); uni_raw[f]["w"].append(w)
        for tw in FREQ_TARGET_WIDTHS:
            fa = freq_alpha_for_width(tw)
            m = train(build(seed, use_freq_width=True, matryoshka_prob=0.0, freq_alpha=fa), seed)
            b, w = evaluate(m)
            freq_raw[tw]["bpc"].append(b); freq_raw[tw]["w"].append(w)
            print(f"  [freq]  tw={tw:<5} w={w:6.1f} BPC={b:.4f}", flush=True)

    def pack(raw, key):
        out = []
        for k, v in raw.items():
            bm, bs = agg(v["bpc"]); wm, ws = agg(v["w"])
            out.append({key: k, "bpc_mean": bm, "bpc_std": bs, "width_mean": wm,
                        "width_std": ws, "bpc_seeds": v["bpc"]})
        return out

    ours = pack(ours_raw, "lambda")
    uniform = pack(uni_raw, "frac")
    freq = pack(freq_raw, "target_width")

    # three-way ranking at matched width: interpolate each arm's mean curve
    def curve(arm):
        xs = np.array([p["width_mean"] for p in arm]); ys = np.array([p["bpc_mean"] for p in arm])
        o = np.argsort(xs); return xs[o], ys[o]
    ux, uy = curve(uniform); fx, fy = curve(freq); cx, cy = curve(ours)
    grid = sorted({round(p["width_mean"]) for p in ours + freq})
    print("\n  three-way ranking at matched width (mean BPC; lower=better):", flush=True)
    print(f"    {'width':>6} {'uniform':>9} {'freq':>9} {'CCCA':>9}", flush=True)
    ranking = []
    for g in grid:
        u = float(np.interp(g, ux, uy)) if ux.min() <= g <= ux.max() else None
        f = float(np.interp(g, fx, fy)) if fx.min() <= g <= fx.max() else None
        c = float(np.interp(g, cx, cy)) if cx.min() <= g <= cx.max() else None
        ranking.append({"width": g, "uniform": u, "freq": f, "ccca": c})
        fs = lambda v: f"{v:9.4f}" if v is not None else f"{'--':>9}"
        print(f"    {g:6d} {fs(u)} {fs(f)} {fs(c)}", flush=True)

    out = {"config": {"D": D, "n_layer": N_LAYER, "T": T, "batch": BATCH, "iters": ITERS,
                      "lr": LR, "seeds": SEEDS, "big": BIG},
           "ours": ours, "uniform": uniform, "freq": freq, "ranking": ranking}
    exp = Path(__file__).resolve().parent.parent / "experiments"
    exp.mkdir(exist_ok=True)
    name = "ccca_frontier_big.json" if BIG else "ccca_frontier.json"
    (exp / name).write_text(json.dumps(out, indent=2), encoding="utf-8")

    # plot (three arms with error bars)
    plots = exp / "plots"; plots.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    uo = sorted(uniform, key=lambda p: p["width_mean"])
    ax.errorbar([p["width_mean"] for p in uo], [p["bpc_mean"] for p in uo],
                yerr=[p["bpc_std"] for p in uo], fmt="o-", color="#888", capsize=3,
                label="Uniform (Matryoshka)")
    fo = sorted(freq, key=lambda p: p["width_mean"])
    ax.errorbar([p["width_mean"] for p in fo], [p["bpc_mean"] for p in fo],
                yerr=[p["bpc_std"] for p in fo], fmt="^-", color="#1d9e75", capsize=3,
                label="Adaptive-frequency")
    ax.errorbar([p["width_mean"] for p in ours], [p["bpc_mean"] for p in ours],
                yerr=[p["bpc_std"] for p in ours], fmt="s-", color="#d1495b", capsize=3,
                label="Learned (CCCA)")
    ax.set_xlabel("Average active width (dimensions)")
    ax.set_ylabel("Validation BPC  (lower = better)")
    ax.set_title(f"Allocation policies at matched width ({'big' if BIG else 'mean±std, %d seeds' % len(SEEDS)})")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    pname = "ccca_frontier_big.png" if BIG else "ccca_frontier.png"
    fig.savefig(plots / pname, dpi=150)
    print(f"\n✓ wrote experiments/{name} and plots/{pname}", flush=True)


if __name__ == "__main__":
    main()
