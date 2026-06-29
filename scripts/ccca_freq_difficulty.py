"""Does token frequency track difficulty? (reviewer-requested mechanism check)

The paper's explanation is: frequent bytes are predictable, so width pays off on
them, so a frequency prior allocates well. This script tests the missing link:
  - per-byte log-frequency (from data/byte/train.bin)
  - per-byte intrinsic difficulty = mean per-token CE of a full-width reference
    model, aggregated by the target byte
  - corr(log-frequency, difficulty)  [+ scatter]
and, for a trained CCCA gate, the mean allocated width per difficulty quartile.

Outputs experiments/ccca_freq_difficulty.json and plots/freq_vs_difficulty.png.
"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")

import numpy as np, torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model.adaptive import AdaptiveCapacityTransformer, AdaptiveConfig

DATA = Path(__file__).resolve().parent.parent / "data" / "byte"
D, N_LAYER, N_HEAD, T = 256, 3, 4, 256
BATCH, ITERS, LR, SEED = 24, 1000, 3e-4, 1337
EVAL_BATCHES, EVAL_SEED = 40, 99
LN2 = np.log(2.0)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
ctx = torch.autocast(device_type=device, dtype=dtype) if device == "cuda" else torch.cuda.amp.nullcontext()
train_data = np.memmap(DATA / "train.bin", dtype=np.uint8, mode="r")
val_data = np.memmap(DATA / "val.bin", dtype=np.uint8, mode="r")

def get_batch(data, rng):
    s = rng.integers(0, len(data) - T - 1, size=BATCH); o = s[:, None] + np.arange(T + 1)[None, :]
    seq = torch.from_numpy(data[o].astype(np.int64)).to(device)
    return seq[:, :-1].contiguous(), seq[:, 1:].contiguous()

def build(matryoshka_prob=0.5, force_full=False, capacity_lambda=0.0):
    torch.manual_seed(SEED)
    return AdaptiveCapacityTransformer(AdaptiveConfig(vocab_size=256, block_size=T, n_layer=N_LAYER,
        n_head=N_HEAD, n_embd=D, matryoshka_prob=matryoshka_prob, force_full=force_full,
        capacity_lambda=capacity_lambda)).to(device)

def train(m):
    rng = np.random.default_rng(SEED); opt = m.configure_optimizer(lr=LR); m.train()
    for _ in range(ITERS):
        x, y = get_batch(train_data, rng)
        with ctx: _, loss = m(x, y)
        (loss + m.last_aux_loss).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
    return m

def main():
    print(f"device={device}", flush=True)
    ref = train(build(force_full=True, matryoshka_prob=0.0))   # full-width reference
    gate = train(build(matryoshka_prob=0.5, capacity_lambda=0.5))  # trained CCCA gate

    rng = np.random.default_rng(EVAL_SEED)
    batches = [get_batch(val_data, rng) for _ in range(EVAL_BATCHES)]

    # per-token CE (difficulty) from reference, plus per-token gate width
    ce_all, y_all, alpha_all = [], [], []
    ref.eval(); gate.eval()
    with torch.no_grad():
        for x, y in batches:
            with ctx:
                logits, _ = ref(x, y)
                gate(x, y)
            ce = F.cross_entropy(logits.reshape(-1, 256).float(), y.reshape(-1), reduction="none")
            ce_all.append(ce.cpu().numpy()); y_all.append(y.reshape(-1).cpu().numpy())
            alpha_all.append(gate.last_alpha.reshape(-1).float().cpu().numpy())
    ce = np.concatenate(ce_all); ytok = np.concatenate(y_all); alpha = np.concatenate(alpha_all)

    # per-byte difficulty (mean CE for predicting that byte) and frequency
    counts = np.bincount(np.asarray(train_data[:2_000_000]), minlength=256).astype(np.float64)
    logf = np.log(counts + 1.0)
    diff_byte, present = np.zeros(256), counts > 0
    for b in range(256):
        m = ytok == b
        diff_byte[b] = ce[m].mean() / LN2 if m.any() else np.nan
    ok = present & np.isfinite(diff_byte)
    # correlation over byte types, and frequency-weighted
    r_types = float(np.corrcoef(logf[ok], diff_byte[ok])[0, 1])
    w = counts[ok] / counts[ok].sum()
    lf, df = logf[ok], diff_byte[ok]
    r_weighted = float((w * (lf - (w*lf).sum()) * (df - (w*df).sum())).sum() /
                       (np.sqrt((w*(lf-(w*lf).sum())**2).sum()) * np.sqrt((w*(df-(w*df).sum())**2).sum())))
    print(f"  corr(log-freq, difficulty): types r={r_types:.3f}  freq-weighted r={r_weighted:.3f}", flush=True)

    # gate width by difficulty quartile (per-token)
    q = np.quantile(ce / LN2, [0.25, 0.5, 0.75])
    dtok = ce / LN2
    buckets = {"Q1 easy": dtok <= q[0], "Q2": (dtok > q[0]) & (dtok <= q[1]),
               "Q3": (dtok > q[1]) & (dtok <= q[2]), "Q4 hard": dtok > q[2]}
    width_by_q = {n: float(alpha[mk].mean() * D) for n, mk in buckets.items()}
    print("  gate width by difficulty quartile:", {k: round(v,1) for k,v in width_by_q.items()}, flush=True)

    out = {"corr_logfreq_difficulty_types": r_types, "corr_logfreq_difficulty_weighted": r_weighted,
           "gate_width_by_quartile": width_by_q}
    exp = Path(__file__).resolve().parent.parent / "experiments"; exp.mkdir(exist_ok=True)
    (exp / "ccca_freq_difficulty.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    plots = exp / "plots"; plots.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(logf[ok], diff_byte[ok], s=8, alpha=0.5, color="#185fa5")
    ax.set_xlabel("log frequency (byte)"); ax.set_ylabel("difficulty (mean BPC to predict byte)")
    ax.set_title(f"Frequent bytes are easier (r={r_types:.2f}, freq-weighted {r_weighted:.2f})")
    ax.grid(True, alpha=0.3); fig.tight_layout(); fig.savefig(plots / "freq_vs_difficulty.png", dpi=150)
    print("\n✓ wrote experiments/ccca_freq_difficulty.json and plots/freq_vs_difficulty.png", flush=True)

if __name__ == "__main__":
    main()
