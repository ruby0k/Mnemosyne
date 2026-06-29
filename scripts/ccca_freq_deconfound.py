"""De-confound the frequency-vs-uniform comparison.

In ccca_frontier.py, frequency models are trained *at* their allocation while
uniform is one matryoshka model *truncated* — a protocol confound. Here we apply
BOTH policies as inference-time masks on the SAME matryoshka-trained model (which
was trained on per-token random widths, so both constant-width and frequency
masks are in-distribution). This isolates the allocation policy from training.

If frequency still beats uniform at matched mean width here, the win is real.
"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")

import numpy as np, torch
from model.adaptive import AdaptiveCapacityTransformer, AdaptiveConfig

DATA = Path(__file__).resolve().parent.parent / "data" / "byte"
D, N_LAYER, N_HEAD, T = 256, 3, 4, 256
BATCH, ITERS, LR = 24, 1000, 3e-4
EVAL_BATCHES, EVAL_SEED = 30, 99
SEEDS = [1337, 1338, 1339]
TARGET_WIDTHS = [16.0, 25.0, 38.0, 60.0]
LN2 = np.log(2.0)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
ctx = torch.autocast(device_type=device, dtype=dtype) if device == "cuda" else torch.cuda.amp.nullcontext()
train_data = np.memmap(DATA / "train.bin", dtype=np.uint8, mode="r")
val_data = np.memmap(DATA / "val.bin", dtype=np.uint8, mode="r")

_counts = np.bincount(np.asarray(train_data[:2_000_000]), minlength=256).astype(np.float64)
_p = _counts / _counts.sum()
_base = np.log(_counts + 1.0); _base = (_base - _base.min()) / (_base.max() - _base.min() + 1e-9)

def freq_alpha_for_width(tw, floor_dims=2):
    floor = floor_dims / D; lo, hi = 0.0, 50.0
    for _ in range(40):
        s = (lo + hi) / 2; a = np.clip(s * _base, floor, 1.0)
        if float((_p * a).sum()) * D < tw: lo = s
        else: hi = s
    return torch.tensor(np.clip(((lo + hi) / 2) * _base, floor, 1.0), dtype=torch.float32)

def get_batch(data, rng):
    s = rng.integers(0, len(data) - T - 1, size=BATCH); o = s[:, None] + np.arange(T + 1)[None, :]
    seq = torch.from_numpy(data[o].astype(np.int64)).to(device)
    return seq[:, :-1].contiguous(), seq[:, 1:].contiguous()

def train_base(seed):
    torch.manual_seed(seed)
    m = AdaptiveCapacityTransformer(AdaptiveConfig(vocab_size=256, block_size=T, n_layer=N_LAYER,
        n_head=N_HEAD, n_embd=D, matryoshka_prob=1.0)).to(device)
    opt = m.configure_optimizer(lr=LR); rng = np.random.default_rng(seed); m.train()
    for _ in range(ITERS):
        x, y = get_batch(train_data, rng)
        with ctx: _, loss = m(x, y)
        (loss + m.last_aux_loss).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
    return m

@torch.no_grad()
def ev(m, uniform_width=None, freq=False):
    m.eval(); rng = np.random.default_rng(EVAL_SEED)
    m.config.use_freq_width = freq; m.config.uniform_width = None if freq else uniform_width
    ces, ws = [], []
    for _ in range(EVAL_BATCHES):
        x, y = get_batch(val_data, rng)
        with ctx: _, loss = m(x, y)
        ces.append(loss.item()); ws.append(m.last_avg_width)
    m.config.use_freq_width = False; m.config.uniform_width = None
    return float(np.mean(ces)) / LN2, float(np.mean(ws))

def main():
    print(f"device={device} de-confound: same model, two policies, seeds={SEEDS}", flush=True)
    uni = {tw: [] for tw in TARGET_WIDTHS}; frq = {tw: [] for tw in TARGET_WIDTHS}
    for seed in SEEDS:
        m = train_base(seed)
        for tw in TARGET_WIDTHS:
            ub, uw = ev(m, uniform_width=tw / D)
            uni[tw].append(ub)
            m.freq_alpha.copy_(freq_alpha_for_width(tw).to(device))
            fb, fw = ev(m, freq=True)
            frq[tw].append(fb)
            print(f"  seed {seed} w~{tw:4.0f}: uniform {ub:.4f} (w={uw:.1f}) | freq {fb:.4f} (w={fw:.1f})", flush=True)
    print("\n  matched-width (same model; mean over seeds):", flush=True)
    print(f"    {'width':>6}{'uniform':>10}{'freq':>10}{'Δ(freq-uni)':>13}", flush=True)
    out = []
    for tw in TARGET_WIDTHS:
        u, f = float(np.mean(uni[tw])), float(np.mean(frq[tw]))
        out.append({"target_width": tw, "uniform": u, "freq": f, "delta": f - u})
        verdict = "FREQ WINS" if f < u else "uniform wins"
        print(f"    {tw:6.0f}{u:10.4f}{f:10.4f}{f-u:+13.4f}  [{verdict}]", flush=True)
    exp = Path(__file__).resolve().parent.parent / "experiments"
    (exp / "ccca_freq_deconfound.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n✓ wrote experiments/ccca_freq_deconfound.json", flush=True)

if __name__ == "__main__":
    main()
