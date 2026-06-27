"""Wrapper to run all experiments with explicit file flushing."""
import sys
import os
from pathlib import Path

# Force unbuffered + UTF-8 (Windows consoles default to a locale codec like
# cp1250 that can't encode the → / ✓ / ⚠ characters in progress messages).
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1, encoding='utf-8')  # line buffered
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 1, encoding='utf-8')

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train import train_one_representation

REPS = ["byte", "char", "bpe", "patch"]
data_root = "data"
out_dir = "experiments"

for rep_name in REPS:
    data_dir = str(Path(data_root) / rep_name)
    if not (Path(data_dir) / "train.bin").exists():
        print(f"SKIP {rep_name} — data not found at {data_dir}", flush=True)
        continue

    metrics = train_one_representation(
        rep_name=rep_name,
        data_dir=data_dir,
        out_dir=out_dir,
        max_iters=3000,
        batch_size=32,
        n_layer=6,
        n_head=6,
        n_embd=384,
        learning_rate=3e-4,
        seed=1337,
    )
    print(f"DONE {rep_name}: val_loss={metrics['best_val_loss']:.4f}, BPC={metrics['best_bpc']:.4f}", flush=True)

print("ALL_DONE", flush=True)