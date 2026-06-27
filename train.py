"""Training loop — shared across all representations.

The same training code runs for every representation. Only the data loader
and model config change. This ensures fair comparison.

Improvements (v2):
  - Fair comparison: equalize chars-seen per representation by adjusting iters
  - Text generation samples after training
  - Per-representation learning rate support
  - GQA support via n_kv_head
  - Gradient checkpointing for memory savings (byte/char reps)
  - Longer training defaults (5000 iters)
  - configure_optimizer from model (separate weight decay groups, fused AdamW)
  - Loss EMA tracking
"""

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from model.transformer import Transformer, ModelConfig
from model.megabyte import MegabyteModel, MegabyteConfig


def get_lr(iter_num: int, max_iters: int, warmup: int, lr: float, min_lr: float) -> float:
    if iter_num < warmup:
        return lr * (iter_num + 1) / warmup
    if iter_num > max_iters:
        return min_lr
    ratio = (iter_num - warmup) / (max_iters - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (lr - min_lr)


def load_representation(name: str, data_dir: str):
    """Load a representation and its data."""
    from representations.byte import ByteRepresentation
    from representations.char import CharRepresentation
    from representations.bpe import BPERepresentation
    from representations.patch import PatchRepresentation
    from representations.small_bpe import SmallBPERepresentation
    from representations.word import WordRepresentation
    from representations.bpe_dropout import BPEDropoutRepresentation
    from representations.ngram import NgramByteRepresentation

    meta = json.loads((Path(data_dir) / "meta.json").read_text(encoding="utf-8"))
    block_size = meta["block_size"]

    # Map rep name to class and construction args
    if name == "byte":
        rep = ByteRepresentation(block_size=block_size)
    elif name == "char":
        rep = CharRepresentation(block_size=block_size)
        # Restore the saved vocab: without this, char_to_id/id_to_char stay empty
        # (encode→all zeros, decode→empty) and vocab_size keeps its 256 default.
        rep.char_to_id = meta["vocab"]
        rep.id_to_char = {idx: ch for ch, idx in meta["vocab"].items()}
        rep.config.vocab_size = meta["vocab_size"]
    elif name == "bpe":
        rep = BPERepresentation(block_size=block_size)
    elif name == "bpe_dropout":
        rep = BPEDropoutRepresentation(block_size=block_size, dropout_prob=0.0)
    elif name == "patch":
        rep = PatchRepresentation(block_size=block_size, patch_size=meta["patch_size"])
    elif name.startswith("bpe") and name != "bpe_dropout":
        # SmallBPE: bpe1000, bpe4000, bpe8000
        target_vocab = meta.get("target_vocab", int(name[3:]))
        rep = SmallBPERepresentation(block_size=block_size, vocab_size=target_vocab)
        rep.load_tokenizer(data_dir)
        rep._chars_per_token = meta.get("chars_per_token", 1.0)
    elif name == "word":
        rep = WordRepresentation(block_size=block_size, max_vocab=meta["vocab_size"])
        rep.load_vocab(data_dir)
        rep._chars_per_token = meta.get("chars_per_token", 5.0)
    elif name.startswith("ngram"):
        n = meta.get("n", int(name.replace("ngram", "")))
        rep = NgramByteRepresentation(block_size=block_size, n=n, max_vocab=meta["vocab_size"])
        rep.load_vocab(data_dir)
        rep._chars_per_token = meta.get("chars_per_token", float(n))
    elif name == "mamba":
        # Mamba is handled separately — data is byte-level
        rep = ByteRepresentation(block_size=block_size)
    else:
        raise ValueError(f"Unknown representation: {name}")

    # Load data
    dtype = np.uint8 if meta["dtype"] == "uint8" else np.uint16
    train_data = np.memmap(Path(data_dir) / "train.bin", dtype=dtype, mode="r")
    val_data = np.memmap(Path(data_dir) / "val.bin", dtype=dtype, mode="r")
    return rep, meta, train_data, val_data


def build_model(rep_name: str, meta: dict, n_layer: int = 6, n_head: int = 6,
                n_embd: int = 384, n_kv_head: int | None = None,
                grad_checkpoint: bool = False) -> torch.nn.Module:
    """Build the appropriate model for a representation."""
    if rep_name == "patch":
        cfg = MegabyteConfig(
            vocab_size=meta["vocab_size"],
            patch_size=meta["patch_size"],
            global_seq_len=meta["global_seq_len"],
            global_n_layer=n_layer,
            global_n_head=n_head,
            global_n_kv_head=n_kv_head,
            global_n_embd=n_embd,
            local_n_layer=2,
            local_n_head=4,
            local_n_embd=n_embd // 2,
            grad_checkpoint=grad_checkpoint,
        )
        return MegabyteModel(cfg)
    elif rep_name == "mamba":
        from model.mamba import MambaModel, MambaConfig
        cfg = MambaConfig(
            vocab_size=meta["vocab_size"],
            d_model=n_embd,
            n_layer=n_layer,
            block_size=meta["block_size"],
        )
        return MambaModel(cfg)
    else:
        cfg = ModelConfig(
            vocab_size=meta["vocab_size"],
            block_size=meta["block_size"],
            n_layer=n_layer,
            n_head=n_head,
            n_kv_head=n_kv_head,
            n_embd=n_embd,
            grad_checkpoint=grad_checkpoint,
        )
        return Transformer(cfg)


# Default learning rates per representation type.
# Smaller models (tiny vocab) can use higher LR; large-vocab BPE needs lower LR
# because the embedding gradients dominate.
DEFAULT_LR = {
    "byte": 3e-4,
    "char": 3e-4,
    "patch": 3e-4,
    "bpe": 1e-4,         # 16M params, 80% embeddings → lower LR
    "bpe_dropout": 1e-4,
    "bpe1000": 3e-4,
    "bpe4000": 2e-4,
    "bpe8000": 2e-4,
    "word": 2e-4,
    "ngram2": 3e-4,
    "ngram3": 2e-4,
    "mamba": 3e-4,
}


def generate_samples(model, rep, n_samples: int = 3, max_new_tokens: int = 200) -> list[str]:
    """Generate text samples from a trained model."""
    device = next(model.parameters()).device
    samples = []
    prompts = ["", "Once upon a time", "The little girl"]

    for i in range(min(n_samples, len(prompts))):
        prompt = prompts[i]
        try:
            if hasattr(rep, 'encode'):
                encoded = rep.encode(prompt) if prompt else np.array([], dtype=np.int64)
            else:
                encoded = np.array([], dtype=np.int64)

            if len(encoded) == 0:
                # Start from a random token
                if isinstance(model, MegabyteModel):
                    # Megabyte needs patch-shaped input
                    p_size = model.config.patch_size
                    g_seq = model.config.global_seq_len
                    idx = torch.randint(0, 256, (1, 1, p_size), device=device)
                else:
                    # Draw the seed token from the model's true embedding size,
                    # not rep.vocab_size — they can differ (e.g. small-BPE whose
                    # actual vocab is below the requested target), and an
                    # out-of-range index triggers a CUDA device-side assert.
                    idx = torch.randint(0, model.config.vocab_size, (1, 1), device=device)
            else:
                if isinstance(model, MegabyteModel):
                    # Reshape bytes into patches
                    p_size = model.config.patch_size
                    pad = p_size - (len(encoded) % p_size) if len(encoded) % p_size else 0
                    if pad:
                        encoded = np.concatenate([encoded, np.zeros(pad, dtype=np.int64)])
                    n_patches = len(encoded) // p_size
                    idx = torch.from_numpy(encoded[:n_patches * p_size].astype(np.int64))
                    idx = idx.reshape(1, n_patches, p_size).to(device)
                else:
                    idx = torch.from_numpy(encoded.astype(np.int64)).unsqueeze(0).to(device)

            out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.8, top_k=50)

            # Decode
            if isinstance(model, MegabyteModel):
                out_bytes = out[0].reshape(-1).cpu().numpy()
                text = rep.decode(out_bytes) if hasattr(rep, 'decode') else "<bytes>"
            else:
                out_ids = out[0].cpu().numpy()
                text = rep.decode(out_ids) if hasattr(rep, 'decode') else "<tokens>"

            samples.append({"prompt": prompt, "text": text})
        except Exception as e:
            samples.append({"prompt": prompt, "text": f"<generation error: {e}>"})

    return samples


def train_one_representation(
    rep_name: str,
    data_dir: str,
    out_dir: str,
    max_iters: int = 5000,
    batch_size: int = 16,
    eval_iters: int = 20,
    eval_interval: int = 500,
    learning_rate: float | None = None,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 256,
    n_kv_head: int | None = None,
    grad_checkpoint: bool = False,
    seed: int = 1337,
    fair_iters: bool = True,
    target_chars: int = 0,
    embed_lr_scale: float = 1.0,
    generate_after: bool = True,
):
    """Train one representation and return metrics.

    Args:
        fair_iters: If True, adjust max_iters so each rep sees the same total chars.
        target_chars: If >0, use this as the target chars-seen (overrides fair_iters calc).
        embed_lr_scale: Scale factor for embedding learning rate (1.0 = same as model).
        generate_after: If True, generate text samples after training.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    rep, meta, train_data, val_data = load_representation(rep_name, data_dir)

    # ── Fair comparison: equalize chars seen ──────────────────────────────────
    chars_per_token = meta.get("chars_per_token", 1.0)
    block_size = meta["block_size"]
    chars_per_iter = batch_size * block_size * chars_per_token

    if fair_iters and target_chars == 0:
        # Use the byte representation as reference: batch=16, block=1024, cpt=1.0
        # → reference chars = 16 * 1024 * 1.0 * max_iters
        ref_chars_per_iter = 16 * 1024 * 1.0
        target_chars = ref_chars_per_iter * max_iters

    if target_chars > 0:
        adjusted_iters = max(100, int(target_chars / chars_per_iter))
        if adjusted_iters != max_iters:
            print(f"  [fair] Adjusting iters: {max_iters} → {adjusted_iters} "
                  f"(cpt={chars_per_token:.1f}, chars/iter={chars_per_iter:.0f})", flush=True)
        max_iters = adjusted_iters

    # ── Build model ───────────────────────────────────────────────────────────
    # Enable gradient checkpointing for long-sequence reps (byte, char, patch)
    use_grad_ckpt = grad_checkpoint or (block_size >= 1024 and device == "cuda")
    model = build_model(rep_name, meta, n_layer, n_head, n_embd, n_kv_head, use_grad_ckpt).to(device)

    total_params = model.num_parameters()
    embed_params = model.num_embedding_parameters()
    modeling_params = model.num_modeling_parameters()

    # ── Learning rate ─────────────────────────────────────────────────────────
    if learning_rate is None:
        learning_rate = DEFAULT_LR.get(rep_name, 3e-4)

    print(f"\n{'='*60}", flush=True)
    print(f"Representation: {rep_name}", flush=True)
    print(f"Vocab size: {meta['vocab_size']}, Block size: {block_size}", flush=True)
    print(f"Params: {total_params:,} total | {embed_params:,} embed ({100*embed_params/total_params:.1f}%) | {modeling_params:,} modeling", flush=True)
    print(f"Train tokens: {meta['train_tokens']:,}, Val tokens: {meta['val_tokens']:,}", flush=True)
    print(f"Chars/token: {chars_per_token:.2f}, Iters: {max_iters}, LR: {learning_rate:.2e}", flush=True)
    print(f"Grad checkpoint: {use_grad_ckpt}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = model.configure_optimizer(
        lr=learning_rate, weight_decay=0.1,
        beta1=0.9, beta2=0.95, embed_lr_scale=embed_lr_scale,
    )
    ctx = torch.autocast(device_type=device, dtype=dtype) if device == "cuda" else torch.cuda.amp.nullcontext()

    def get_batch(data, split_name):
        return rep.get_batch(data, batch_size, device)

    def estimate_loss():
        model.eval()
        losses = {}
        for split_name, data in [("train", train_data), ("val", val_data)]:
            l = torch.zeros(eval_iters)
            for k in range(eval_iters):
                x, y = get_batch(data, split_name)
                with ctx:
                    _, loss = model(x, y)
                l[k] = loss.item()
            losses[split_name] = l.mean().item()
        model.train()
        return losses

    metrics = {
        "rep_name": rep_name,
        "vocab_size": meta["vocab_size"],
        "block_size": block_size,
        "total_params": total_params,
        "embed_params": embed_params,
        "modeling_params": modeling_params,
        "train_tokens": meta["train_tokens"],
        "val_tokens": meta["val_tokens"],
        "chars_per_token": chars_per_token,
        "learning_rate": learning_rate,
        "max_iters": max_iters,
        "grad_checkpoint": use_grad_ckpt,
        "iters": [],
    }

    t0 = time.time()
    best_val_loss = float("inf")
    loss_ema = None

    for iter_num in range(max_iters):
        lr = get_lr(iter_num, max_iters, min(100, max_iters // 10), learning_rate, learning_rate * 0.1)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, "train")
        with ctx:
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Loss EMA
        loss_val = loss.item()
        if loss_ema is None:
            loss_ema = loss_val
        else:
            loss_ema = 0.98 * loss_ema + 0.02 * loss_val

        if iter_num % 100 == 0:
            dt = time.time() - t0
            tok_s = batch_size * block_size * 100 / max(dt, 1e-6)
            print(f"  iter {iter_num:5d}: loss {loss_val:.4f} (ema {loss_ema:.4f}), lr {lr:.2e}, {tok_s:.0f} tok/s", flush=True)
            t0 = time.time()

        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            losses = estimate_loss()
            bpc = rep.bpc_from_loss(losses["val"])
            row = {
                "iter": iter_num,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "bpc": bpc,
                "lr": lr,
                "loss_ema": loss_ema,
            }
            metrics["iters"].append(row)
            print(f"  [eval] iter {iter_num}: train {losses['train']:.4f}, val {losses['val']:.4f}, BPC {bpc:.4f}", flush=True)
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]

    metrics["best_val_loss"] = best_val_loss
    metrics["best_bpc"] = rep.bpc_from_loss(best_val_loss)

    # ── Generate text samples ─────────────────────────────────────────────────
    if generate_after:
        print(f"\n  Generating samples...", flush=True)
        try:
            samples = generate_samples(model, rep, n_samples=3, max_new_tokens=200)
            metrics["samples"] = samples
            for s in samples:
                preview = s["text"][:200].replace("\n", " ")
                print(f"    [{s['prompt'] or '<empty>'}] → {preview}...", flush=True)
        except Exception as e:
            metrics["samples"] = []
            print(f"    Generation failed: {e}", flush=True)

    out_path = Path(out_dir) / f"{rep_name}_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\n  → Saved metrics to {out_path}", flush=True)
    print(f"  Best val loss: {best_val_loss:.4f}, BPC: {metrics['best_bpc']:.4f}\n", flush=True)

    return metrics


def main():
    # Force UTF-8 console output: Windows consoles default to a locale codec
    # (e.g. cp1250) that can't encode the → characters in progress messages.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--rep", required=True,
                        help="Which representation to train (byte, char, bpe, patch, bpe1000, bpe4000, bpe8000, word, bpe_dropout, ngram2, ngram3, mamba)")
    parser.add_argument("--data-dir", required=True, help="Path to prepared data")
    parser.add_argument("--out-dir", default="experiments", help="Where to save metrics")
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-kv-head", type=int, default=None, help="GQA: number of KV heads (None = n_head)")
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (None = per-rep default)")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--no-fair-iters", action="store_true", help="Disable fair iteration adjustment")
    parser.add_argument("--target-chars", type=int, default=0, help="Override target chars for fair comparison")
    parser.add_argument("--embed-lr-scale", type=float, default=1.0, help="Scale embedding LR relative to model LR")
    parser.add_argument("--no-generate", action="store_true", help="Skip text generation after training")
    parser.add_argument("--grad-checkpoint", action="store_true", help="Force gradient checkpointing")
    args = parser.parse_args()

    train_one_representation(
        rep_name=args.rep,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        eval_iters=args.eval_iters,
        eval_interval=args.eval_interval,
        learning_rate=args.lr,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        n_embd=args.n_embd,
        grad_checkpoint=args.grad_checkpoint,
        seed=args.seed,
        fair_iters=not args.no_fair_iters,
        target_chars=args.target_chars,
        embed_lr_scale=args.embed_lr_scale,
        generate_after=not args.no_generate,
    )


if __name__ == "__main__":
    main()