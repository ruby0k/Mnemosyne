"""Training loop — shared across all representations.

The same training code runs for every representation. Only the data loader
and model config change. This ensures fair comparison.
"""

import argparse
import json
import math
import time
from dataclasses import asdict
from datetime import datetime
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

    meta = json.loads((Path(data_dir) / "meta.json").read_text())
    block_size = meta["block_size"]

    # Map rep name to class and construction args
    if name == "byte":
        rep = ByteRepresentation(block_size=block_size)
    elif name == "char":
        rep = CharRepresentation(block_size=block_size)
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
    else:
        raise ValueError(f"Unknown representation: {name}")

    # Load data
    dtype = np.uint8 if meta["dtype"] == "uint8" else np.uint16
    train_data = np.memmap(Path(data_dir) / "train.bin", dtype=dtype, mode="r")
    val_data = np.memmap(Path(data_dir) / "val.bin", dtype=dtype, mode="r")
    return rep, meta, train_data, val_data


def build_model(rep_name: str, meta: dict, n_layer: int = 6, n_head: int = 6, n_embd: int = 384) -> torch.nn.Module:
    """Build the appropriate model for a representation."""
    if rep_name == "patch":
        cfg = MegabyteConfig(
            vocab_size=meta["vocab_size"],
            patch_size=meta["patch_size"],
            global_seq_len=meta["global_seq_len"],
            global_n_layer=n_layer,
            global_n_head=n_head,
            global_n_embd=n_embd,
            local_n_layer=2,
            local_n_head=4,
            local_n_embd=n_embd // 2,
        )
        return MegabyteModel(cfg)
    else:
        cfg = ModelConfig(
            vocab_size=meta["vocab_size"],
            block_size=meta["block_size"],
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
        )
        return Transformer(cfg)


def train_one_representation(
    rep_name: str,
    data_dir: str,
    out_dir: str,
    max_iters: int = 3000,
    batch_size: int = 32,
    eval_iters: int = 100,
    eval_interval: int = 300,
    learning_rate: float = 3e-4,
    n_layer: int = 6,
    n_head: int = 6,
    n_embd: int = 384,
    seed: int = 1337,
):
    """Train one representation and return metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    rep, meta, train_data, val_data = load_representation(rep_name, data_dir)
    model = build_model(rep_name, meta, n_layer, n_head, n_embd).to(device)

    total_params = model.num_parameters()
    embed_params = model.num_embedding_parameters()
    modeling_params = model.num_modeling_parameters()

    print(f"\n{'='*60}")
    print(f"Representation: {rep_name}")
    print(f"Vocab size: {meta['vocab_size']}, Block size: {meta['block_size']}")
    print(f"Params: {total_params:,} total | {embed_params:,} embed ({100*embed_params/total_params:.1f}%) | {modeling_params:,} modeling")
    print(f"Train tokens: {meta['train_tokens']:,}, Val tokens: {meta['val_tokens']:,}")
    print(f"{'='*60}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
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
        "block_size": meta["block_size"],
        "total_params": total_params,
        "embed_params": embed_params,
        "modeling_params": modeling_params,
        "train_tokens": meta["train_tokens"],
        "val_tokens": meta["val_tokens"],
        "chars_per_token": rep.chars_per_token,
        "iters": [],
    }

    t0 = time.time()
    best_val_loss = float("inf")

    for iter_num in range(max_iters):
        lr = get_lr(iter_num, max_iters, 100, learning_rate, learning_rate * 0.1)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, "train")
        with ctx:
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if iter_num % 100 == 0:
            dt = time.time() - t0
            tok_s = batch_size * meta["block_size"] * 100 / max(dt, 1e-6)
            print(f"  iter {iter_num:5d}: loss {loss.item():.4f}, lr {lr:.2e}, {tok_s:.0f} tok/s", flush=True)
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
            }
            metrics["iters"].append(row)
            print(f"  [eval] iter {iter_num}: train {losses['train']:.4f}, val {losses['val']:.4f}, BPC {bpc:.4f}", flush=True)
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]

    metrics["best_val_loss"] = best_val_loss
    metrics["best_bpc"] = rep.bpc_from_loss(best_val_loss)

    out_path = Path(out_dir) / f"{rep_name}_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"\n  → Saved metrics to {out_path}")
    print(f"  Best val loss: {best_val_loss:.4f}, BPC: {metrics['best_bpc']:.4f}\n")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep", required=True,
                        help="Which representation to train (byte, char, bpe, patch, bpe1000, bpe4000, bpe8000, word, bpe_dropout, ngram2, ngram3)")
    parser.add_argument("--data-dir", required=True, help="Path to prepared data")
    parser.add_argument("--out-dir", default="experiments", help="Where to save metrics")
    parser.add_argument("--max-iters", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    train_one_representation(
        rep_name=args.rep,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        learning_rate=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()