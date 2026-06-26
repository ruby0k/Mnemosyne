<div align="center">

# 📊 Mnemosyne

> How does the way we represent text to a language model affect its learning?
> This project systematically experiments with different data representations
> on a small, fast-to-train model to find out what works best.

</div>

## Vision

Standard LLMs use BPE subword tokenization — a 50K+ vocabulary that consumes 38%+
of a small model's parameters in embeddings alone. But is that the best we can do?

**Mnemosyne** tests alternative representations:

- **Byte-level** — raw UTF-8 bytes, no tokenizer needed (vocab=256)
- **Character-level** — direct character modeling (vocab=~100)
- **Patch-based** — group bytes/chars into patches (Megabyte-style)
- **Hierarchical** — multi-resolution (byte → patch → sentence)
- **Token-free** — learned continuous representations
- **Compression-based** — feed compressed/encoded representations

## Experiment Design

Each representation is tested on the **same architecture** (small transformer, ~10-30M params)
on the **same data** (subset of FineWeb-Edu or TinyStories), with the **same training budget**
(iterations, LR schedule). The only variable is how text is encoded before the model sees it.

### Metrics

- **Loss curves** (train/val) — does the representation learn faster or reach lower loss?
- **Bits per character (BPC)** — representation-agnostic quality metric
- **Generation quality** — coherent text generation across representations
- **Parameter efficiency** — how much of the model is "representation" (embedding) vs "modeling"?
- **Training speed** — tokens/sec, wall clock to convergence
- **Memory** — peak VRAM per representation

## Running

```bash
# Install
uv sync

# Prepare data in all representations
uv run python scripts/prepare_tinystories.py

# Run all experiments (each ~10-30 min on RTX 5050)
uv run python scripts/run_experiments.py

# Compare results
uv run python eval.py --compare
```