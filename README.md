# GPT From Scratch

A minimal GPT implementation from scratch in PyTorch — model architecture, BPE tokenizer, training, and inference. Zero HuggingFace dependencies at runtime.

## Project Structure

| File | Description |
|---|---|
| `gpt_model.py` | GPT architecture (LayerNorm, CausalSelfAttention, MLP, TransformerBlock, GPT) |
| `tokenizer.py` | BPE tokenizer from scratch (GPT-2 style byte-level encoding) |
| `convert_weights.py` | Downloads GPT-2 weights from HuggingFace and maps into our model |
| `generate.py` | Text generation using our custom model + tokenizer |
| `run.py` | Interactive CLI for the GPT-2 model |
| `train.py` | Train a tiny GPT from scratch on custom data |

## Usage

### Run with pretrained GPT-2 weights

```bash
# Download + convert weights (requires transformers)
python convert_weights.py gpt2

# Interactive mode
python run.py

# Single prompt
python run.py "The future of AI is"
```

### Train a tiny GPT from scratch

```bash
python train.py
```

Trains a miniature GPT (4 layers, 4 heads, 128-dim). Loss drops in seconds on GPU.

## Architecture

All built from scratch, no external ML libraries except PyTorch:

- **`gpt_model.py`** — Transformer decoder with multi-head causal self-attention, pre-layer normalization, GELU activation, tied embeddings
- **`tokenizer.py`** — Byte-Pair Encoding tokenizer with bytes-to-unicode mapping, regex pre-tokenization, and full BPE merge logic
- **`train.py`** — Full training loop with AdamW optimizer, cross-entropy loss, gradient descent

Default config matches GPT-2 small (124M params): 12 layers, 12 heads, 768 hidden dim.

## Dependencies

- `torch` — runtime (model, training, inference)
- `transformers` + `accelerate` — only for weight conversion (`convert_weights.py`)
