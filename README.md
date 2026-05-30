# GPT From Scratch

A minimal GPT implementation from scratch in PyTorch — model, BPE tokenizer, LoRA fine-tuning, and inference. Zero HuggingFace dependencies at runtime.

## What makes this different

Most GPT implementations stop at architecture. This one also implements **LoRA fine-tuning from scratch** — train adapters on real data with only **0.5% trainable parameters**.

| Feature | Most repos | This repo |
|---|---|---|
| Transformer architecture | ✅ | ✅ |
| BPE tokenizer from scratch | ❌ use HF | ✅ |
| LoRA fine-tuning from scratch | ❌ use PEFT | ✅ |
| Training on real data | ❌ just inference | ✅ |
| Web demo | ❌ | ✅ Gradio |

## Project Structure

| File | Description |
|---|---|
| `gpt_model.py` | GPT architecture (LayerNorm, CausalSelfAttention, MLP, TransformerBlock, GPT) |
| `tokenizer.py` | BPE tokenizer from scratch (GPT-2 style byte-level encoding) |
| `lora.py` | LoRA fine-tuning from scratch (low-rank adapters for attention) |
| `convert_weights.py` | Downloads GPT-2 weights from HuggingFace and maps into our model |
| `generate.py` | Text generation using our custom model + tokenizer |
| `run.py` | Interactive CLI for the GPT-2 model |
| `train.py` | Train a tiny GPT from scratch on custom data |
| `finetune.py` | Fine-tune GPT-2 on real data using LoRA |
| `app.py` | Gradio web demo |

## Quick Start

### 1. Download pretrained weights

```bash
pip install transformers accelerate
python convert_weights.py gpt2
```

### 2. Chat with GPT-2

```bash
python run.py "The future of AI is"
```

### 3. Fine-tune on Shakespeare

```bash
python finetune.py
```

### 4. Web demo

```bash
pip install gradio
python app.py
```

## Architecture

All built from scratch, no external ML libraries except PyTorch:

- **`gpt_model.py`** — Transformer decoder with multi-head causal self-attention, pre-layer normalization, GELU activation, tied embeddings
- **`tokenizer.py`** — Byte-Pair Encoding tokenizer with bytes-to-unicode mapping, regex pre-tokenization, and full BPE merge logic
- **`lora.py`** — Low-Rank Adaptation with forward pass merging, weight freezing, adapter save/load

Default config matches GPT-2 small (124M params): 12 layers, 12 heads, 768 hidden dim.
LoRA uses rank 8 on attention projections (Q, K, V, out) — 590K trainable params.

## Dependencies

- `torch` — runtime (model, training, inference)
- `gradio` — web demo (optional)
- `transformers` + `accelerate` — only for weight conversion
