# GPT From Scratch

A minimal GPT (Generative Pre-trained Transformer) implementation from scratch in PyTorch.

## Project Structure

| File | Description |
|---|---|
| `gpt_model.py` | GPT architecture from scratch (LayerNorm, CausalSelfAttention, MLP, TransformerBlock, GPT) |
| `convert_weights.py` | Downloads GPT-2 weights from HuggingFace and converts them to our custom model format |
| `generate.py` | Text generation using our custom GPT model |
| `run.py` | Run the GPT-2 model interactively |
| `train.py` | Train a tiny GPT from scratch on a custom dataset |

## Usage

### Run with pretrained GPT-2 weights

```bash
# Download weights first
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

Trains a miniature GPT (4 layers, 4 heads, 128-dim). Loss drops in seconds on GPU. Saves to `trained_tiny_gpt.pt`.

## Architecture

The custom GPT model (`gpt_model.py`) implements:

- Token and position embeddings
- Multi-head causal self-attention
- Pre-layer normalization (GPT-2 style)
- GELU activation (tanh approximation)
- Transformer decoder blocks
- Output projection with tied embeddings

Defaults match GPT-2 small (124M params): 12 layers, 12 heads, 768 hidden dim.
