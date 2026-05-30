# GPT From Scratch

I built a GPT implementation from scratch in PyTorch — model architecture, BPE tokenizer, LoRA fine-tuning, inference, and a web demo. Zero HuggingFace dependencies at runtime.

## What I built

| File | What it does |
|---|---|
| `gpt_model.py` | Transformer decoder with multi-head causal attention, layer norm, GELU, tied embeddings |
| `tokenizer.py` | Byte-Pair Encoding tokenizer with bytes-to-unicode mapping and BPE merge logic |
| `lora.py` | Low-Rank Adaptation — freezes base model, trains only small adapter matrices (0.5% of params) |
| `convert_weights.py` | Downloads GPT-2 weights and maps them to my model |
| `generate.py` | Text generation with temperature and top-k sampling |
| `run.py` | Interactive CLI |
| `train.py` | Train a small GPT from scratch on any text |
| `finetune.py` | Fine-tune GPT-2 on real data using LoRA |
| `app.py` | Gradio web demo |

## How to use

```bash
# Download pretrained weights
pip install transformers accelerate
python convert_weights.py gpt2

# Chat
python run.py "The future of AI is"

# Fine-tune on Shakespeare
python finetune.py

# Web demo
pip install gradio
python app.py
```

## Details

- GPT-2 small config: 12 layers, 12 heads, 768 hidden dim (124M params)
- LoRA rank 8 on attention projections — only 590K trainable params
- Fine-tuned on Tiny Shakespeare dataset — shows adaptation from generic text to Shakespearean style
- Web interface built with Gradio
