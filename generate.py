import torch
from pathlib import Path
from gpt_model import GPT


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(weights_path="gpt2_weights.pt", device=None):
    if device is None:
        device = get_device()

    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"Weights not found at {weights_path}")

    # GPT-2 small defaults
    model = GPT(vocab_size=50257, d_model=768, n_heads=12, n_layers=12,
                d_ff=3072, max_seq_len=1024)

    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def encode(prompt):
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokens = tokenizer.encode(prompt, return_tensors="pt")
    return tokens, tokenizer


def decode(tokens, tokenizer):
    return tokenizer.decode(tokens[0], skip_special_tokens=True)


@torch.no_grad()
def generate(model, prompt, max_new=100, temperature=0.8, top_k=40):
    device = next(model.parameters()).device
    input_ids, tokenizer = encode(prompt)
    input_ids = input_ids.to(device)
    output = model.generate(input_ids, max_new_tokens=max_new,
                            temperature=temperature, top_k=top_k)
    return decode(output, tokenizer)


if __name__ == "__main__":
    import sys
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "The future of AI is"

    device = get_device()
    print(f"Using device: {device}")
    print(f"Loading model...")
    model = load_model(device=device)
    result = generate(model, prompt)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {result}")
