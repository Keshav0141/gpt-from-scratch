import torch
import torch.nn.functional as F
import urllib.request
from pathlib import Path
from gpt_model import GPT
from lora import apply_lora, save_lora_weights
from tokenizer import BPETokenizer


def load_data():
    path = Path("tinyshakespeare.txt")
    if not path.exists():
        print("Downloading Tiny Shakespeare dataset...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, path)
    with open(path, encoding="utf-8") as f:
        return f.read()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model with pretrained weights...")
    model = GPT(vocab_size=50257, d_model=768, n_heads=12, n_layers=12, d_ff=3072, max_seq_len=1024)
    state = torch.load("gpt2_weights.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    print("Loading tokenizer...")
    tokenizer = BPETokenizer.from_pretrained("gpt2")

    text = load_data()
    tokens = tokenizer.encode(text)
    tokens = tokens[:50000]
    data = torch.tensor(tokens, dtype=torch.long, device=device)

    seq_len = 256
    stride = 256

    print(f"Dataset: {len(tokens)} tokens")
    print(f"Sequence length: {seq_len}, Stride: {stride}")

    print("\n--- Before fine-tuning ---")
    prompt = "ROMEO:"
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=80, temperature=0.8, top_k=40)
    print(tokenizer.decode(output[0].tolist()))

    print("\nApplying LoRA...")
    model = apply_lora(model, r=8, alpha=16)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=5e-4
    )

    num_epochs = 1
    steps_per_epoch = (len(data) - seq_len) // stride

    print(f"\nTraining for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(0, len(data) - seq_len, stride):
            x = data[i : i + seq_len].unsqueeze(0)
            y = data[i + 1 : i + seq_len + 1].unsqueeze(0)

            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / steps_per_epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    save_lora_weights(model, "lora_shakespeare.pt")

    print("\n--- After fine-tuning ---")
    model.eval()
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=80, temperature=0.8, top_k=40)
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
