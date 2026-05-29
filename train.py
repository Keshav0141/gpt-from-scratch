import torch
import torch.nn.functional as F
from gpt_model import GPT


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tiny GPT config for fast demo training
    model = GPT(vocab_size=256, d_model=128, n_heads=4, n_layers=4, d_ff=512, max_seq_len=128)
    model.to(device)

    # Tiny training dataset: a simple counting sequence
    text = "0 1 2 3 4 5 6 7 8 9 " * 100
    tokens = [ord(c) for c in text]
    data = torch.tensor(tokens, dtype=torch.long).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    seq_len = 64
    num_epochs = 3

    print(f"Training tiny GPT on {device}...")
    for epoch in range(num_epochs):
        total_loss = 0
        steps = 0
        for i in range(0, len(data) - seq_len, seq_len):
            x = data[i : i + seq_len].unsqueeze(0)
            y = data[i + 1 : i + seq_len + 1].unsqueeze(0)

            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/steps:.4f}")

    # Generate sample
    context = torch.tensor([[ord("0")]], device=device)
    output = model.generate(context, max_new_tokens=40, temperature=0.8)
    result = "".join(chr(t) if 32 <= t <= 126 else "." for t in output[0].tolist())
    print(f"\nGenerated: {result}")

    torch.save(model.state_dict(), "trained_tiny_gpt.pt")
    print("Model saved to trained_tiny_gpt.pt")


if __name__ == "__main__":
    train()
