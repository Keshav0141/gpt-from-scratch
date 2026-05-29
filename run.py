import sys
import torch
from pathlib import Path
from gpt_model import GPT


def main():
    weights_path = Path("gpt2_weights.pt")

    if not weights_path.exists():
        print("Weights not found locally. Downloading from HuggingFace...")
        from convert_weights import convert_and_save
        convert_and_save("gpt2")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    from generate import load_model, generate
    model = load_model(device=device)

    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        print(f"Prompt: {prompt}")
        result = generate(model, prompt)
        print(f"Generated:\n{result}")
        return

    print("\nInteractive mode. Type a prompt or 'quit' to exit.\n")
    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.lower() in ("quit", "exit", "q"):
            break
        if not prompt.strip():
            continue
        result = generate(model, prompt)
        print(f"\n{result}\n")


if __name__ == "__main__":
    main()
