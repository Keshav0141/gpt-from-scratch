import torch
import gradio as gr
from gpt_model import GPT
from lora import apply_lora, load_lora_weights
from generate import get_tokenizer, get_device


def load_model(weights_path="gpt2_weights.pt", lora_path=None):
    device = get_device()

    model = GPT(vocab_size=50257, d_model=768, n_heads=12, n_layers=12, d_ff=3072, max_seq_len=1024)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    if lora_path:
        apply_lora(model)
        load_lora_weights(model, lora_path)

    model.to(device)
    model.eval()
    return model


def generate_text(prompt, max_tokens=100, temperature=0.8, top_k=40):
    tokenizer = get_tokenizer()
    device = next(model.parameters()).device
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)

    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)

    return tokenizer.decode(output[0].tolist())


print("Loading model...")
model = load_model(lora_path="lora_shakespeare.pt")

demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Prompt", value="ROMEO:"),
        gr.Slider(10, 200, value=100, label="Max tokens"),
        gr.Slider(0.1, 1.5, value=0.8, label="Temperature"),
        gr.Slider(1, 100, value=40, label="Top-K"),
    ],
    outputs=gr.Textbox(label="Generated", lines=10),
    title="GPT From Scratch",
    description="Fine-tuned on Shakespeare. Built from scratch in PyTorch.",
)

if __name__ == "__main__":
    demo.launch()
