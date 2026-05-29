import torch
from transformers import GPT2LMHeadModel
from gpt_model import GPT


def convert_and_save(model_size="gpt2", output_path=None):
    if output_path is None:
        output_path = f"{model_size.replace('/', '-')}_weights.pt"

    print(f"Downloading {model_size} from HuggingFace...")
    hf_model = GPT2LMHeadModel.from_pretrained(model_size)
    config = hf_model.config

    d_ff = config.n_inner if config.n_inner is not None else 4 * config.hidden_size

    our_model = GPT(
        vocab_size=config.vocab_size,
        d_model=config.hidden_size,
        n_heads=config.num_attention_heads,
        n_layers=config.num_hidden_layers,
        d_ff=d_ff,
        max_seq_len=config.n_positions,
    )

    hf_state = hf_model.state_dict()
    new_state = {}
    errors = []

    # Embeddings
    new_state["wte.weight"] = hf_state["transformer.wte.weight"]
    new_state["wpe.weight"] = hf_state["transformer.wpe.weight"]

    # Final layer norm
    new_state["ln_f.weight"] = hf_state["transformer.ln_f.weight"]
    new_state["ln_f.bias"] = hf_state["transformer.ln_f.bias"]

    # LM head
    if "lm_head.weight" in hf_state:
        new_state["lm_head.weight"] = hf_state["lm_head.weight"]

    for i in range(config.num_hidden_layers):
        # Layer norms
        new_state[f"h.{i}.ln_1.weight"] = hf_state[f"transformer.h.{i}.ln_1.weight"]
        new_state[f"h.{i}.ln_1.bias"] = hf_state[f"transformer.h.{i}.ln_1.bias"]
        new_state[f"h.{i}.ln_2.weight"] = hf_state[f"transformer.h.{i}.ln_2.weight"]
        new_state[f"h.{i}.ln_2.bias"] = hf_state[f"transformer.h.{i}.ln_2.bias"]

        # Attention: split combined QKV
        c_weight = hf_state[f"transformer.h.{i}.attn.c_attn.weight"]
        c_bias = hf_state[f"transformer.h.{i}.attn.c_attn.bias"]
        q_w, k_w, v_w = c_weight.chunk(3, dim=0)
        q_b, k_b, v_b = c_bias.chunk(3, dim=0)
        new_state[f"h.{i}.attn.q_proj.weight"] = q_w
        new_state[f"h.{i}.attn.q_proj.bias"] = q_b
        new_state[f"h.{i}.attn.k_proj.weight"] = k_w
        new_state[f"h.{i}.attn.k_proj.bias"] = k_b
        new_state[f"h.{i}.attn.v_proj.weight"] = v_w
        new_state[f"h.{i}.attn.v_proj.bias"] = v_b

        new_state[f"h.{i}.attn.out_proj.weight"] = hf_state[f"transformer.h.{i}.attn.c_proj.weight"]
        new_state[f"h.{i}.attn.out_proj.bias"] = hf_state[f"transformer.h.{i}.attn.c_proj.bias"]

        # MLP
        new_state[f"h.{i}.mlp.c_fc.weight"] = hf_state[f"transformer.h.{i}.mlp.c_fc.weight"]
        new_state[f"h.{i}.mlp.c_fc.bias"] = hf_state[f"transformer.h.{i}.mlp.c_fc.bias"]
        new_state[f"h.{i}.mlp.c_proj.weight"] = hf_state[f"transformer.h.{i}.mlp.c_proj.weight"]
        new_state[f"h.{i}.mlp.c_proj.bias"] = hf_state[f"transformer.h.{i}.mlp.c_proj.bias"]

    our_model.load_state_dict(new_state, strict=True)

    torch.save(our_model.state_dict(), output_path)
    print(f"Saved {model_size} weights to {output_path}")
    return output_path


if __name__ == "__main__":
    import sys
    size = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    convert_and_save(size)
