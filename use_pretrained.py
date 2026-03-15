import json
import os
from dataclasses import replace

import numpy as np
import tensorflow as tf
import tiktoken
import torch

from gpt import GPT_CONFIG_124M, GPTConfig, GPTModel
from gpt_download import load_gpt2_params_from_tf_ckpt
from train import generate_sample_text, token_ids_to_text

MODEL_CONFIGS = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

PRETRAINED_124M_CONFIG = replace(
    GPT_CONFIG_124M, context_length=1024, enable_qkv_bias=True
)
PRETRAINED_355M_CONFIG = replace(
    GPT_CONFIG_124M,
    embedding_dimensionality=1024,
    num_heads=16,
    num_transformer_blocks=24,
    context_length=1024,
    enable_qkv_bias=True,
)


def assign(left: torch.Tensor, right: torch.Tensor):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")

    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt: GPTModel, params):
    gpt.positional_embeddings.weight = assign(
        gpt.positional_embeddings.weight, params["wpe"]
    )
    gpt.token_embeddings.weight = assign(gpt.token_embeddings.weight, params["wte"])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1
        )
        gpt.transformer_blocks[b].mha.W_query.weight = assign(
            gpt.transformer_blocks[b].mha.W_query.weight, q_w.T
        )
        gpt.transformer_blocks[b].mha.W_key.weight = assign(
            gpt.transformer_blocks[b].mha.W_key.weight, k_w.T
        )
        gpt.transformer_blocks[b].mha.W_value.weight = assign(
            gpt.transformer_blocks[b].mha.W_value.weight, v_w.T
        )

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1
        )
        gpt.transformer_blocks[b].mha.W_query.bias = assign(
            gpt.transformer_blocks[b].mha.W_query.bias, q_b
        )
        gpt.transformer_blocks[b].mha.W_key.bias = assign(
            gpt.transformer_blocks[b].mha.W_key.bias, k_b
        )
        gpt.transformer_blocks[b].mha.W_value.bias = assign(
            gpt.transformer_blocks[b].mha.W_value.bias, v_b
        )

        gpt.transformer_blocks[b].mha.out_projection.weight = assign(
            gpt.transformer_blocks[b].mha.out_projection.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T,
        )
        gpt.transformer_blocks[b].mha.out_projection.bias = assign(
            gpt.transformer_blocks[b].mha.out_projection.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"],
        )

        gpt.transformer_blocks[b].dense.layers[0].weight = assign(
            gpt.transformer_blocks[b].dense.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T,
        )
        gpt.transformer_blocks[b].dense.layers[0].bias = assign(
            gpt.transformer_blocks[b].dense.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"],
        )
        gpt.transformer_blocks[b].dense.layers[2].weight = assign(
            gpt.transformer_blocks[b].dense.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T,
        )
        gpt.transformer_blocks[b].dense.layers[2].bias = assign(
            gpt.transformer_blocks[b].dense.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"],
        )

        gpt.transformer_blocks[b].attention_layernorm.scale = assign(
            gpt.transformer_blocks[b].attention_layernorm.scale,
            params["blocks"][b]["ln_1"]["g"],
        )
        gpt.transformer_blocks[b].attention_layernorm.shift = assign(
            gpt.transformer_blocks[b].attention_layernorm.shift,
            params["blocks"][b]["ln_1"]["b"],
        )
        gpt.transformer_blocks[b].dense_layernorm.scale = assign(
            gpt.transformer_blocks[b].dense_layernorm.scale,
            params["blocks"][b]["ln_2"]["g"],
        )
        gpt.transformer_blocks[b].dense_layernorm.shift = assign(
            gpt.transformer_blocks[b].dense_layernorm.shift,
            params["blocks"][b]["ln_2"]["b"],
        )

    gpt.output_layernorm.scale = assign(gpt.output_layernorm.scale, params["g"])
    gpt.output_layernorm.shift = assign(gpt.output_layernorm.shift, params["b"])
    gpt.output_head.weight = assign(gpt.output_head.weight, params["wte"])


def load_from_checkpoint(model: GPTModel, param_string: str = "124"):
    tf_ckpt_path = tf.train.latest_checkpoint(f"pretrained_weights/{param_string}M")
    with open(
        os.path.join(f"pretrained_weights/{param_string}M", "hparams.json"),
        encoding="utf-8",
    ) as f:
        settings = json.load(f)

    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
    load_weights_into_gpt(model, params)


def get_pretrained_model(
    device: torch.device,
    config: GPTConfig = PRETRAINED_124M_CONFIG,
    param_string: str = "124",
):
    model = GPTModel(config)
    model.eval()
    load_from_checkpoint(model, param_string)
    model.to(device)

    return model


def main():
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_pretrained_model(device)

    token_ids = generate_sample_text(
        model,
        context="Every effort moves you",
        length_to_generate=100000,
        tokenizer=tokenizer,
        device=device,
        top_k=50,
        temperature=1.5,
    )
    print("Output text: \n,", token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":
    main()
