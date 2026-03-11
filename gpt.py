from dataclasses import dataclass

import torch
import torch.nn as nn

from layer_norm import LayerNorm

# TODO: do we need to train on bf16 rather than default of float32?
# torch.set_default_dtype(torch.bfloat16)
# torch.set_default_device('cuda')


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    context_length: int
    embedding_dimensionality: int
    num_heads: int
    num_transformer_blocks: int
    dropout_rate: float
    enable_qkv_bias: bool


GPT_CONFIG_124M = GPTConfig(
    vocab_size=50257,
    context_length=1024,
    embedding_dimensionality=768,
    num_heads=12,
    num_transformer_blocks=12,
    dropout_rate=0.1,
    enable_qkv_bias=False,
)


class GPTModel(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.token_embeddings = nn.Embedding(
            num_embeddings=cfg.vocab_size, embedding_dim=cfg.embedding_dimensionality
        )
        self.positional_embeddings = nn.Embedding(
            num_embeddings=cfg.context_length,
            embedding_dim=cfg.embedding_dimensionality,
        )
        self.embedding_dropout = nn.Dropout(cfg.dropout_rate)

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.num_transformer_blocks)]
        )

        self.output_layernorm = LayerNorm(cfg.embedding_dimensionality)
        self.output_head = nn.Linear(
            cfg.embedding_dimensionality, cfg.vocab_size, bias=False
        )

    def forward(self, in_idx: torch.Tensor):
        batch_size, sequence_length = in_idx.shape
        token_embedding_output = self.token_embeddings(in_idx)
        positional_embedding_output = self.positional_embeddings(
            torch.arange(sequence_length)
        )
        x = token_embedding_output + positional_embedding_output
        x = self.embedding_dropout(x)

        x = self.transformer_blocks(x)

        x = self.output_layernorm(x)
        logits = self.output_head(x)
        return logits


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()

    def forward(self, x: torch.Tensor):
        # apply layernorm
        # apply MHA
        # apply dropout

        # insert skip

        # apply layernorm
        # apply fully connected
        # apply dropout

        # insert skip
        pass


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()

    def forward(self, x: torch.Tensor):
        # crazy batch stuff
        pass


class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()

    def forward(self, x: torch.Tensor):
        # nn.Sequential(
        #   linear, output 4 * dimensionality?
        #   GELU
        #   linear
        # )
        pass


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor):
        # some numerical approximation
        pass


def main():
    model = GPTModel(GPT_CONFIG_124M)
    total_params = sum([torch.numel(param) for param in model.parameters()])
    print(f"Total params for the model: {total_params}")


if __name__ == "__main__":
    main()
