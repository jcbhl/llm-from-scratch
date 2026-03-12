from dataclasses import dataclass

import tiktoken
import torch
import torch.nn as nn

from layer_norm import LayerNorm


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
            torch.arange(sequence_length, device=in_idx.device)
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

        self.attention_layernorm = LayerNorm(cfg.embedding_dimensionality)
        self.mha = MultiHeadAttention(
            d_in=cfg.embedding_dimensionality,
            d_out=cfg.embedding_dimensionality,
            context_length=cfg.context_length,
            dropout_rate=cfg.dropout_rate,
            num_heads=cfg.num_heads,
            enable_qkv_bias=cfg.enable_qkv_bias,
        )
        self.attention_dropout = nn.Dropout(cfg.dropout_rate)

        self.dense_layernorm = LayerNorm(cfg.embedding_dimensionality)
        self.dense = FeedForward(cfg)
        self.dense_dropout = nn.Dropout(cfg.dropout_rate)

    def forward(self, x: torch.Tensor):
        skip = x
        x = self.attention_layernorm(x)
        x = self.mha(x)
        x = self.attention_dropout(x)
        x = x + skip

        skip = x
        x = self.dense_layernorm(x)
        x = self.dense(x)
        x = self.dense_dropout(x)
        x = x + skip
        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout_rate: float,
        num_heads: int,
        enable_qkv_bias=False,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dimensions = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=enable_qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=enable_qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=enable_qkv_bias)
        self.out_projection = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout_rate)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor):
        batch_size, num_tokens, d_in = x.shape
        queries: torch.Tensor = self.W_query(x)
        keys: torch.Tensor = self.W_key(x)
        values: torch.Tensor = self.W_value(x)

        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dimensions)
        values = values.view(
            batch_size, num_tokens, self.num_heads, self.head_dimensions
        )
        queries = queries.view(
            batch_size, num_tokens, self.num_heads, self.head_dimensions
        )

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attention_scores = queries @ keys.transpose(2, 3)
        mask_bools = self.mask.bool()[:num_tokens, :num_tokens]

        attention_scores.masked_fill_(mask_bools, -torch.inf)

        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        attention_weights: torch.Tensor = self.dropout(attention_weights)

        context_vector = (attention_weights @ values).transpose(1, 2)
        context_vector = context_vector.contiguous().view(
            batch_size, num_tokens, self.d_out
        )
        context_vector = self.out_projection(context_vector)
        return context_vector


class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(cfg.embedding_dimensionality, 4 * cfg.embedding_dimensionality),
            GELU(),
            nn.Linear(4 * cfg.embedding_dimensionality, cfg.embedding_dimensionality),
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


def greedy_generate(
    model: GPTModel,
    input_batch: torch.Tensor,
    length_to_generate: int,
    context_size: int,
    tokenizer: tiktoken.Encoding,
):
    for _ in range(length_to_generate):
        trimmed_input = input_batch[:, -context_size:]
        with torch.no_grad():
            logits = model(trimmed_input)

        predictions = logits[:, -1, :]
        probabilities = torch.softmax(predictions, dim=-1)
        next_token_id = torch.argmax(probabilities, dim=-1, keepdim=True)
        input_batch = torch.cat((input_batch, next_token_id), dim=1)

        print(tokenizer.decode(next_token_id[0].tolist()), end=" ", flush=True)


def run_inference(model: GPTModel, tokenizer: tiktoken.Encoding):
    num_tokens_to_generate = 5000
    initial_context = "Hello world!"
    tokenized_context = tokenizer.encode(initial_context)
    tokenized_tensor = torch.tensor(tokenized_context).to("cuda").unsqueeze(0)
    print(initial_context, end=" ")

    greedy_generate(
        model,
        tokenized_tensor,
        num_tokens_to_generate,
        GPT_CONFIG_124M.context_length,
        tokenizer,
    )


def main():
    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()
    model = model.to("cuda").to(torch.bfloat16)

    total_params = sum([torch.numel(param) for param in model.parameters()])
    print(f"Total params for the model: {total_params}")

    byte_size = sum(
        [torch.numel(param) * param.element_size() for param in model.parameters()]
    )
    size_mb = byte_size / (1024 * 1024)
    print(f"Overall model size: {size_mb:.2f} MB\n\n")

    run_inference(model, tokenizer)


if __name__ == "__main__":
    main()
