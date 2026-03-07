import torch

device = "cpu"


class SelfAttention(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.Q = torch.nn.Linear(d_in, d_out, bias=False, device=device)
        self.K = torch.nn.Linear(d_in, d_out, bias=False, device=device)
        self.V = torch.nn.Linear(d_in, d_out, bias=False, device=device)

    def forward(self, input_embeddings: torch.tensor):
        queries = self.Q(input_embeddings)
        keys = self.K(input_embeddings)
        values = self.V(input_embeddings)

        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )

        context_vector = attention_weights @ values
        return context_vector


class CausalMaskedAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout_rate):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.Q = torch.nn.Linear(d_in, d_out, bias=False)
        self.K = torch.nn.Linear(d_in, d_out, bias=False)
        self.V = torch.nn.Linear(d_in, d_out, bias=False)
        self.dropout = torch.nn.Dropout(dropout_rate)

        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    # input_batch is a [batch_size x sequence_length x embedding_dimensions] tensor
    def forward(self, input_batch):
        batch_size, num_tokens, embedding_dimensions = input_batch.shape

        queries = self.Q(input_batch)
        keys = self.K(input_batch)
        values = self.V(input_batch)

        attention_scores = queries @ keys.transpose(1, 2)
        attention_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attention_weights = torch.softmax(
            attention_scores / embedding_dimensions**0.5, dim=-1
        )
        attention_weights = self.dropout(attention_weights)

        context_vectors = attention_weights @ values

        return context_vectors


def main():
    torch.manual_seed(123)
    d_in = 3
    d_out = 2
    # attention = SelfAttention(d_in, d_out)

    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],  # Your     (x^1)
            [0.55, 0.87, 0.66],  # journey  (x^2)
            [0.57, 0.85, 0.64],  # starts   (x^3)
            [0.22, 0.58, 0.33],  # with     (x^4)
            [0.77, 0.25, 0.10],  # one      (x^5)
            [0.05, 0.80, 0.55],  # step     (x^6)
        ],
        device=device,
    )

    input_batch = torch.stack((inputs, inputs), dim=0)

    context_length = input_batch.shape[1]
    causal_attention = CausalMaskedAttention(d_in, d_out, context_length, 0.0)
    context_vectors = causal_attention(input_batch)
    print(context_vectors.shape)


if __name__ == "__main__":
    main()
