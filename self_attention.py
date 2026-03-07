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


def main():
    torch.manual_seed(789)
    attention = SelfAttention(3, 2)

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

    print(attention(inputs))


if __name__ == "__main__":
    main()
