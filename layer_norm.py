import torch


class LayerNorm(torch.nn.Module):
    EPSILON = 1e-5

    def __init__(self, embedding_dimensionality):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(embedding_dimensionality))
        self.shift = torch.nn.Parameter(torch.zeros(embedding_dimensionality))

    def forward(self, inputs: torch.Tensor):
        mean = inputs.mean(dim=-1, keepdim=True)
        variance = inputs.var(dim=-1, keepdim=True, unbiased=False)

        normalized_inputs = (inputs - mean) / torch.sqrt(variance + self.EPSILON)
        return self.scale * normalized_inputs + self.shift


def main():
    size = 10
    layernorm = LayerNorm(size)
    input = torch.rand(size, dtype=torch.float32)
    print(f"Mean {input.mean()}\n var {input.var(unbiased=False)}")
    output = layernorm(input)
    print(f"Mean {output.mean()}\n var {output.var(unbiased=False)}")
    print(output)


if __name__ == "__main__":
    main()
