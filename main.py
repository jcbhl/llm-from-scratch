import torch
import torch.nn as nn

torch.manual_seed(42)


class NeuralNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 30),
            nn.ReLU(),
            torch.nn.Linear(30, 20),
            nn.ReLU(),
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, input):
        logits = self.layers(input)
        return logits


def main():
    print(f"Torch version: {torch.__version__}")
    print(f"GPU status: {torch.cuda.is_available()}")

    network = NeuralNetwork(1, 1)
    print(
        f"Number of trainable parameters in NN: {sum(p.numel() for p in network.parameters() if p.requires_grad)}"
    )
    for parameter in network.parameters():
        print(parameter.shape)
    print(network.forward(torch.tensor([0.0])))


if __name__ == "__main__":
    main()
