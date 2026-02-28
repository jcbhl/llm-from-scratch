import matplotlib

matplotlib.use("qtagg")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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


class PlaceholderDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        return x, y

    def __len__(self):
        return self.labels.shape[0]


def main():
    print(f"Torch version: {torch.__version__}")
    print(f"GPU status: {torch.cuda.is_available()}")

    x_train = torch.tensor(
        [[-1.2, 3.1], [-0.9, 2.9], [-0.5, 2.6], [2.3, -1.1], [2.7, -1.5]]
    )
    y_train = torch.tensor([0, 0, 0, 1, 1])
    train_ds = PlaceholderDataset(x_train, y_train)
    train_loader = DataLoader(
        train_ds, batch_size=2, shuffle=True, num_workers=0, drop_last=True
    )

    x_test = torch.tensor([[-0.8, 2.8], [2.6, -1.6]])
    y_test = torch.tensor([0, 1])
    test_ds = PlaceholderDataset(x_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=0)

    network = NeuralNetwork(2, 2)
    optimizer = torch.optim.SGD(network.parameters(), lr=0.05)

    print(
        f"Total trainable params: {sum([parameter.numel() for parameter in network.parameters() if parameter.requires_grad])}"
    )

    EPOCHS = 100
    loss_values = []
    for epoch in range(EPOCHS):
        print(f"Starting epoch: {epoch}")
        network.train()

        for batch_index, (features, labels) in enumerate(train_loader):
            logits = network(features)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                f"Done with epoch {epoch}, batch {batch_index}/{len(train_loader)}"
                f"Loss: {loss:.2f}"
            )
            loss_values.append(loss.detach().numpy())

    network.eval()
    with torch.no_grad():
        outputs = network(x_test)
        print(f"Raw model outputs: {outputs}")
        predictions = torch.softmax(outputs, dim=1)
        print(predictions)

    plt.plot(loss_values)
    plt.show()


if __name__ == "__main__":
    main()
