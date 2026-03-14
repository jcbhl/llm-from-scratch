import sys
import time

import pandas as pd
import tiktoken
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from gpt import GPTModel
from use_pretrained import PRETRAINED_124M_CONFIG, get_pretrained_model

BATCH_SIZE = 8


class SpamDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer: tiktoken.Encoding,
        max_length=None,
        pad_token_id=50256,
    ):
        self.df = pd.read_csv(path, header=None)

        self.encoded_texts = [tokenizer.encode(text) for text in self.df[1]]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length

            self.encoded_texts = [
                encoded_text[: self.max_length] for encoded_text in self.encoded_texts
            ]

        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

        pass

    def __getitem__(self, index):
        token_ids = self.encoded_texts[index]
        label = 1 if self.df.iloc[index][0] == "spam" else 0
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(
            label, dtype=torch.long
        )

    def __len__(self):
        return len(self.df)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


def get_data_loaders(tokenizer, base_path):
    train_dataset = SpamDataset(
        path=base_path + "train.csv", max_length=None, tokenizer=tokenizer
    )
    val_dataset = SpamDataset(
        path=base_path + "val.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer,
    )

    test_dataset = SpamDataset(
        path=base_path + "test.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


def prepare_for_finetuning(
    model: GPTModel, device: torch.device, tokenizer: tiktoken.Encoding
):
    for param in model.parameters():
        param.requires_grad = False

    model.output_head = torch.nn.Linear(
        PRETRAINED_124M_CONFIG.embedding_dimensionality, 2, device=device
    )

    for param in model.transformer_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.output_layernorm.parameters():
        param.requires_grad = True


def calc_accuracy_loader(
    data_loader: DataLoader, model: GPTModel, device: torch.device, num_batches=None
):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()

        else:
            break
    return correct_predictions / num_examples


def calc_loss_loader(
    data_loader: DataLoader, model: GPTModel, device: torch.device, num_batches=None
):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def calc_loss_batch(
    input_batch: torch.tensor,
    target_batch: torch.tensor,
    model: GPTModel,
    device: torch.device,
):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def evaluate_model(
    model: GPTModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    eval_iter: int,
):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_classifier(
    model: GPTModel,
    optimizer: AdamW,
    device: torch.device,
    num_epochs: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    eval_freq: int,
    eval_iter: int,
):
    global_step = 0

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            model.zero_grad()

            loss = calc_loss_batch(input_batch, target_batch, model, device)

            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )

        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter
        )

        print(f"Training accuracy: {train_accuracy * 100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy * 100:.2f}%")


def print_training_statistics(
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    model: GPTModel,
    start_time: float,
    end_time: float,
):
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(val_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)

    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")


def save_model_weights(model: GPTModel):
    torch.save(model.state_dict(), "spam-classifier.pth")
    print("Done saving model.")


def chat(model: GPTModel, tokenizer: tiktoken.Encoding, device: torch.device):
    model.load_state_dict(torch.load("spam-classifier.pth", map_location=device, weights_only=True))
    model.eval()

    print("Spam classifier ready. Type a message to classify (or 'quit' to exit).\n")

    while True:
        try:
            text = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if text.strip().lower() in ("quit", "exit"):
            break
        if not text.strip():
            continue

        token_ids = tokenizer.encode(text)
        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(input_tensor)[:, -1, :]

        predicted_label = torch.argmax(logits, dim=-1).item()
        print("SPAM" if predicted_label == 1 else "NOT SPAM")


def main():
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_pretrained_model(device)
    prepare_for_finetuning(model, device, tokenizer)

    if sys.argv[1] == "train":
        base_path = "data/sms_spam_collection/"
        train_loader, val_loader, test_loader = get_data_loaders(tokenizer, base_path)

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
        num_epochs = 5

        start_time = time.time()
        train_classifier(
            model,
            optimizer,
            device,
            num_epochs,
            train_loader,
            val_loader,
            eval_freq=50,
            eval_iter=5,
        )
        end_time = time.time()

        print_training_statistics(
            device, train_loader, val_loader, test_loader, model, start_time, end_time
        )
        save_model_weights(model)
    elif sys.argv[1] == "chat":
        chat(model, tokenizer, device)


if __name__ == "__main__":
    main()
