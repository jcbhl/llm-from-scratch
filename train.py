import sys

import tiktoken
import torch
from torch.utils.data import DataLoader

from gpt import GPT_CONFIG_124M, GPTModel
from tokenizer import get_dataloader

BATCH_SIZE = 16
LEARNING_RATE = 5e-4
NUM_EPOCHS = 10
VAL_FRAC = 0.1


def text_to_token_ids(text: str, tokenizer: tiktoken.Encoding):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids: torch.Tensor, tokenizer: tiktoken.Encoding):
    flattened = token_ids.squeeze(0).tolist()
    return tokenizer.decode(flattened)


def load_verdict():
    with open("./data/the-verdict.txt") as f:
        file_contents = f.read()

    return file_contents


def load_wikipedia():
    from datasets import load_dataset

    dataset = load_dataset("wikimedia/wikipedia", "20231101.en")
    train_set = dataset["train"]
    num_samples = 5
    subset = train_set.select(range(num_samples))
    overall_dataset = "<|endoftext|>".join(subset["text"])
    return overall_dataset


def compute_loss_for_batch(
    model: GPTModel,
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    device: torch.device,
):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)

    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten(),
    )
    return loss


def compute_loss_for_loader(
    model: GPTModel, loader: DataLoader, device: torch.device, num_batches: int
):
    total_loss = 0.0
    if len(loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(loader)
    else:
        num_batches = min(num_batches, len(loader))

    for i, (input_batch, target_batch) in enumerate(loader):
        if i < num_batches:
            loss = compute_loss_for_batch(model, input_batch, target_batch, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches


def generate_sample_text(
    model: GPTModel,
    context: str,
    length_to_generate: int,
    tokenizer: tiktoken.Encoding,
    device: torch.device,
):
    model.eval()
    context_size = model.positional_embeddings.weight.shape[0]
    encoded_context = text_to_token_ids(context, tokenizer).to(device)

    with torch.no_grad():
        model_inputs = encoded_context

        for _ in range(length_to_generate):
            trimmed_input = model_inputs[:, -context_size:]

            logits = model(trimmed_input)

            predictions = logits[:, -1, :]
            probabilities = torch.softmax(predictions, dim=-1)
            next_token_id = torch.argmax(probabilities, dim=-1, keepdim=True)
            model_inputs = torch.cat((model_inputs, next_token_id), dim=1)

            print(
                tokenizer.decode(next_token_id[0].tolist()).rstrip(),
                end="",
                flush=True,
            )

    model.train()


def evaluate_model(
    model: GPTModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    eval_iter: int,
):
    model.eval()
    with torch.no_grad():
        train_loss = compute_loss_for_loader(
            model, train_loader, device, num_batches=eval_iter
        )
        val_loss = compute_loss_for_loader(
            model, val_loader, device, num_batches=eval_iter
        )
    model.train()

    return train_loss, val_loss


def train_model(
    model: GPTModel,
    optimizer: torch.optim.AdamW,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    tokenizer: tiktoken.Encoding,
    batches_per_eval: int,
    eval_iter: int,
    initial_context: str,
):
    tokens_seen, global_step = 0, -1

    model.train()
    for epoch in range(NUM_EPOCHS):
        for input_batch, target_batch in train_loader:
            model.zero_grad()

            loss = compute_loss_for_batch(model, input_batch, target_batch, device)
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % batches_per_eval == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )

                print(f"Epoch {epoch} (Step {global_step:06d}):")
                print(f"Train loss {train_loss:.3f}")
                print(f"Val loss {val_loss:.3f}")

                generate_sample_text(model, initial_context, 100, tokenizer, device)


def get_dataloaders(raw_dataset: str):
    corpus_breakpoint = int(len(raw_dataset) * (1.0 - VAL_FRAC))
    training_set = raw_dataset[:corpus_breakpoint]
    val_set = raw_dataset[corpus_breakpoint:]
    train_loader = get_dataloader(
        training_set, batch_size=BATCH_SIZE, max_length=GPT_CONFIG_124M.context_length
    )
    val_loader = get_dataloader(
        val_set,
        batch_size=BATCH_SIZE,
        max_length=GPT_CONFIG_124M.context_length,
    )

    return train_loader, val_loader


def main():
    arg = sys.argv[1]
    match arg:
        case "verdict":
            dataset_str = load_verdict()
        case "wikipedia":
            dataset_str = load_wikipedia()
        case _:
            raise RuntimeError("unknown training set")

    tokenizer = tiktoken.get_encoding("gpt2")

    train_loader, val_loader = get_dataloaders(dataset_str)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(GPT_CONFIG_124M)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    train_model(
        model,
        optimizer,
        train_loader,
        val_loader,
        device,
        tokenizer,
        batches_per_eval=5,
        eval_iter=5,
        initial_context="Every effort moves you",
    )


if __name__ == "__main__":
    main()
