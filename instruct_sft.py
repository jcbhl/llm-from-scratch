import json
import time
from functools import partial

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

from train import train_model
from use_pretrained import PRETRAINED_355M_CONFIG, get_pretrained_model

BASE_PATH = "data/instruct/"

TRAIN_FRAC = 0.85
VAL_FRAC = 0.1
BATCH_SIZE = 8


def get_prompt_template(entry: dict):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


def get_response_template(entry: dict):
    return f"\n\n### Response:\n{entry['output']}"


class InstructDataset(Dataset):
    def __init__(self, data: list[dict], tokenizer: tiktoken.Encoding):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_and_input = get_prompt_template(entry)
            response = get_response_template(entry)
            full_text = instruction_and_input + response

            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, idx):
        return self.encoded_texts[idx]

    def __len__(self):
        return len(self.encoded_texts)


# Pads each element in an input batch to the length of the longest sequence in the batch, with masking of the padded tokens to ignore_index
def pad_collator(
    batch: list[str],
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length: int = None,
    device: torch.device = "cpu",
):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


def get_data_loaders(device: torch.device, tokenizer: tiktoken.Encoding):
    with open(BASE_PATH + "instruction-data.json") as f:
        data = json.load(f)

    train_breakpoint = int(len(data) * TRAIN_FRAC)
    val_breakpoint = train_breakpoint + int(len(data) * VAL_FRAC)
    train_data = data[:train_breakpoint]
    val_data = data[train_breakpoint:val_breakpoint]
    test_data = data[val_breakpoint:]

    train_ds = InstructDataset(train_data, tokenizer)
    val_ds = InstructDataset(val_data, tokenizer)
    test_ds = InstructDataset(test_data, tokenizer)

    collator = partial(pad_collator, device=device, allowed_max_length=1024)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        collate_fn=collator,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, collate_fn=collator, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        collate_fn=collator,
        shuffle=True,
        drop_last=True,
    )

    return train_loader, val_loader, test_loader


def train_and_time(tokenizer, device, model, train_loader, val_loader):
    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    train_model(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        tokenizer=tokenizer,
        batches_per_eval=5,
        eval_iter=5,
        initial_context=get_prompt_template(
            {
                "instruction": "Convert the active sentence to passive: 'The chef cooks the meal every day.'",
                "input": "",
            }
        ),
        num_epochs=4,
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")


def main():
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_pretrained_model(device, PRETRAINED_355M_CONFIG, "355")

    train_loader, val_loader, test_loader = get_data_loaders(device, tokenizer)

    train_and_time(tokenizer, device, model, train_loader, val_loader)


if __name__ == "__main__":
    main()
