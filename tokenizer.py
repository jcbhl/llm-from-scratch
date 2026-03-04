import re

import tiktoken
import torch
from torch.utils.data import DataLoader


class Tokenizer:
    UNKNOWN_TOKEN = "<|UNK|>"
    EOF_TOKEN = "<|EOF|>"

    def __init__(self, training_set: str):
        sorted_vocabulary = self._get_sorted_vocabulary(training_set)
        self.id_to_token = dict(enumerate(sorted_vocabulary))
        self.token_to_id = {v: k for k, v in enumerate(self.id_to_token.values())}

    def _get_sorted_vocabulary(self, training_set):
        all_training_tokens = Tokenizer.tokenize(training_set)
        sorted_vocab = sorted(set(all_training_tokens))
        sorted_vocab.extend([self.EOF_TOKEN, self.UNKNOWN_TOKEN])
        return sorted_vocab

    def encode(self, raw_input: str):
        input_tokens = Tokenizer.tokenize(raw_input)
        encoded = [
            self.token_to_id.get(token, self.token_to_id[self.UNKNOWN_TOKEN])
            for token in input_tokens
        ]
        return encoded

    def decode(self, output_token_ids: list[int]):
        output_text = " ".join(
            [self.id_to_token[token_id] for token_id in output_token_ids]
        )
        output_text_no_punctuation_whitespace = re.sub(
            r'\s+([,.?!"()\'])', r"\1", output_text
        )
        return output_text_no_punctuation_whitespace

    def tokenize(s: str):
        tokens_with_whitespace = re.split(r'([,.:;?_!"()\']|--|\s)', s)
        tokens_without_whitespace = [
            token.strip() for token in tokens_with_whitespace if token.strip()
        ]
        return tokens_without_whitespace


class SlidingWindowDataset:
    def __init__(self, corpus: str, context_length: int, stride: int):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        tokenized_corpus = self.tokenizer.encode(corpus)

        self.input_token_ids = []
        self.output_token_ids = []

        for i in range(0, len(tokenized_corpus) - context_length, stride):
            input_tokens = tokenized_corpus[i : i + context_length]
            output_tokens = tokenized_corpus[i + 1 : i + 1 + context_length]
            self.input_token_ids.append(torch.tensor(input_tokens))
            self.output_token_ids.append(torch.tensor(output_tokens))

    def __len__(self):
        return len(self.input_token_ids)

    def __getitem__(self, idx):
        return self.input_token_ids[idx], self.output_token_ids[idx]


def get_dataloader(
    corpus: str,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    dataset = SlidingWindowDataset(corpus, context_length=max_length, stride=stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


def main():
    with open("data/the-verdict.txt") as f:
        file_contents = f.read()

    dataloader = get_dataloader(file_contents, max_length=5)
    print(next(iter(dataloader)))


if __name__ == "__main__":
    main()
