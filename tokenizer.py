import re


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


def main():
    with open("data/the-verdict.txt") as f:
        file_contents = f.read()

    tokenizer = Tokenizer(file_contents)
    encoded = tokenizer.encode(""""It's the last he painted, you know,"
                           Mrs. Gisburn said with pardonable pride.""")
    decoded = tokenizer.decode(encoded)
    print(decoded)

    text = "Hello, do you like tea? <|EOF|> In the sunlit terraces of the palace."
    print(tokenizer.decode(tokenizer.encode(text)))


if __name__ == "__main__":
    main()
