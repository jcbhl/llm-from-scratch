# LLM From Scratch

This repo contains a basic GPT-2 model implementation in PyTorch, built by following the [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch) textbook. Includes the full transformer architecture, from-scratch training, pretrained weight loading, and classifier + instruction finetuning.

I'd definitely recommend the textbook as an introduction to the concepts that back modern language models, as it was pretty comprehensible for me without any prior background in language modeling tasks. It doesn't cover some of the more advanced/recent concepts like RL generally, RLVR, performance optimizations (FlashAttention/KV caching), etc., but it gives you the fundamentals needed to explore those concepts. Having a GPU is definitely helpful for some of the examples, but not required. I also think the support for [Apple Silicon](https://developer.apple.com/metal/pytorch/) in PyTorch has gotten better since the book was written, so that's an alternative option to make it faster for laptops.

I followed along just by writing small Python scripts, which worked well for me, but you might want to use a proper Jupyter notebook since much of the example code is written with that in mind. A notebook environment also helps to avoid some of the cost of importing some of the big libraries like torch/tensorflow, since that can slow down the iteration speed when just running scripts after each change.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Training a model from scratch

Train on the included sample corpus:

```bash
uv run train.py verdict
```

Or on Wikipedia samples:

```bash
uv run train.py wikipedia
```

## Loading pretrained GPT-2 weights

Download pretrained weights and generate text:

```bash
uv run use_pretrained.py
```

This downloads the GPT-2 124M checkpoint from OpenAI's Azure bucket and generates text from a prompt.

## Fine-tuning

### Spam classification

Prepare the dataset, then train:

```bash
uv run prepare_spam_dataset.py
uv run spam_sft.py train
```

Chat with the trained classifier:

```bash
uv run spam_sft.py chat
```

### Instruction fine-tuning

```bash
uv run instruct_sft.py
```

## Formatting

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. It's included as a dev dependency:

```bash
uv run ruff format .
uv run ruff check --fix .
```

## Project structure

| File | Description |
|---|---|
| `gpt.py` | GPT-2 model architecture (attention, transformer blocks, config) |
| `train.py` | Training loop with evaluation and text generation |
| `tokenizer.py` | Custom tokenizer and dataloader utilities |
| `use_pretrained.py` | Load OpenAI GPT-2 weights into the model |
| `spam_sft.py` | Spam classifier fine-tuning and chat interface |
| `instruct_sft.py` | Instruction-following fine-tuning |
| `prepare_spam_dataset.py` | Download and split SMS spam dataset |
| `gpt_download.py` | GPT-2 weight download utility |
| `self_attention.py` | Standalone attention mechanism examples |
| `layer_norm.py` | Layer normalization implementation |
