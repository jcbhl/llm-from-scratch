import torch

def main():
    print(f"Torch version: {torch.__version__}")
    print(f"GPU status: {torch.cuda.is_available()}")


if __name__ == "__main__":
    main()
