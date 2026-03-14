import os
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

TRAIN_FRAC = 0.7
VAL_FRAC = 0.1
TEST_FRAC = 0.2


def download_and_unzip_spam_data(
    url: str, zip_path: str, extracted_path: str, data_path: Path
):
    if data_path.exists():
        print(f"{data_path} already exists. Skipping download and extraction.")
        return

    with urllib.request.urlopen(url) as response, open(zip_path, "wb") as out_file:
        out_file.write(response.read())

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_path)
    print(f"File downloaded and saved as {data_path}")


def get_balanced_dataset(df: pd.DataFrame):
    ham = df.loc[df[0] == "ham"]
    spam = df.loc[df[0] == "spam"]
    min_length = min(len(ham), len(spam))

    trimmed_ham = ham.sample(min_length)
    trimmed_spam = spam.sample(min_length)

    return pd.concat([trimmed_ham, trimmed_spam]).sample(frac=1).reset_index(drop=True)


def split_dataset(df: pd.DataFrame, train_frac: float, val_frac: float):
    train_end = int(len(df) * train_frac)
    val_end = train_end + int(len(df) * val_frac)

    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    return train_df, val_df, test_df


def main():
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "data/sms_spam_collection.zip"
    extracted_path = "data/sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

    df = pd.read_csv(data_file_path, sep="\t", header=None)
    balanced_dataset = get_balanced_dataset(df)

    train, val, test = split_dataset(balanced_dataset, TRAIN_FRAC, VAL_FRAC)

    train.to_csv(extracted_path + "/train.csv", index=None)
    val.to_csv(extracted_path + "/val.csv", index=None)
    test.to_csv(extracted_path + "/test.csv", index=None)


if __name__ == "__main__":
    main()
