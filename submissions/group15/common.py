"""
Shared code for the whole team. Everyone imports from here so that
we all read the data and preprocess text in exactly the same way.
"""
import re

import pandas as pd

DATA_DIR = "data"


def load_data(data_dir=DATA_DIR):
    """Read the three competition files."""
    train = pd.read_csv(f"{data_dir}/train.csv")
    unlabeled = pd.read_csv(f"{data_dir}/unlabeled.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")
    return train, unlabeled, test


def clean_text(text, mode="raw"):
    """
    mode="raw":     keep the text exactly as it is (default)
    mode="flatten": turn every run of whitespace (spaces, newlines, tabs)
                    into one space, so blank lines can't be used as a clue
    """
    if mode == "raw":
        return text
    if mode == "flatten":
        return re.sub(r"\s+", " ", text).strip()
    raise ValueError(f"unknown mode: {mode}")


if __name__ == "__main__":
    # quick sanity check: python common.py
    train, unlabeled, test = load_data()
    print("train", train.shape, "unlabeled", unlabeled.shape, "test", test.shape)

    example = "Ingredients:\n\n1 cup sugar\n\n2 eggs"
    print("raw:    ", repr(clean_text(example, "raw")))
    print("flatten:", repr(clean_text(example, "flatten")))

    flat = train["text"].map(lambda t: clean_text(t, "flatten"))
    print("passages with a blank line after flatten:", flat.str.contains("\n\n").sum())
