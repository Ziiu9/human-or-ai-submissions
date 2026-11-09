"""Model A: word TF-IDF + Logistic Regression."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import load_data, get_split, evaluate, SEED


def build_model():
    return make_pipeline(
        TfidfVectorizer(),
        LogisticRegression(max_iter=1000, random_state=SEED),
    )


if __name__ == "__main__":
    train, unlabeled, test = load_data()
    train_part, val_part = get_split(train)
    evaluate(build_model(), train_part, val_part)