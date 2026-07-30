"""
train_classifier.py

Trains a multi-label classifier (TF-IDF + One-vs-Rest Logistic Regression)
on the labeled narratives to predict economic categories from text.

Usage:
    python src/train_classifier.py
"""

import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split

LABELED_PATH = "data/labeled/narratives_labeled.csv"
LEXICON_PATH = "lexicons/category_keywords.json"
MODEL_PATH = "src/model.pkl"
VECTORIZER_PATH = "src/vectorizer.pkl"
SPLIT_PATH = "data/processed/test_split.csv"  # saved for evaluate_model.py

RANDOM_STATE = 42
TEST_SIZE = 0.25


def load_categories() -> list:
    with open(LEXICON_PATH, "r", encoding="utf-8") as f:
        lexicon = json.load(f)
    return list(lexicon.keys())


def main():
    df = pd.read_csv(LABELED_PATH)
    categories = load_categories()

    # drop rows with no labels at all - not useful for training
    df["label_sum"] = df[categories].sum(axis=1)
    df = df[df["label_sum"] > 0].reset_index(drop=True)
    print(f"Training on {len(df)} labeled rows (rows with >=1 category).")

    X = df["clean_text"].fillna("")
    y = df[categories]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=2000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    clf.fit(X_train_vec, y_train)

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    # save test split (with original text + language_tag) for evaluation
    test_df = df.loc[X_test.index].copy()
    test_df.to_csv(SPLIT_PATH, index=False)

    print(f"Model saved -> {MODEL_PATH}")
    print(f"Vectorizer saved -> {VECTORIZER_PATH}")
    print(f"Test split saved -> {SPLIT_PATH} ({len(test_df)} rows)")


if __name__ == "__main__":
    main()
