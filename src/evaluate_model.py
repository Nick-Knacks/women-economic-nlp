"""
evaluate_model.py

Evaluates the trained classifier on the held-out test split:
  1. Per-category precision/recall/F1
  2. A basic bias check comparing performance across language_tag
     subgroups (sw / en / mixed)
  3. A simple PII flag check on raw text (privacy pass)

Usage:
    python src/evaluate_model.py
"""

import re
import json
import joblib
import pandas as pd
from sklearn.metrics import classification_report, f1_score

SPLIT_PATH = "data/processed/test_split.csv"
LEXICON_PATH = "lexicons/category_keywords.json"
MODEL_PATH = "src/model.pkl"
VECTORIZER_PATH = "src/vectorizer.pkl"

# very simple patterns - flags likely phone numbers / ID-like numbers
PII_PATTERNS = [
    r"\b0[17]\d{8}\b",       # Kenyan-style phone numbers e.g. 0712345678
    r"\b\d{7,10}\b",         # generic long digit sequences (possible ID numbers)
]


def load_categories() -> list:
    with open(LEXICON_PATH, "r", encoding="utf-8") as f:
        lexicon = json.load(f)
    return list(lexicon.keys())


def check_pii(text: str) -> bool:
    """Return True if text contains a pattern that looks like PII."""
    return any(re.search(p, text) for p in PII_PATTERNS)


def run_bias_check(df: pd.DataFrame, y_true, y_pred, categories: list):
    """Compare micro-F1 across language_tag subgroups."""
    df = df.copy().reset_index(drop=True)
    print("\n--- Bias check: F1 by language subgroup ---")
    for lang in df["language_tag"].unique():
        mask = df["language_tag"] == lang
        if mask.sum() == 0:
            continue
        f1 = f1_score(
            y_true[mask.values], y_pred[mask.values],
            average="micro", zero_division=0
        )
        print(f"  {lang:>8} (n={mask.sum():>2}): micro-F1 = {f1:.2f}")


def main():
    df = pd.read_csv(SPLIT_PATH)
    categories = load_categories()

    clf = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    X_test = df["clean_text"].fillna("")
    y_true = df[categories].values
    X_test_vec = vectorizer.transform(X_test)
    y_pred = clf.predict(X_test_vec)

    print("--- Classification report (per category) ---")
    print(classification_report(
        y_true, y_pred, target_names=categories, zero_division=0
    ))

    run_bias_check(df, y_true, y_pred, categories)

    print("\n--- Privacy pass: checking raw text for possible PII ---")
    flagged = df[df["text"].apply(check_pii)] if "text" in df.columns else pd.DataFrame()
    if len(flagged) > 0:
        print(f"  {len(flagged)} row(s) flagged for possible PII (phone/ID numbers).")
        print("  Review these rows before sharing/publishing this dataset.")
    else:
        print("  No obvious PII patterns detected in this sample.")


if __name__ == "__main__":
    main()
