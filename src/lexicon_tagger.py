"""
lexicon_tagger.py

Applies keyword + fuzzy matching against the seed lexicon to auto-label
each cleaned narrative with one or more categories (multi-label).

These are "silver" labels - review and correct a sample of them by hand
before training the classifier.

Usage:
    python src/lexicon_tagger.py
"""

import json
import pandas as pd
from rapidfuzz import fuzz

CLEAN_PATH = "data/processed/narratives_clean.csv"
LEXICON_PATH = "lexicons/category_keywords.json"
OUT_PATH = "data/labeled/narratives_labeled.csv"

FUZZY_THRESHOLD = 85  # 0-100, higher = stricter match


def load_lexicon(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tag_sentence(text: str, lexicon: dict) -> list:
    """Return list of categories whose keywords match the text."""
    matched = []
    for category, keywords in lexicon.items():
        for kw in keywords:
            if kw in text:
                matched.append(category)
                break
            # fuzzy match catches spelling variants / partial phrases
            if fuzz.partial_ratio(kw, text) >= FUZZY_THRESHOLD:
                matched.append(category)
                break
    return matched


def auto_label_dataset(df: pd.DataFrame, lexicon: dict) -> pd.DataFrame:
    df = df.copy()
    categories = list(lexicon.keys())

    # get matched categories per row
    df["matched_categories"] = df["clean_text"].apply(
        lambda t: tag_sentence(t, lexicon)
    )

    # expand into binary columns, one per category
    for cat in categories:
        df[cat] = df["matched_categories"].apply(lambda cats: int(cat in cats))

    return df


def main():
    df = pd.read_csv(CLEAN_PATH)
    lexicon = load_lexicon(LEXICON_PATH)
    labeled = auto_label_dataset(df, lexicon)
    labeled.to_csv(OUT_PATH, index=False)

    print(f"Auto-labeled {len(labeled)} rows -> {OUT_PATH}")
    print("\nLabel counts per category:")
    for cat in lexicon.keys():
        print(f"  {cat}: {labeled[cat].sum()}")

    unlabeled = labeled[labeled["matched_categories"].apply(len) == 0]
    print(f"\n{len(unlabeled)} rows got NO category match - review these manually.")


if __name__ == "__main__":
    main()
