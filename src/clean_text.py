"""
clean_text.py

Cleans and normalizes raw Swahili/English/mixed-language narratives.
Adds a language tag (sw, en, mixed) to each row.

Usage:
    python src/clean_text.py
"""

import re
import pandas as pd
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # make langdetect deterministic

RAW_PATH = "data/raw/narratives_raw.csv"
OUT_PATH = "data/processed/narratives_clean.csv"

# Common spelling variants / normalization map (extend as you find more)
VARIANT_MAP = {
    "chamaa": "chama",
    "akibaa": "akiba",
    "mkopoo": "mkopo",
}

# A few common Swahili stopwords/markers used for a lightweight
# mixed-language heuristic (backup if langdetect is uncertain)
SWAHILI_MARKERS = [
    "na", "ya", "wa", "za", "ni", "kwa", "sana", "kila", "sasa",
    "hivi", "kutumia", "kupata", "kuwa", "nimekuwa", "ninafanya"
]


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\-]", " ", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text)        # collapse whitespace
    return text.strip()


def normalize_variants(text: str) -> str:
    """Replace known spelling variants with their standard form."""
    words = text.split()
    normalized = [VARIANT_MAP.get(w, w) for w in words]
    return " ".join(normalized)


def tag_language(text: str) -> str:
    """
    Tag a sentence as 'sw', 'en', or 'mixed'.
    Uses langdetect first, then a keyword heuristic to catch
    code-switched (mixed) sentences that langdetect misclassifies
    as purely one language.
    """
    try:
        detected = detect(text)
    except Exception:
        detected = "unknown"

    words = set(text.split())
    swahili_hits = len(words.intersection(SWAHILI_MARKERS))

    # crude mixed-language heuristic: if detected as English but has
    # Swahili marker words present (or vice versa), call it "mixed"
    has_english_like = bool(re.search(r"\b(the|is|and|my|i|to|for)\b", text))

    if swahili_hits > 0 and has_english_like:
        return "mixed"
    if detected == "sw":
        return "sw"
    if detected == "en":
        return "en"
    return "mixed" if swahili_hits > 0 else "unknown"


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clean_text"] = df["text"].apply(normalize_text)
    df["clean_text"] = df["clean_text"].apply(normalize_variants)
    df["language_tag"] = df["clean_text"].apply(tag_language)
    return df


def main():
    df = pd.read_csv(RAW_PATH)
    cleaned = clean_dataframe(df)
    cleaned.to_csv(OUT_PATH, index=False)
    print(f"Cleaned {len(cleaned)} rows -> {OUT_PATH}")
    print(cleaned["language_tag"].value_counts())


if __name__ == "__main__":
    main()
