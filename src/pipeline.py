"""
pipeline.py

Shared functions used by both the CLI scripts (clean_text.py,
lexicon_tagger.py, evaluate_model.py) and the Streamlit demo app.
No script-running logic lives here - just reusable functions.
"""

import re
import json
from rapidfuzz import fuzz
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

VARIANT_MAP = {
    "chamaa": "chama",
    "akibaa": "akiba",
    "mkopoo": "mkopo",
}

SWAHILI_MARKERS = [
    "na", "ya", "wa", "za", "ni", "kwa", "sana", "kila", "sasa",
    "hivi", "kutumia", "kupata", "kuwa", "nimekuwa", "ninafanya"
]

FUZZY_THRESHOLD = 85

PII_PATTERNS = [
    r"\b0[17]\d{8}\b",   # Kenyan-style phone numbers e.g. 0712345678
    r"\b\d{7,10}\b",     # generic long digit sequences (possible ID numbers)
]


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_variants(text: str) -> str:
    words = text.split()
    normalized = [VARIANT_MAP.get(w, w) for w in words]
    return " ".join(normalized)


def tag_language(text: str) -> str:
    try:
        detected = detect(text)
    except Exception:
        detected = "unknown"

    words = set(text.split())
    swahili_hits = len(words.intersection(SWAHILI_MARKERS))
    has_english_like = bool(re.search(r"\b(the|is|and|my|i|to|for)\b", text))

    if swahili_hits > 0 and has_english_like:
        return "mixed"
    if detected == "sw":
        return "sw"
    if detected == "en":
        return "en"
    return "mixed" if swahili_hits > 0 else "unknown"


def load_lexicon(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tag_sentence(text: str, lexicon: dict) -> list:
    matched = []
    for category, keywords in lexicon.items():
        for kw in keywords:
            if kw in text:
                matched.append(category)
                break
            if fuzz.partial_ratio(kw, text) >= FUZZY_THRESHOLD:
                matched.append(category)
                break
    return matched


def check_pii(text: str) -> bool:
    return any(re.search(p, text) for p in PII_PATTERNS)


def predict_sentence(raw_text, model, vectorizer, lexicon, categories):
    """
    Runs the full single-sentence pipeline:
    clean -> language tag -> lexicon match -> model prediction -> PII check.
    Returns a dict ready to display in the UI.
    """
    clean = normalize_variants(normalize_text(raw_text))
    lang = tag_language(clean)
    lexicon_matches = tag_sentence(clean, lexicon)

    X = vectorizer.transform([clean])
    probs = model.predict_proba(X)  # list of arrays, one per label (OneVsRest)

    # OneVsRestClassifier.predict_proba returns shape (n_samples, n_labels)
    # for most estimators - handle both possible shapes defensively.
    try:
        prob_values = probs[0]  # shape (n_labels,)
    except (IndexError, TypeError):
        prob_values = [p[0][1] for p in probs]  # fallback shape

    model_predictions = {
        cat: float(prob) for cat, prob in zip(categories, prob_values)
    }

    pii_flag = check_pii(raw_text)

    return {
        "clean_text": clean,
        "language_tag": lang,
        "lexicon_matches": lexicon_matches,
        "model_predictions": model_predictions,
        "pii_flag": pii_flag,
    }
