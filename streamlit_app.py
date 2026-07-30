"""
streamlit_app.py

Interactive demo: type a sentence, see it cleaned, language-tagged,
lexicon-matched, classified by the trained model, and PII-checked.

This is the entry point Streamlit Community Cloud runs directly.
"""

import os
import sys
import json
import joblib
import streamlit as st

# make src/ importable regardless of where this file is run from
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from pipeline import predict_sentence  # noqa: E402

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "src", "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "src", "vectorizer.pkl")
LEXICON_PATH = os.path.join(BASE_DIR, "lexicons", "category_keywords.json")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    with open(LEXICON_PATH, "r", encoding="utf-8") as f:
        lexicon = json.load(f)
    categories = list(lexicon.keys())
    return model, vectorizer, lexicon, categories


st.set_page_config(page_title="Women's Economic Narratives NLP", layout="centered")
st.title("Women's Economic Narratives - NLP Demo")
st.write(
    "Type a sentence in Swahili, English, or mixed language describing an "
    "economic situation (savings, debt, chama, business risk, care work, "
    "health shocks, or lending readiness) and see how the pipeline "
    "processes it."
)

model, vectorizer, lexicon, categories = load_artifacts()

example = "Nimechelewa kulipa deni langu la chama na nina wasiwasi watanitoa."
sentence = st.text_area("Enter a sentence:", value=example, height=100)

if st.button("Analyze", type="primary"):
    if not sentence.strip():
        st.warning("Please enter a sentence first.")
    else:
        result = predict_sentence(sentence, model, vectorizer, lexicon, categories)

        st.subheader("Cleaned text")
        st.code(result["clean_text"])

        st.subheader("Detected language")
        st.write(result["language_tag"])

        st.subheader("Lexicon keyword matches")
        if result["lexicon_matches"]:
            st.write(", ".join(result["lexicon_matches"]))
        else:
            st.write("No keyword matches found.")

        st.subheader("Model predictions (confidence per category)")
        sorted_preds = sorted(
            result["model_predictions"].items(), key=lambda x: x[1], reverse=True
        )
        for cat, prob in sorted_preds:
            st.write(f"{cat}")
            st.progress(min(max(prob, 0.0), 1.0))
            st.caption(f"{prob:.0%} confidence")

        st.subheader("Privacy check")
        if result["pii_flag"]:
            st.error("Possible PII detected (phone/ID-like number) - review before sharing.")
        else:
            st.success("No obvious PII patterns detected.")

st.divider()
st.caption(
    "Prototype demo built on a small synthetic dataset. See the project "
    "README and model card for limitations before treating any output "
    "here as validated."
)
