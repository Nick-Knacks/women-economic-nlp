# Women's Economic Narratives NLP Pipeline

A small end-to-end NLP pipeline demonstrating skills relevant to an
AI/NLP internship focused on Swahili and mixed-language narratives about
women's economic lives (savings, debt, chama obligations, business risk,
care work, health shocks, and lending readiness).

## Pipeline Overview

```
raw narratives  →  cleaning + language tagging  →  lexicon auto-tagging
      →  (manual review)  →  classifier training  →  evaluation + bias check
      →  documentation
```

## Project Structure

```
women-economic-nlp/
├── data/
│   ├── raw/narratives_raw.csv          # 100 synthetic example narratives
│   ├── processed/narratives_clean.csv  # cleaned + language-tagged
│   ├── processed/test_split.csv        # held-out test set (created by training)
│   └── labeled/narratives_labeled.csv  # lexicon-tagged (silver labels)
├── lexicons/category_keywords.json     # seed keyword lexicon per category
├── src/
│   ├── clean_text.py                   # cleaning + normalization + language tag
│   ├── lexicon_tagger.py               # rule-based multi-label auto-tagging
│   ├── train_classifier.py             # TF-IDF + Logistic Regression training
│   └── evaluate_model.py               # metrics + bias check + PII check
├── docs/
│   ├── annotation_guidelines.md        # category definitions + edge cases
│   ├── data_dictionary.md              # column definitions for every CSV
│   └── model_card.md                   # performance, limitations, ethics notes
├── requirements.txt
└── README.md
```

## How to Run

```bash
# 1. clean and language-tag the raw narratives
python3 src/clean_text.py

# 2. auto-tag categories using the lexicon
python3 src/lexicon_tagger.py
# -> open data/labeled/narratives_labeled.csv and manually review/correct
#    a sample of rows (especially any with zero matched categories)

# 3. train the classifier
python3 src/train_classifier.py

# 4. evaluate: per-category metrics + language-subgroup bias check + PII scan
python3 src/evaluate_model.py
```

## Why This Structure

This project mirrors the five core responsibility areas of the
internship it was built to demonstrate:

1. **Data collection & cleaning** → `clean_text.py`, `narratives_raw.csv`
2. **Domain category development** → `category_keywords.json`,
   `annotation_guidelines.md`
3. **NLP workflow testing** → `lexicon_tagger.py`, `train_classifier.py`
4. **Documentation** → `docs/` folder
5. **Responsible AI** → bias/subgroup check and PII scan in
   `evaluate_model.py`, ethics notes in `model_card.md`

## Notes

The dataset in `data/raw/narratives_raw.csv` is **synthetic** — written to
demonstrate the pipeline, not collected from real research participants.
See `docs/model_card.md` for full limitations and ethical considerations
before treating any output here as validated or production-ready.
