# Model Card

## Overview
A multi-label text classifier that predicts which economic-narrative
categories (savings, debt, chama obligations, business risk, care work,
health shock, lending readiness) apply to a given sentence describing a
woman's economic situation, in Swahili, English, or code-switched text.

## Model Details
- **Type:** One-vs-Rest Logistic Regression (`scikit-learn`)
- **Features:** TF-IDF, unigrams + bigrams, max 2000 features
- **Training data:** Synthetic narrative dataset (100 sentences),
  auto-labeled via lexicon matching and manually reviewed
- **Labels:** 7 categories, multi-label (a sentence may have 0+ labels)
- **Train/test split:** 75% / 25%, random_state=42

## Intended Use
A proof-of-concept / prototype demonstrating a lightweight NLP workflow for
categorizing informal, multilingual narratives about women's economic
lives — intended to support routing to relevant support pathways (savings
products, credit programs, health-shock emergency support, etc.), not to
make final decisions about individuals.

## Performance
See console output from `evaluate_model.py` for per-category precision,
recall, and F1 on the held-out test split. Because the dataset is small
(100 synthetic sentences) and synthetic, these metrics should be treated
as illustrative of the pipeline, not as validated real-world performance.

## Known Limitations
- **Small, synthetic dataset.** These sentences were written to
  demonstrate the pipeline, not collected from real research participants.
  Real deployment requires a much larger, ethically-sourced dataset.
- **Code-switching detection is heuristic.** The `language_tag` field uses
  `langdetect` plus a simple Swahili-keyword check; it will misclassify
  some heavily-mixed or slang-heavy sentences.
- **Lexicon-seeded labels carry the biases of the seed lexicon.** Terms
  not anticipated by the initial keyword list (regional slang, newer
  phrasing) will be under-detected until the lexicon is expanded.
- **Small-sample bias check.** The subgroup analysis in
  `evaluate_model.py` (F1 by language_tag) is only indicative with this
  sample size — a production system needs a much larger test set to draw
  reliable conclusions about differential performance across dialects or
  language-mixing styles.

## Bias & Fairness Notes
`evaluate_model.py` includes a basic subgroup check comparing model
performance across the `sw` / `en` / `mixed` language tags. Any meaningful
F1 gap between subgroups (e.g. lower performance on `mixed` sentences)
would indicate the model or lexicon under-represents code-switched
phrasing, and the lexicon/training data should be expanded to close that
gap before any real-world use.

## Privacy Notes
`evaluate_model.py` includes a simple regex-based PII check (phone-number
and long-digit-sequence patterns) as a first-pass privacy screen. This is
**not** a comprehensive PII detector — a real deployment should use a
proper NER-based anonymization step (e.g. `spaCy` NER) before any
narrative data is stored, shared, or annotated by third parties, and
should follow informed-consent and data-minimization practices given the
sensitivity of the information being collected.

## Ethical Considerations
This project is explicitly about representing women's own language for
their economic realities, rather than imposing external
(e.g. bank-defined) categories. Any expansion of this work should continue
to validate categories and terminology with community input, not just
researcher assumptions, to avoid misrepresenting how women actually
describe their needs and barriers.
