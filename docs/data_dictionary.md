# Data Dictionary

## `data/raw/narratives_raw.csv`
| Column | Type | Description |
|---|---|---|
| `id` | int | Unique row identifier |
| `text` | string | Raw narrative sentence (Swahili, English, or mixed) |

## `data/processed/narratives_clean.csv`
| Column | Type | Description |
|---|---|---|
| `id` | int | Unique row identifier |
| `text` | string | Original raw text |
| `clean_text` | string | Lowercased, punctuation-stripped, spelling-normalized text |
| `language_tag` | string | One of `sw` (Swahili), `en` (English), `mixed`, or `unknown` |

## `data/labeled/narratives_labeled.csv`
| Column | Type | Description |
|---|---|---|
| `id`, `text`, `clean_text`, `language_tag` | — | Same as above |
| `matched_categories` | list (string repr) | Categories matched by the lexicon tagger |
| `savings` | binary (0/1) | 1 if sentence relates to savings behavior |
| `debt` | binary (0/1) | 1 if sentence relates to debt/borrowing |
| `chama_obligations` | binary (0/1) | 1 if sentence relates to chama contributions/obligations |
| `business_risk` | binary (0/1) | 1 if sentence relates to business threats/losses |
| `care_work` | binary (0/1) | 1 if sentence relates to caregiving labor |
| `health_shock` | binary (0/1) | 1 if sentence relates to an acute illness/medical cost event |
| `lending_readiness` | binary (0/1) | 1 if sentence relates to collateral/guarantors/loan readiness |

Note: labels are **multi-label** — a single row can have multiple category
columns set to 1. See `annotation_guidelines.md` for rules on overlapping
categories.

## `data/processed/test_split.csv`
Held-out test rows (subset of `narratives_labeled.csv`) saved by
`train_classifier.py` and used by `evaluate_model.py` for scoring and the
language-subgroup bias check. Same columns as `narratives_labeled.csv`.

## `lexicons/category_keywords.json`
JSON object mapping each category name to a list of seed keywords/phrases
(Swahili and English) used by the rule-based tagger for first-pass labeling.
