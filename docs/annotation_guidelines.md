# Annotation Guidelines

This document defines the seven economic-narrative categories used in this
project, with example phrases in Swahili and English, and rules for
handling edge cases where a sentence touches more than one category.

## Categories

### 1. Savings (`savings`)
Language about setting aside money, however small, for future use.
- *"Nimekuwa nikiweka akiba kidogo kila wiki."*
- *"I have been saving small amounts every week."*

### 2. Debt (`debt`)
Language about borrowing, owing money, or struggling to repay.
- *"Nimekopa pesa kutoka kwa jirani."*
- *"I borrowed money from my neighbor."*

### 3. Chama Obligations (`chama_obligations`)
Language about informal savings groups (chama), contributions, or
meeting-related obligations.
- *"Chama chetu kinatakiwa tuchangie kila mwezi."*
- *"Our chama requires us to contribute every month."*

### 4. Business Risk (`business_risk`)
Language about threats to a business - losses, competition, market
conditions, or risk of closing.
- *"Biashara yangu ilipata hasara kubwa."*
- *"My business suffered a big loss."*

### 5. Care Work (`care_work`)
Language about time/labor spent caring for children, elderly relatives,
or sick family members - not the illness itself, but the caregiving burden.
- *"Ninatumia muda mwingi kutunza wazazi wangu wazee."*
- *"I spend most of my time caring for my elderly parents."*

### 6. Health Shock (`health_shock`)
Language about an acute illness or medical event and its financial impact.
- *"Mtoto wangu ni mgonjwa na nimelazimika kutumia akiba yote."*
- *"My child is sick and I had to use all my savings."*

### 7. Lending Readiness (`lending_readiness`)
Language about collateral, guarantors, financial preparedness, or
explicit readiness/hesitation to take a loan.
- *"Sina dhamana ya kutosha kupata mkopo."*
- *"I don't have enough collateral to get a loan."*

## Edge Cases & Multi-Label Rules

- **A sentence can and often should carry more than one label.**
  Example: *"My child's hospital costs forced me to sell my property"*
  → both `health_shock` AND `business_risk` (property/asset loss).

- **Care work vs. health shock:** if the sentence is about ongoing caregiving
  labor (time, routine), label `care_work`. If it's about an acute illness
  event and its cost, label `health_shock`. Many sentences carry both.

- **Debt vs. chama obligations:** if the debt is specifically tied to a
  chama contribution or chama loan, label both `debt` and
  `chama_obligations`. General bank/individual borrowing is `debt` only.

- **Lending readiness vs. debt:** `lending_readiness` describes the
  *conditions* for taking a loan (collateral, confidence, guarantor) rather
  than an existing debt. A sentence can mention both if someone is
  currently in debt AND assessing readiness for a new loan.

- **Mixed-language sentences:** label based on meaning, not language.
  Code-switched sentences (e.g. *"Nilijaribu ku-save lakini..."*) are
  labeled the same way as monolingual ones - the `language_tag` column
  tracks language separately from category labels.

## Review Process

1. The lexicon tagger (`lexicon_tagger.py`) produces first-pass "silver"
   labels via keyword + fuzzy matching.
2. A human reviewer checks a sample (recommended: at least 20-30% of rows,
   and all rows with zero matched categories) and corrects mislabels.
3. Corrected labels become the "gold" set used to train the classifier.
