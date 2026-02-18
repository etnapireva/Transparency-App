# Evaluation (Thesis)

Short evaluation for the DIELLA AI project: **sentiment** (gold labels) and **topic coherence** (NPMI).

## Quick run

From the project folder (where `config.py` and `run_evaluation.py` live):

```bash
python run_evaluation.py
```

Results are printed and saved to `evaluation_results.json`.

---

## 1. Sentiment evaluation

- **Gold file:** `evaluation_sentiment_gold.csv`
- **Columns:** `Speech` (text), `GoldLabel` (one of: **Pozitiv**, **Neutral**, **Negativ**)
- **Metrics:** Accuracy, macro F1, weighted F1, per-class report, confusion matrix

### How to add more gold labels

1. Open `evaluation_sentiment_gold.csv`.
2. Add rows: one column with the statement text, the other with `Pozitiv`, `Neutral`, or `Negativ` (or English: Positive, Neutral, Negative).
3. For a thesis, 30–50 manually labeled statements are enough. You can sample from `diella_speeches_clean.csv` and label them yourself.

---

## 2. Topic coherence (NPMI)

- **Data:** Main corpus (`diella_speeches_clean.csv`); script uses up to 500 rows.
- **Metric:** Document-level NPMI over NMF top words (higher = more coherent topics).
- No manual labels needed.

---

## For the thesis write-up

- **Sentiment:** Report accuracy and macro F1; mention that VADER was evaluated on a manually annotated sample of *n* statements.
- **Topics:** Report the NPMI coherence value and briefly explain that it measures how often the top topic words co-occur in documents.
- **Limitations:** Small gold set, English-oriented VADER, document-level coherence (not sliding window).
