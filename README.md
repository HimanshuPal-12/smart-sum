# SmartSum - Length-Controlled Text Summarization with Information Preservation

A simple NLP class project (Himanshu Pal and Umesh Mamgai). The app creates
extractive summaries at **20%, 40%, and 60%** of the original article length
using a pretrained BERT sentence embedding model. No model training, no paid
APIs - just a clear, explainable pipeline.

## Team and work split

**Himanshu Pal**

- Core summarization pipeline (`summarizer.py`): sentence splitting, BERT
  sentence embeddings, document centroid, and cosine-similarity ranking
- Length control: word-budget selection of complete sentences and
  restoration of original sentence order
- Project setup: Python environment, pinned `requirements.txt`, and the
  Git/GitHub repository

**Umesh Mamgai**

- Evaluation module (`evaluation.py`): word counts, Flesch Reading Ease,
  and ROUGE scoring against reference summaries
- Streamlit interface (`app.py`): article input and file upload,
  sample-article dropdown, side-by-side summary display, and the manual
  factual-review checklist
- Sample dataset and tests: the nine articles and reference summaries in
  `sample_data/`, plus the acceptance tests in `tests/test_summarizer.py`

**Joint work (both members)**

- Project idea and algorithm selection (extractive, centrality-based
  summarization)
- Code review of each other's modules and verification of the measured
  results reported in this README
- Report writing and viva preparation

## How it works (algorithm)

1. **Input**: paste an article or upload a plain `.txt` file.
2. **Sentence splitting**: NLTK splits the text into sentences and the
   original wording and order are preserved.
3. **Sentence embeddings**: each sentence is encoded into a numeric vector
   with `sentence-transformers/bert-base-nli-mean-tokens` (a pretrained
   BERT-based sentence embedding model; it is not trained or fine-tuned here).
4. **Document centroid**: the mean of all sentence embeddings represents the
   whole document.
5. **Ranking**: each sentence gets a cosine-similarity score against the
   centroid. Higher score = more representative of the document. This is a
   simple centrality heuristic, not a supervised importance classifier.
6. **Length control**: for each ratio `r`, the word budget is
   `floor(original_words * r)`. Sentences are walked in rank order and a
   complete sentence is added only if it keeps the total at or below the
   budget.
7. **Restore order**: selected sentences are put back in their original
   document order, so the summary reads naturally.
8. **Evaluation**: word counts, Flesch Reading Ease, optional ROUGE (only
   when a reference summary is supplied), and a manual factual-review
   checklist.

BERT never writes the summary. It only produces the embeddings used for
ranking; the summary sentences are copied from the source text (extractive,
not abstractive).

## Folder structure

```
length_controlled_summarization/
|-- app.py                 # Streamlit user interface
|-- summarizer.py          # BERT embeddings, ranking, and length control
|-- evaluation.py          # Word counts, readability, optional ROUGE
|-- requirements.txt       # Python dependencies
|-- README.md              # This file
|-- demo_check.py          # One-off script that prints example results
|-- sample_data/            # Demo dataset (articles + references, see next section)
|   |-- articles/           # 9 sample articles: 01_education ... 09_history
|   `-- references/         # Matching human-written reference summaries
|-- tests/
    |-- test_summarizer.py        # Acceptance tests (T1-T10 core checks)
```

## Setup (Windows PowerShell)

Python 3.10 or 3.11 is required. From this folder:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the app

```powershell
streamlit run app.py
```

The browser opens at `http://localhost:8501`. Paste an article (or upload a
`.txt` file), optionally paste a reference summary, and press
**Generate summaries**.

## First-run notes (one-time downloads)

- **Model download**: the first run downloads
  `sentence-transformers/bert-base-nli-mean-tokens` (about 450 MB) from
  Hugging Face into the local cache (`%USERPROFILE%\.cache\huggingface`).
  Internet access is needed once; after that the model loads from disk.
- **NLTK data**: the app automatically downloads the `punkt` / `punkt_tab`
  sentence tokenizer data on first use. No manual step is normally needed.

## Evaluation explained (and what it cannot prove)

- **Length control** - words are counted as whitespace-separated tokens after
  trimming, everywhere in the app. Each summary shows the target budget, the
  actual word count, and the achieved percentage. Because complete sentences
  are selected, actual lengths are close to, but not exactly, the target.
- **Readability** - Flesch Reading Ease via `textstat`, with a plain-language
  label. It estimates linguistic difficulty only.
- **ROUGE-1/2/L** - computed with `rouge-score` only when a reference summary
  is supplied. ROUGE measures word/sequence overlap with the reference; it is
  not proof of factual accuracy or complete information preservation.
- **Factual accuracy** - no automatic score is invented. The app shows a
  manual review checklist (names, dates, numbers, key claims, negations,
  unsupported statements). Manual review is required.

## Sample dataset (9 articles + references)

`sample_data/` contains nine short original articles on varied topics -
education, health, environment, business, technology, science, sports,
culture, and history - plus a human-written reference summary (about 40%
of the article length) for each, used for ROUGE evaluation.

All articles and reference summaries were written by the project team for
demonstration purposes; there is no third-party copyright material, and the
dataset is intentionally small. It is used for testing and demonstration
only - the sentence-embedding model is pretrained and is never trained or
fine-tuned on this data.

Measured results at the 40% level (`python demo_check.py`,
model `bert-base-nli-mean-tokens`, measured on 19 Nov 2026):

| Topic | Words | Achieved % | ROUGE-1 | ROUGE-2 | ROUGE-L | Flesch |
|---|---|---|---|---|---|---|
| Education | 345 | 39.7% | 52.1 | 17.3 | 30.7 | 29.9 |
| Health | 319 | 37.3% | 53.1 | 33.0 | 41.5 | 44.8 |
| Environment | 329 | 39.5% | 42.7 | 16.5 | 32.0 | 33.2 |
| Business | 307 | 38.8% | 50.6 | 16.7 | 39.2 | 36.0 |
| Technology | 368 | 39.7% | 55.6 | 20.5 | 34.6 | 39.2 |
| Science | 329 | 39.8% | 46.5 | 18.2 | 31.2 | 32.2 |
| Sports | 324 | 37.7% | 55.5 | 28.0 | 43.4 | 34.4 |
| Culture | 297 | 38.7% | 64.1 | 34.8 | 50.0 | 30.0 |
| History | 311 | 36.3% | 52.2 | 20.0 | 35.3 | 48.4 |
| **Average** | **325** | **38.6%** | **52.5** | **22.8** | **37.5** | **36.5** |

("Achieved %" is the actual summary length as a percentage of the original
article when targeting 40%.) ROUGE numbers measure word overlap with the
reference summaries; they are not proof of factual accuracy. To try a
sample in the app, choose one from the "Try a sample article" dropdown and
press **Load sample** - this fills both the article and the reference box.

## Tests

From the project folder with the virtual environment activated:

```powershell
python tests\test_summarizer.py
```

This runs the acceptance checks for the core logic: non-empty summaries made
only of source sentences (T1), increasing budgets/lengths (T2), restored
source order (T3), empty input (T4), one-sentence input (T5), budgets too
small for any sentence (T6), UTF-8 `.txt` readability (T7), ROUGE without a
reference (T8), ROUGE with a reference (T9), and a friendly model-loading
error (T10). A dataset-wide check also generates summaries for **all nine
sample articles** at all three levels and computes ROUGE against each
article's reference. The first run downloads the model, so it needs
internet access.

## Limitations

- Extractive only: sentences are copied from the source; nothing is rewritten
  or generated.
- Target lengths are approximate because whole sentences are kept.
- The centrality heuristic can rank a repetitive but unimportant sentence
  highly; it does not "understand" facts.
- The embedding model is English-focused.
- Factual accuracy must be checked manually; no automated score can prove it.
- If the primary model cannot load in your environment, the documented
  fallback is `sentence-transformers/all-MiniLM-L6-v2` (smaller and faster,
  but a different model - document the change if used). This project uses the
  primary `bert-base-nli-mean-tokens` model.

