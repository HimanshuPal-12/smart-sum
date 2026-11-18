# SmartSum - Length-Controlled Text Summarization with Information Preservation

A simple NLP class project (Himanshu Pal and Umesh). The app creates
extractive summaries at **20%, 40%, and 60%** of the original article length
using a pretrained BERT sentence embedding model. No model training, no paid
APIs - just a clear, explainable pipeline.

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
|-- sample_data/
|   |-- sample_article.txt        # Demo article
|   |-- reference_summary.txt     # Optional human summary, for ROUGE demo
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

## Sample use

1. Run the app (`streamlit run app.py`).
2. Open `sample_data/sample_article.txt`, copy its contents, and paste them
   into "Paste your article".
3. (Optional) paste `sample_data/reference_summary.txt` into the reference
   box to see ROUGE scores.
4. Press **Generate summaries** and compare the 20%, 40%, and 60% results.

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
error (T10). The first run downloads the model, so it needs internet access.

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

