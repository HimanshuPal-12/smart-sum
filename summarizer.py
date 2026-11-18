"""Core summarization logic for SmartSum.

Pipeline:
  1. Split text into sentences using NLTK (original wording is preserved).
  2. Encode sentences with a pretrained BERT sentence embedding model.
  3. Build a document centroid (mean of all sentence embeddings).
  4. Rank sentences by cosine similarity to the centroid.
  5. Select complete sentences inside a word budget for each target ratio.
  6. Restore the original sentence order in the final summary.

This is an extractive method: selected sentences are copied from the source
text. BERT only provides the embeddings used for ranking; it does not write
the summary.
"""

import math

import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "sentence-transformers/bert-base-nli-mean-tokens"

# Module-level caches so the model and NLTK data are loaded only once.
_MODEL = None
_NLTK_READY = False


def _ensure_nltk_punkt():
    """Make sure the NLTK sentence tokenizer data is available (one-time)."""
    global _NLTK_READY
    if _NLTK_READY:
        return
    for resource in ("punkt", "punkt_tab"):  # punkt_tab is needed by newer NLTK
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not download the NLTK '{resource}' tokenizer data. "
                    "Connect to the internet and run: "
                    f"python -c \"import nltk; nltk.download('{resource}')\" "
                    "once, then restart the application."
                ) from exc
    _NLTK_READY = True


def load_model():
    """Load the pretrained BERT sentence embedding model once and cache it."""
    global _MODEL
    if _MODEL is None:
        try:
            _MODEL = SentenceTransformer(MODEL_NAME)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load the sentence embedding model '{MODEL_NAME}'. "
                "The first run needs internet access to download it "
                "(about 450 MB). After the first download it is cached on disk. "
                f"Details: {exc}"
            ) from exc
    return _MODEL


def split_into_sentences(text):
    """Split text into a list of sentences using NLTK, in original order."""
    _ensure_nltk_punkt()
    return [sentence.strip() for sentence in nltk.sent_tokenize(text) if sentence.strip()]


def rank_sentences(sentences, model):
    """Score each sentence by cosine similarity to the document centroid.

    The centroid is the mean of all sentence embeddings, a simple
    representation of the whole document. Higher score means the sentence is
    more representative of the document. This is a centrality heuristic,
    not a supervised importance classifier.
    """
    embeddings = model.encode(sentences)
    centroid = embeddings.mean(axis=0).reshape(1, -1)
    scores = cosine_similarity(embeddings, centroid).flatten()
    return scores


def _word_count(text):
    """Count words as whitespace-separated tokens after trimming."""
    return len(text.split())


def summarize_to_ratio(text, ratio, model):
    """Build an extractive summary of about `ratio` of the original words.

    Selects complete high-scoring sentences within the word budget
    (floor(original_words * ratio)), then restores source order.
    Returns a dict with the summary and transparent metadata.
    """
    total_words = _word_count(text)
    percent = int(round(ratio * 100))
    result = {
        "ratio": ratio,
        "target_words": math.floor(total_words * ratio),
        "summary": "",
        "actual_words": 0,
        "actual_percent": 0.0,
        "num_source_sentences": 0,
        "num_sentences_selected": 0,
        "selected_indices": [],
        "budget_exceeded": False,
        "notes": [],
    }

    sentences = split_into_sentences(text)
    result["num_source_sentences"] = len(sentences)

    if not sentences:
        result["notes"].append("Input contains no sentences.")
        return result

    if len(sentences) == 1:
        # A single sentence cannot be shortened by selecting whole sentences.
        summary = sentences[0]
        result.update(
            {
                "summary": summary,
                "actual_words": _word_count(summary),
                "actual_percent": 100.0 if total_words else 0.0,
                "num_sentences_selected": 1,
                "selected_indices": [0],
            }
        )
        result["notes"].append(
            "The text contains a single sentence, so sentence-level extraction "
            "cannot shorten it. The original sentence is returned unchanged."
        )
        return result

    scores = rank_sentences(sentences, model)
    ranked_indices = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)

    budget = result["target_words"]
    chosen = []
    used_words = 0
    for index in ranked_indices:
        sentence_words = _word_count(sentences[index])
        if used_words + sentence_words <= budget:
            chosen.append(index)
            used_words += sentence_words

    if not chosen:
        # No complete sentence fits the budget: keep the best sentence and
        # clearly disclose that the target length was exceeded.
        chosen = [ranked_indices[0]]
        result["budget_exceeded"] = True
        result["notes"].append(
            f"No complete sentence fits inside the {percent}% word budget "
            f"({budget} words), so the highest-ranked sentence was included "
            "and the target length is exceeded."
        )

    chosen.sort()  # restore the original document order

    summary = " ".join(sentences[index] for index in chosen)
    result.update(
        {
            "summary": summary,
            "actual_words": _word_count(summary),
            "actual_percent": (
                round(100.0 * _word_count(summary) / total_words, 1) if total_words else 0.0
            ),
            "num_sentences_selected": len(chosen),
            "selected_indices": chosen,
        }
    )
    result["notes"].append(
        "Complete original sentences are kept, so the actual length is close "
        "to the requested percentage but cannot match it exactly."
    )
    return result
