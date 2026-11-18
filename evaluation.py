"""Simple evaluation helpers: word counts, readability, optional ROUGE.

The same word-count rule is used everywhere in the project: split on
whitespace after trimming, with empty text returning zero. Punctuation
stays attached to words and is not counted separately.
"""

import textstat

_READABILITY_LABELS = [
    (90, "Very easy"),
    (80, "Easy"),
    (70, "Fairly easy"),
    (60, "Plain English"),
    (50, "Fairly difficult"),
    (30, "Difficult"),
    (0, "Very difficult"),
]


def word_count(text):
    """Return the number of whitespace-separated words in `text`."""
    return len(text.split())


def readability_score(text):
    """Return (Flesch Reading Ease score, plain-language label).

    Higher scores mean easier reading. This estimates linguistic
    difficulty only; it says nothing about factual correctness.
    """
    score = textstat.flesch_reading_ease(text)
    for threshold, label in _READABILITY_LABELS:
        if score >= threshold:
            return score, label
    return score, "Confusing"


def rouge_scores(prediction, reference):
    """Return ROUGE-1/ROUGE-2/ROUGE-L F1 scores for prediction vs reference.

    Returns None when no reference summary is provided. ROUGE measures word
    and sequence overlap with a reference summary; it is not proof of
    factual accuracy or complete information preservation.
    """
    if not reference or not reference.strip():
        return None

    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        "ROUGE-1": round(scores["rouge1"].fmeasure * 100, 1),
        "ROUGE-2": round(scores["rouge2"].fmeasure * 100, 1),
        "ROUGE-L": round(scores["rougeL"].fmeasure * 100, 1),
    }
