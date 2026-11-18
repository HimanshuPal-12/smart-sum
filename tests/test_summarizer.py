"""Basic acceptance tests for the summarizer and evaluation helpers.

Covers the core checks from the project test plan (T1-T10 where they can be
tested without the UI). Run from the project folder with:

    python tests\\test_summarizer.py

It also works with pytest if pytest is installed. The first run downloads
the BERT sentence model (about 450 MB) and needs internet access.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from evaluation import readability_score, rouge_scores, word_count  # noqa: E402
from summarizer import (  # noqa: E402
    load_model,
    rank_sentences,
    split_into_sentences,
    summarize_to_ratio,
)

SAMPLE_ARTICLE_PATH = os.path.join(PROJECT_ROOT, "sample_data", "sample_article.txt")

# A short text where no complete sentence can fit a 20% word budget
# (2 sentences, 35 words total -> 20% budget = 7 words).
SHORT_BUDGET_TEXT = (
    "The small research team at the university spent three years studying the "
    "migration patterns of birds across two different continents. Their detailed "
    "final report was published in a well-known international journal and "
    "received considerable attention."
)

ONE_SENTENCE_TEXT = "This single sentence is the entire document."


def _sample_article():
    with open(SAMPLE_ARTICLE_PATH, encoding="utf-8") as handle:
        return handle.read()


# ---------- evaluation.word_count (consistent rule) ----------
def test_word_count_rule():
    assert word_count("") == 0
    assert word_count("   ") == 0
    assert word_count("  One two three.  ") == 3


# ---------- T7: sample .txt files are readable UTF-8 ----------
def test_t7_sample_txt_files_readable():
    article = _sample_article()
    assert word_count(article) > 100
    with open(
        os.path.join(PROJECT_ROOT, "sample_data", "reference_summary.txt"),
        encoding="utf-8",
    ) as handle:
        assert len(handle.read().split()) > 10


# ---------- T1: normal article -> non-empty summaries of source sentences ----------
def test_t1_normal_article():
    model = load_model()
    text = _sample_article()
    source_sentences = set(split_into_sentences(text))
    for ratio in (0.2, 0.4, 0.6):
        result = summarize_to_ratio(text, ratio, model)
        assert result["summary"], f"empty summary for ratio {ratio}"
        for sentence in split_into_sentences(result["summary"]):
            assert sentence in source_sentences, (
                f"summary contains a sentence that is not in the source: {sentence!r}"
            )


# ---------- T2: target budgets increase; actual lengths grow ----------
def test_t2_length_levels():
    model = load_model()
    text = _sample_article()
    r20 = summarize_to_ratio(text, 0.2, model)
    r40 = summarize_to_ratio(text, 0.4, model)
    r60 = summarize_to_ratio(text, 0.6, model)
    assert r20["target_words"] < r40["target_words"] < r60["target_words"]
    assert r20["actual_words"] <= r40["actual_words"] <= r60["actual_words"]


# ---------- T3: output sentences keep their original order ----------
def test_t3_sentence_order():
    model = load_model()
    text = _sample_article()
    sentences = split_into_sentences(text)
    result = summarize_to_ratio(text, 0.2, model)

    selected = result["selected_indices"]
    assert selected == sorted(selected), "selected sentences are not in source order"

    # The ranking order itself must differ from source order, so ordering
    # restoration is actually being exercised.
    scores = rank_sentences(sentences, model)
    ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
    assert ranked[: len(selected)] != selected, "ranking order equals source order"


# ---------- T4: empty input -> friendly handling, no crash ----------
def test_t4_empty_input():
    model = load_model()
    result = summarize_to_ratio("   \n  ", 0.2, model)
    assert result["summary"] == ""
    assert result["notes"], "expected an informative note for empty input"


# ---------- T5: one sentence -> return it with an explanation ----------
def test_t5_one_sentence():
    model = load_model()
    result = summarize_to_ratio(ONE_SENTENCE_TEXT, 0.2, model)
    assert result["summary"] == ONE_SENTENCE_TEXT
    assert result["actual_percent"] == 100.0
    assert any("single sentence" in note for note in result["notes"])


# ---------- T6: budget too small -> top sentence, target exceeded ----------
def test_t6_very_short_budget():
    model = load_model()
    result = summarize_to_ratio(SHORT_BUDGET_TEXT, 0.2, model)
    assert result["summary"], "summary must never be silently empty"
    assert result["budget_exceeded"] is True
    assert result["actual_words"] > result["target_words"]
    # The returned sentence must be one of the original sentences.
    assert result["summary"] in SHORT_BUDGET_TEXT


# ---------- T8: ROUGE unavailable without a reference ----------
def test_t8_rouge_without_reference():
    assert rouge_scores("The cat sat on the mat.", "") is None
    assert rouge_scores("The cat sat on the mat.", "   ") is None
    assert rouge_scores("The cat sat on the mat.", None) is None


# ---------- T9: ROUGE numbers with a real reference ----------
def test_t9_rouge_with_reference():
    scores = rouge_scores("The cat sat on the mat.", "A cat sat on the mat.")
    assert set(scores) == {"ROUGE-1", "ROUGE-2", "ROUGE-L"}
    for value in scores.values():
        assert 0.0 <= value <= 100.0


# ---------- readability returns a score and label ----------
def test_readability_score():
    score, label = readability_score("The cat sat on the mat. It was happy.")
    assert isinstance(score, float)
    assert isinstance(label, str) and label


# ---------- T10: model-loading failure gives an actionable message ----------
def test_t10_model_error_is_friendly():
    import summarizer

    original_class = summarizer.SentenceTransformer
    original_model = summarizer._MODEL
    summarizer.SentenceTransformer = None  # force any load attempt to fail
    summarizer._MODEL = None
    try:
        try:
            summarizer.load_model()
        except RuntimeError as exc:
            message = str(exc)
            assert "Could not load" in message
            assert "internet" in message or "download" in message
        else:
            raise AssertionError("expected a RuntimeError with a friendly message")
    finally:
        summarizer.SentenceTransformer = original_class
        summarizer._MODEL = original_model


# ---------- consistent metadata ----------
def test_metadata_matches_summary():
    model = load_model()
    result = summarize_to_ratio(_sample_article(), 0.4, model)
    assert result["actual_words"] == word_count(result["summary"])
    assert result["num_sentences_selected"] == len(result["selected_indices"])


if __name__ == "__main__":
    print("Loading model (first run downloads it; this can take a few minutes)...")
    test_names = sorted(
        name for name in globals() if name.startswith("test_") and callable(globals()[name])
    )
    failures = 0
    for name in test_names:
        try:
            globals()[name]()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    total = len(test_names)
    print(f"\n{total - failures}/{total} tests passed.")
    sys.exit(1 if failures else 0)
