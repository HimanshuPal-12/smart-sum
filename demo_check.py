"""One-off end-to-end demo: prints the three summaries and metrics.

Not part of the app; used to verify and record example results.
Run from the project folder:  python demo_check.py
"""

from evaluation import readability_score, rouge_scores, word_count
from summarizer import load_model, summarize_to_ratio

with open("sample_data/sample_article.txt", encoding="utf-8") as handle:
    article = handle.read()
with open("sample_data/reference_summary.txt", encoding="utf-8") as handle:
    reference = handle.read().strip()

model = load_model()
print(f"ORIGINAL: {word_count(article)} words")
print("=" * 70)

results = [summarize_to_ratio(article, ratio, model) for ratio in (0.2, 0.4, 0.6)]
for res in results:
    percent = int(res["ratio"] * 100)
    score, label = readability_score(res["summary"])
    rouge = rouge_scores(res["summary"], reference)
    print(
        f"\n{percent}% summary -> target {res['target_words']} words, "
        f"actual {res['actual_words']} words ({res['actual_percent']}%), "
        f"{res['num_sentences_selected']} sentences, "
        f"Flesch {score:.1f} ({label})"
    )
    if rouge:
        print(
            f"ROUGE-1 {rouge['ROUGE-1']}  ROUGE-2 {rouge['ROUGE-2']}  "
            f"ROUGE-L {rouge['ROUGE-L']}"
        )
    print("-" * 70)
    print(res["summary"][:400])
