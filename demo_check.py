"""One-off end-to-end demo: summarizes every sample article and prints a table.

Not part of the app; used to verify and record example results.
Run from the project folder:  python demo_check.py
"""

import os

from evaluation import readability_score, rouge_scores, word_count
from summarizer import load_model, summarize_to_ratio

ARTICLES_DIR = os.path.join("sample_data", "articles")
REFERENCES_DIR = os.path.join("sample_data", "references")

model = load_model()

names = sorted(f for f in os.listdir(ARTICLES_DIR) if f.endswith(".txt"))
rows = []
rouge_totals = {"ROUGE-1": 0.0, "ROUGE-2": 0.0, "ROUGE-L": 0.0}
with_reference = 0
for name in names:
    with open(os.path.join(ARTICLES_DIR, name), encoding="utf-8") as handle:
        article = handle.read()
    reference = ""
    reference_path = os.path.join(REFERENCES_DIR, name)
    if os.path.isfile(reference_path):
        with open(reference_path, encoding="utf-8") as handle:
            reference = handle.read().strip()

    result = summarize_to_ratio(article, 0.4, model)
    score, _label = readability_score(result["summary"])
    rouge = rouge_scores(result["summary"], reference) if reference else None
    if rouge:
        with_reference += 1
        for key in rouge_totals:
            rouge_totals[key] += rouge[key]
    rows.append((name, word_count(article), result["actual_percent"], score, rouge))

print(f"{'article':<22}{'words':>6}{'40% level':>12}{'Flesch':>8}   ROUGE-1 / ROUGE-2 / ROUGE-L")
print("-" * 96)
for name, words, percent, score, rouge in rows:
    rouge_txt = "n/a" if not rouge else (
        f"{rouge['ROUGE-1']:>5.1f} / {rouge['ROUGE-2']:>5.1f} / {rouge['ROUGE-L']:>5.1f}"
    )
    print(f"{name:<22}{words:>6}{percent:>11}%{score:>8.1f}   {rouge_txt}")

if with_reference:
    print("-" * 96)
    print(f"Averages over {with_reference} articles at the 40% level:")
    print(
        f"  ROUGE-1 {rouge_totals['ROUGE-1'] / with_reference:.1f}  "
        f"ROUGE-2 {rouge_totals['ROUGE-2'] / with_reference:.1f}  "
        f"ROUGE-L {rouge_totals['ROUGE-L'] / with_reference:.1f}"
    )

# Show one full example (the first article) at all three levels.
first = names[0]
with open(os.path.join(ARTICLES_DIR, first), encoding="utf-8") as handle:
    article = handle.read()
print("=" * 96)
print(f"Full example: {first} ({word_count(article)} words)")
for ratio in (0.2, 0.4, 0.6):
    res = summarize_to_ratio(article, ratio, model)
    print(
        f"\n{int(ratio * 100)}% summary -> target {res['target_words']} words, "
        f"actual {res['actual_words']} words ({res['actual_percent']}%):"
    )
    print(res["summary"])

