"""SmartSum - Length-Controlled Text Summarizer (Streamlit user interface).

Collects the input article, generates 20%, 40%, and 60% extractive
summaries, and displays summaries, length/readability metrics, optional
ROUGE scores, and a manual factual-review checklist.
"""

import os

import pandas as pd
import streamlit as st

from evaluation import readability_score, rouge_scores, word_count
from summarizer import MODEL_NAME, load_model, split_into_sentences, summarize_to_ratio

RATIOS = [0.2, 0.4, 0.6]

SAMPLE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")


def _sample_articles():
    """Load the bundled sample articles and their reference summaries.

    Returns an ordered dict {label: {"text": ..., "reference": ...}} built
    from sample_data/articles/*.txt. The matching file in
    sample_data/references/ (same file name) is used as the reference
    summary; it is an empty string when no reference exists.
    """
    articles_dir = os.path.join(SAMPLE_DATA_DIR, "articles")
    references_dir = os.path.join(SAMPLE_DATA_DIR, "references")
    samples = {}
    if not os.path.isdir(articles_dir):
        return samples
    for file_name in sorted(os.listdir(articles_dir)):
        if not file_name.endswith(".txt"):
            continue
        stem = os.path.splitext(file_name)[0]
        # "01_education" -> "Education" (drop the ordering number).
        topic = stem.split("_", 1)[1] if "_" in stem else stem
        label = topic.replace("_", " ").capitalize()
        with open(os.path.join(articles_dir, file_name), encoding="utf-8") as handle:
            text = handle.read()
        reference = ""
        reference_path = os.path.join(references_dir, file_name)
        if os.path.isfile(reference_path):
            with open(reference_path, encoding="utf-8") as handle:
                reference = handle.read().strip()
        samples[label] = {"text": text, "reference": reference}
    return samples

st.set_page_config(page_title="SmartSum - Length-Controlled Text Summarizer", page_icon="📝")

st.title("SmartSum - Length-Controlled Text Summarizer")
st.write(
    "Creates extractive summaries at 20%, 40%, and 60% of the input length "
    "using a pretrained BERT sentence embedding model."
)
st.caption(f"Embedding model: {MODEL_NAME} (loaded once and cached; the first run downloads it)")

# ---------------- 1. Input ----------------
st.header("1. Provide your article")

# Optional: load one of the bundled sample articles (sample_data/articles).
samples = _sample_articles()
NO_SAMPLE = "(none - paste or upload your own article)"
sample_choice = st.selectbox("Try a sample article (optional)", [NO_SAMPLE] + list(samples))
if st.button("Load sample") and sample_choice != NO_SAMPLE:
    st.session_state["article_text"] = samples[sample_choice]["text"]
    st.session_state["reference_text"] = samples[sample_choice]["reference"]
    st.success(f"Loaded the '{sample_choice}' sample article (and its reference summary) below.")

uploaded_file = st.file_uploader("Upload a plain .txt file (optional)", type=["txt"])
if uploaded_file is not None:
    try:
        file_text = uploaded_file.read().decode("utf-8")
        st.session_state["article_text"] = file_text
        st.success(
            f"Loaded '{uploaded_file.name}' ({word_count(file_text)} words) "
            "into the text area below."
        )
    except (UnicodeDecodeError, ValueError):
        st.error(
            "Could not read the uploaded file. Please upload a plain .txt file "
            "saved with UTF-8 encoding."
        )

article_text = st.text_area("Paste your article", height=220, key="article_text")
reference_text = st.text_area(
    "Reference summary (optional - only needed to calculate ROUGE scores)",
    height=120,
    key="reference_text",
)
generate = st.button("Generate summaries", type="primary")

if generate:
    text = (article_text or "").strip()

    if not text:
        st.warning("The input is empty. Paste an article or upload a .txt file, then try again.")
        st.stop()

    total_words = word_count(text)
    if total_words < 30:
        st.warning(
            f"The article is very short ({total_words} words). Results are shown, "
            "but summaries work best on longer articles (100+ words)."
        )

    # Load the model with an actionable message instead of a traceback.
    try:
        model = load_model()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    try:
        sentences = split_into_sentences(text)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    if len(sentences) < 2:
        st.info(
            "The text contains only one sentence. Sentence-level extraction "
            "cannot shorten a single sentence, so the original sentence is "
            "returned unchanged."
        )

    st.caption(f"Original article: {total_words} words, {len(sentences)} sentences.")

    results = [summarize_to_ratio(text, ratio, model) for ratio in RATIOS]

    # ---------------- 2. Summaries ----------------
    st.header("2. Summaries")
    for res in results:
        percent = int(res["ratio"] * 100)
        title = (
            f"{percent}% summary - target {res['target_words']} words | "
            f"actual {res['actual_words']} words ({res['actual_percent']}% of original)"
        )
        with st.expander(title, expanded=(percent == 40)):
            st.write(res["summary"])
            for note in res["notes"]:
                st.caption(note)

    # ---------------- 3. Comparison ----------------
    st.header("3. Length and readability comparison")
    rows = []
    for res in results:
        if res["summary"]:
            score, label = readability_score(res["summary"])
            readability = f"{score:.1f} ({label})"
        else:
            readability = "n/a"
        rows.append(
            {
                "Summary level": f"{int(res['ratio'] * 100)}%",
                "Target words": res["target_words"],
                "Actual words": res["actual_words"],
                "Achieved % of original": f"{res['actual_percent']}%",
                "Sentences selected": res["num_sentences_selected"],
                "Flesch Reading Ease": readability,
            }
        )
    st.table(pd.DataFrame(rows))
    st.caption(
        "Actual lengths are approximate because complete original sentences are "
        "selected; sentence boundaries rarely match the exact word budget."
    )

    # ---------------- 4. ROUGE (only with a reference) ----------------
    st.header("4. ROUGE scores")
    if reference_text and reference_text.strip():
        st.write(
            "ROUGE-1/ROUGE-2/ROUGE-L compare each generated summary with the "
            "supplied reference using word overlap (higher is better). These are "
            "overlap-based measures, not proof of factual accuracy or complete "
            "information preservation."
        )
        rouge_rows = []
        for res in results:
            scores = rouge_scores(res["summary"], reference_text)
            rouge_rows.append({"Summary level": f"{int(res['ratio'] * 100)}%", **scores})
        st.table(pd.DataFrame(rouge_rows))
    else:
        st.info("Reference summary not provided - ROUGE not calculated.")

    # ---------------- 5. Manual factual review ----------------
    st.header("5. Factual review (manual)")
    st.write(
        "Automatic scores cannot prove factual accuracy. Manually check each "
        "summary against the source article:"
    )
    st.markdown(
        "- [ ] Names are unchanged\n"
        "- [ ] Dates are unchanged\n"
        "- [ ] Numbers and units are unchanged\n"
        "- [ ] Key claims keep their original meaning\n"
        "- [ ] Negations and qualifiers (not, only, may) are preserved\n"
        "- [ ] No sentence lost context that changes its meaning"
    )
    st.warning("Manual review required - record reviewer findings in the project report.")
