# -*- coding: utf-8 -*-
"""
Pipeline: corpus preprocessing for LDA.
Input: Scopus export CSV (e.g. raw_corpus.csv) on AI ethics.
Usage: python -m bertlda.preprocess_corpus --preprocess-corpus filename.csv
"""
import argparse
import json
import os
import pickle
import re
from collections import Counter
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

import nltk

COL_TITLE = "Title"
COL_ABSTRACT = "Abstract"
COL_YEAR = "Year"

VALID_ACRONYMS = {
    "ai", "ml", "nlp", "llm", "llms", "gpt", "genai", "gdpr", "eu ai act",
    "hipaa", "ccpa", "pipa", "api", "ux", "hci", "ip", "eu", "un", "usa",
}

QUERY_TERMS = {
    "ai", "artificial", "intelligence", "generative", "genai", "llm", "llms",
    "gpt", "foundation", "models", "large", "language", "chatbot", "chatbots",
    "text", "generation", "governance", "ai governance",
    "ethic", "ethics", "ethical", "ai ethics",
    "regulation", "regulatory", "ai act", "eu ai act", "gdpr",
    "responsible", "responsible ai", "accountability", "accountable",
    "policy", "algorithmic",
    "bias", "fairness", "equity", "equitable",
    "transparency", "transparent", "explainability", "explainable", "xai",
    "privacy", "confidential", "safety", "safe ai", "secure",
    "trust", "trustworthy", "trustworthy ai",
    "human-centered", "human in the loop", "human oversight",
    "autonomy", "dignity", "justice", "inclusion", "inclusive",
    "discrimination", "harm", "risk", "impact",
    "social", "societal", "human rights", "moral", "values",
    "audit", "oversight",
}

METHOD_TERMS = {
    "study", "studies", "result", "results", "analysis", "analyses",
    "method", "methods", "approach", "approaches", "paper", "research",
    "article", "finding", "findings", "show", "shows", "showed",
    "propose", "proposed", "discuss", "discussed", "present", "presented",
    "based", "use", "used", "using", "also", "however", "thus",
}

DF_THRESHOLD = 3.0
TOP_CANDIDATES = 60

EXPORT_COLS = [
    COL_TITLE, COL_ABSTRACT, "Authors", COL_YEAR, "Source title",
    "Author Keywords", "Index Keywords", "Cited by", "DOI", "Publisher",
    "Open Access", "Language of Original Document", "Link", "Document Type",
]

RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_EMAIL = re.compile(r"\S+@\S+")
RE_CITATION = re.compile(r"\[\s*\d+(?:\s*[,-]\s*\d+)*\s*\]")
RE_COPYRIGHT = re.compile(
    r"(?:\(\s*c\s*\)|©)\s*\d{4}[^.]*?(?:\.|$)"
    r"|all\s+rights?\s+reserved[^.]*?(?:\.|$)"
    r"|published\s+by\s+(?:elsevier|springer|wiley|taylor|informa|mdpi|"
    r"oxford|cambridge|sage|emerald|frontiers)[^.]*?(?:\.|$)"
    r"|this\s+is\s+an\s+open\s+access\s+article[^.]*?(?:\.|$)",
    re.IGNORECASE,
)
RE_NO_ALNUM = re.compile(r"[^a-z0-9\-\s]")
RE_ONLY_DIGIT = re.compile(r"^[\d\-]+$")
RE_EXTRA_DASH = re.compile(r"(^-+)|(-+$)")


def setup_plotting():
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["font.size"] = 11
    plt.rcParams["figure.facecolor"] = "#EFEBE4"
    plt.rcParams["axes.facecolor"] = "#EFEBE4"


def download_nltk():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)


def load_and_clean(input_file):
    df_raw = pd.read_csv(input_file)
    print(f"Documents loaded (raw corpus): {len(df_raw):,}")

    df = df_raw.copy()
    df[COL_ABSTRACT] = df[COL_ABSTRACT].replace(
        r"^\[No abstract available\]$", np.nan, regex=True
    )
    n_before_na = len(df)
    df = df.dropna(subset=[COL_ABSTRACT]).reset_index(drop=True)
    n_without_abstract = n_before_na - len(df)
    print(f"Documents without abstract (removed): {n_without_abstract}")
    print(f"Documents with abstract: {len(df):,}")

    df["text"] = df[COL_TITLE].astype(str) + ". " + df[COL_ABSTRACT].astype(str)
    return df, df_raw, n_without_abstract


def lemmatize(word, lemmatizer):
    n = lemmatizer.lemmatize(word, pos="n")
    if n != word:
        return n
    return lemmatizer.lemmatize(word, pos="v")


def clean_for_lda(text, stopwords, lemmatizer, min_len=3):
    t = str(text).lower()
    t = RE_URL.sub(" ", t)
    t = RE_EMAIL.sub(" ", t)
    t = RE_CITATION.sub(" ", t)
    t = RE_COPYRIGHT.sub(" ", t)
    t = RE_NO_ALNUM.sub(" ", t)

    output = []
    for w in t.split():
        w = RE_EXTRA_DASH.sub("", w)
        if not w or RE_ONLY_DIGIT.match(w):
            continue
        lemma = lemmatize(w, lemmatizer)
        if lemma in stopwords:
            continue
        if len(lemma) < min_len and lemma not in VALID_ACRONYMS:
            continue
        output.append(lemma)
    return output


def classify_term(w):
    if w in QUERY_TERMS:
        return "QUERY"
    if w in METHOD_TERMS:
        return "METHODOLOGICAL"
    return "GENERAL CONTENT"


def decide(row):
    if row["type"] in ("QUERY", "METHODOLOGICAL") and row["DF_pct"] >= DF_THRESHOLD:
        return "REMOVE"
    return "KEEP"


def justify(row, query_terms, method_terms):
    term = row["term"]
    if row["decision"] == "REMOVE":
        if term in query_terms:
            return ("Search-query root term. Present by design; retaining it would "
                    "overshadow thematic findings (algorithmic noise).")
        elif term in method_terms:
            return ("Generic academic boilerplate. Does not help discriminate "
                    "underlying topics in AI ethics.")
        else:
            return ("High document frequency (DF%) without thematic discrimination.")
    else:
        if row["type"] == "GENERAL CONTENT":
            return ("Content term with moderate/low DF%. Kept because it adds "
                    "valuable thematic nuance.")
        else:
            return ("Kept because it does not exceed the empirical noise threshold.")


def build_stopword_candidates(docs_tokensensens, term_freq, n_docs):
    doc_freq = Counter()
    for d in docs_tokensensens:
        for w in set(d):
            doc_freq[w] += 1

    candidates = []
    for w, dfreq in doc_freq.most_common(TOP_CANDIDATES):
        candidates.append({
            "term": w,
            "DF_abs": dfreq,
            "DF_pct": round(100 * dfreq / n_docs, 2),
            "TF_total": term_freq[w],
        })
    return pd.DataFrame(candidates)


def build_domain_stopwords(candidates_df, stopwords_func):
    candidates_df = candidates_df.copy()
    candidates_df["type"] = candidates_df["term"].apply(classify_term)
    candidates_df["decision"] = candidates_df.apply(decide, axis=1)
    candidates_df["justification"] = candidates_df.apply(
        justify, axis=1, query_terms=QUERY_TERMS, method_terms=METHOD_TERMS
    )

    domain_stopwords = set(candidates_df.loc[
        candidates_df["decision"] == "REMOVE", "term",
    ])
    stopwords_final = stopwords_func | domain_stopwords
    return candidates_df, stopwords_final


def save_stopword_artifacts(candidates_df, stopwords_func, stopwords_final, out_dir):
    decisions = candidates_df[[
        "term", "DF_pct", "decision", "type", "justification",
    ]].rename(columns={
        "term": "Term", "DF_pct": "DF %", "decision": "Decision",
        "type": "Type", "justification": "Justification",
    })
    decisions.to_excel(os.path.join(out_dir, "stopword_decisions.xlsx"), index=False)

    layer2 = pd.DataFrame({
        "term": sorted(candidates_df[
            (candidates_df["decision"] == "REMOVE") &
            (candidates_df["type"] == "QUERY")
        ]["term"]),
        "layer": "Layer 2 - Search-query terms",
    })
    layer3 = pd.DataFrame({
        "term": sorted(candidates_df[
            (candidates_df["decision"] == "REMOVE") &
            (candidates_df["type"] == "METHODOLOGICAL")
        ]["term"]),
        "layer": "Layer 3 - Methodological boilerplate",
    })

    stopwords_3layers = pd.concat([
        pd.DataFrame({
            "term": sorted(stopwords_func),
            "layer": "Layer 1 - Functional (NLTK)",
        }),
        layer2, layer3,
    ], ignore_index=True)
    stopwords_3layers.to_excel(os.path.join(out_dir, "stopwords_3layers.xlsx"), index=False)
    return decisions


def generate_wordcloud(freq, out_path, title):
    wc = WordCloud(
        width=1600, height=800, background_color="#EFEBE4",
        colormap="viridis", max_words=150, collocations=False, random_state=42,
    ).generate_from_frequencies(freq)
    plt.figure(figsize=(14, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=15, pad=15)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def save_comparison_table(docs_tokensensens, term_freq, lengths, final_docs,
                          term_freq_final, lens_final, out_dir):
    comparison_table = pd.DataFrame({
        "Metric": [
            "Documents", "Vocabulary (unique terms)", "Total tokens",
            "Average tokens per document", "Empty documents",
        ],
        "BEFORE (layer 1)": [
            len(docs_tokensensens), len(term_freq), int(lengths.sum()),
            round(lengths.mean(), 2), int((lengths == 0).sum()),
        ],
        "AFTER (3 layers)": [
            len(final_docs), len(term_freq_final), int(lens_final.sum()),
            round(lens_final.mean(), 2), int((lens_final == 0).sum()),
        ],
    })
    comparison_table.to_excel(
        os.path.join(out_dir, "comparison_before_after.xlsx"), index=False
    )
    return comparison_table


def export_corpora(df, final_docs, lens_final, out_dir):
    export_cols = [c for c in EXPORT_COLS if c in df.columns]

    df_raw_export = df[export_cols].copy()
    df_raw_export.to_csv(
        os.path.join(out_dir, "corpus_raw.csv"), index=False, encoding="utf-8-sig"
    )

    prep = df[export_cols].copy()
    prep["tokens"] = [" ".join(d) for d in final_docs]
    prep["n_tokens"] = lens_final.values
    prep = prep[prep["n_tokens"] > 0]
    prep.to_csv(
        os.path.join(out_dir, "corpus_preprocessed.csv"),
        index=False, encoding="utf-8-sig"
    )
    return prep


def preprocess_corpus(input_file, out_dir="./outputs"):
    os.makedirs(out_dir, exist_ok=True)
    setup_plotting()
    download_nltk()

    lemmatizer = WordNetLemmatizer()
    stopwords_func = set(nltk_stopwords.words("english"))
    print(f"Functional stopwords (layer 1, NLTK): {len(stopwords_func)}")

    df, df_raw, n_without_abstract = load_and_clean(input_file)

    # First pass: functional stopwords only
    t0 = perf_counter()
    docs_tokensensens = [clean_for_lda(d, stopwords_func, lemmatizer) for d in df["text"]]
    print(f"Processed {len(docs_tokensensens):,} documents in {perf_counter() - t0:.1f}s")

    lengths = pd.Series([len(d) for d in docs_tokensensens])
    print("Document length (tokens) after initial cleaning:")
    print(lengths.describe())
    n_empty = (lengths == 0).sum()
    print(f"Empty documents after layer 1: {n_empty}")

    empty_examples = []
    if n_empty:
        for j in lengths[lengths == 0].index[:5]:
            empty_examples.append({
                "index": int(j),
                "title": df.loc[j, COL_TITLE],
                "abstract_original": str(df.loc[j, COL_ABSTRACT])[:300],
            })
            print(f"\n  [{j}] {df.loc[j, 'text'][:200]}")

    term_freq = Counter(w for d in docs_tokensensens for w in d)
    print(f"\nVocabulary (layer 1): {len(term_freq):,} unique terms")
    print("\nTop 40 (before domain stopwords):")
    for w, c in term_freq.most_common(40):
        print(f"  {w:<22} {c:>7,}")

    generate_wordcloud(
        term_freq, os.path.join(out_dir, "wordcloud_before.png"),
        "Full corpus -- BEFORE (functional stopwords only)",
    )
    print("Saved: wordcloud_before.png")

    # Stopword candidates (layers 2 and 3)
    n_docs = len(docs_tokensensens)
    candidates_df = build_stopword_candidates(docs_tokensensens, term_freq, n_docs)
    print("\nTop 60 candidate stopwords (by DF%):")
    print(candidates_df.head(60).to_string(index=False))

    candidates_df, stopwords_final = build_domain_stopwords(
        candidates_df, stopwords_func
    )
    save_stopword_artifacts(candidates_df, stopwords_func, stopwords_final, out_dir)
    print("\nSaved: stopword_decisions.xlsx")
    print("Saved: stopwords_3layers.xlsx")

    # Second pass: final stopwords (3 layers)
    t0 = perf_counter()
    final_docs = [clean_for_lda(d, stopwords_final, lemmatizer) for d in df["text"]]
    print(f"\nReprocessed with 3-layer stopwords in {perf_counter() - t0:.1f}s")

    lens_final = pd.Series([len(d) for d in final_docs])
    term_freq_final = Counter(w for d in final_docs for w in d)

    comparison_table = save_comparison_table(
        docs_tokensensens, term_freq, lengths, final_docs,
        term_freq_final, lens_final, out_dir,
    )
    print("\nComparison table BEFORE vs AFTER:")
    print(comparison_table.to_string(index=False))
    print("Saved: comparison_before_after.xlsx")

    generate_wordcloud(
        term_freq_final, os.path.join(out_dir, "wordcloud_after.png"),
        "Full corpus -- AFTER (3-layer stopwords)",
    )
    print("Saved: wordcloud_after.png")

    wc_before = WordCloud(
        width=1600, height=800, background_color="#EFEBE4",
        colormap="viridis", max_words=150, collocations=False, random_state=42,
    ).generate_from_frequencies(term_freq)
    wc_after = WordCloud(
        width=1600, height=800, background_color="#EFEBE4",
        colormap="viridis", max_words=150, collocations=False, random_state=42,
    ).generate_from_frequencies(term_freq_final)

    fig, ax = plt.subplots(2, 1, figsize=(14, 12))
    ax[0].imshow(wc_before, interpolation="bilinear")
    ax[0].set_title("BEFORE -- functional stopwords only", fontsize=14)
    ax[0].axis("off")
    ax[1].imshow(wc_after, interpolation="bilinear")
    ax[1].set_title("AFTER -- three stopword layers", fontsize=14)
    ax[1].axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "wordclouds_before_after_paired.png"),
        bbox_inches="tight",
    )
    plt.close()
    print("Saved: wordclouds_before_after_paired.png")

    prep = export_corpora(df, final_docs, lens_final, out_dir)
    print(f"\nSaved: corpus_raw.csv ({len(df):,} documents)")
    print(f"Saved: corpus_preprocessed.csv ({len(prep):,} documents)")

    candidates_df.to_excel(
        os.path.join(out_dir, "tf_df_frequencies.xlsx"), index=False
    )
    print("Saved: tf_df_frequencies.xlsx")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Raw corpus (Scopus export):           {len(df_raw):,}")
    print(f"Without abstract (removed):           {n_without_abstract}")
    print(f"Corpus with abstract (df):            {len(df):,}")
    print(f"Empty documents after layer 1:        {n_empty}")
    print(f"Empty documents after 3 layers:       {int((lens_final == 0).sum())}")
    print(f"Final preprocessed corpus (>0 tok):   {len(prep):,}")
    print(f"Vocabulary BEFORE:                    {len(term_freq):,}")
    print(f"Vocabulary AFTER:                     {len(term_freq_final):,}")

    # Checkpoint
    with open(os.path.join(out_dir, "checkpoint_S03.pkl"), "wb") as f:
        pickle.dump({
            "df": df,
            "final_docs": final_docs,
            "stopwords_final": stopwords_final,
        }, f)

    with open(os.path.join(out_dir, "empty_examples.json"), "w",
              encoding="utf-8") as f:
        json.dump(empty_examples, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    return {
        "df": df,
        "df_raw": df_raw,
        "final_docs": final_docs,
        "stopwords_final": stopwords_final,
        "candidates_df": candidates_df,
        "prep": prep,
    }


def main():
    parser = argparse.ArgumentParser(
        description="bertlda: corpus preprocessing for LDA."
    )
    parser.add_argument(
        "--preprocess-corpus", metavar="FILENAME.csv",
        help="Path to Scopus export CSV to preprocess.",
    )
    parser.add_argument(
        "--out-dir", default="./outputs",
        help="Output directory (default: ./outputs).",
    )
    args = parser.parse_args()

    if args.preprocess_corpus:
        preprocess_corpus(args.preprocess_corpus, args.out_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
