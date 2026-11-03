# -*- coding: utf-8 -*-
"""
fase1a_lda.py
=============
Phase 1a - LDA diagnosis and K search.

Reads:   ck_corpus_base.pkl
Writes:  ck_corpus_lda.pkl (docs_tokens, dictionary, corpus, dict_eval, n_docs)
         outputs/results/lda/lda_k_search.csv
"""
from __future__ import annotations

import os
import sys
from time import perf_counter

import numpy as np
import pandas as pd

try:
    _DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _DIR = os.getcwd()
sys.path.insert(0, _DIR)

from config import CFG, RUTAS, RESULTS_DIR
from pipeline_base import (
    asegurar_carpetas, cargar_checkpoint, guardar_checkpoint,
    guardar_csv, checkpoint_existe, _fmt, print_separador,
)
from text_processing import build_stopwords, clean_for_lda, get_params


def _fmt_m(m, s):
    return f"{int(m):3d}m {s:05.2f}s"


def lda_metrics(lda, corpus, docs_tokens, dict_eval, k, top_n=10):
    from gensim.models.coherencemodel import CoherenceModel

    topics_words = [[w for w, _ in lda.show_topic(i, topn=top_n)] for i in range(k)]
    m = {}
    for clave, medida in [("cv", "c_v"), ("npmi", "c_npmi")]:
        try:
            m[clave] = round(float(CoherenceModel(
                topics=topics_words, texts=docs_tokens, dictionary=dict_eval,
                coherence=medida, processes=1).get_coherence()), 4)
        except Exception:
            m[clave] = 0.0
    plano = [w for grupo in topics_words for w in grupo]
    m["diversity"] = round(len(set(plano)) / max(len(plano), 1), 4)
    m["perplexity"] = round(float(lda.log_perplexity(corpus)), 4)
    dominant = [int(np.argmax([p for _, p in
                               lda.get_document_topics(d, minimum_probability=0)]))
                for d in corpus]
    tam = np.array([dominant.count(i) for i in range(k)], dtype=float)
    tam = tam[tam > 0]
    m["dominance"] = round(float(tam.max() / tam.sum()), 4) if len(tam) else 1.0
    p = tam / tam.sum() if len(tam) else np.array([1.0])
    m["entropy_norm"] = round(float(-(p * np.log(p + 1e-12)).sum()
                                     / np.log(max(k, 2))), 4)
    m["topics_empty"] = int(k - len(tam))
    m["K"] = k
    return m


def main() -> dict:
    asegurar_carpetas()
    print_separador("PHASE 1a - LDA DIAGNOSIS AND K SEARCH")
    t_total = perf_counter()

    from gensim.models.ldamodel import LdaModel
    from gensim.models.ldamulticore import LdaMulticore
    from gensim.corpora import Dictionary

    base = cargar_checkpoint("ck_corpus_base")
    df = base["df"].reset_index(drop=True)
    P = base["params"]
    n_docs = len(df)
    print(f"  {n_docs:,} documents [{P['escala']}]")
    print(f"  K grid: {P['k_min']}-{P['k_max']} step {P['k_step']}")

    # Tokenization (with checkpoint reuse)
    ck_path = "ck_corpus_lda"
    docs_tokens = None
    if checkpoint_existe(ck_path):
        d = cargar_checkpoint(ck_path)
        if d.get("n_docs") == n_docs:
            docs_tokens = (d.get("docs_tokens") or d.get("docs_tok")
                           or d.get("docs_tokensens"))
            id2word = d["id2word"]
            corpus = d["corpus"]
            dict_eval = d["dict_eval"]
            print("  LDA checkpoint recovered.")

    if docs_tokens is None:
        stops = build_stopwords()
        print(f"  Stopwords: {len(stops):,}")
        docs_raw = df["texto"].tolist()
        from joblib import Parallel, delayed

        # parallel tokenization
        def _tok(d):
            return clean_for_lda(d, stops)
        NC = 8
        ncfg = CFG["compute"]["n_cores"]
        import multiprocessing
        if ncfg is None:
            ncfg = max(1, multiprocessing.cpu_count() - 1)
        from joblib import Parallel, delayed
        docs_tokens = Parallel(n_jobs=ncfg, backend="loky")(
            delayed(_tok)(d) for d in docs_raw
        )

        print("  Detecting bigrams...")
        from gensim.models.phrases import Phrases, Phraser
        bigram = Phrases(docs_tokens, min_count=P["bigram_min_count"],
                        threshold=P["bigram_threshold"])
        bigram_phraser = Phraser(bigram)
        docs_tokens = [bigram_phraser[d] for d in docs_tokens]

        id2word = Dictionary(docs_tokens)
        id2word.filter_extremes(no_below=P["no_below"], no_above=P["no_above"])
        corpus = [id2word.doc2bow(d) for d in docs_tokens]

        dict_eval = Dictionary(docs_tokens)
        dict_eval.filter_extremes(no_below=2, no_above=0.9)

        guardar_checkpoint({
            "docs_tokens": docs_tokens, "id2word": id2word, "corpus": corpus,
            "dict_eval": dict_eval, "n_docs": n_docs,
        }, ck_path)

    print(f"  Train vocab: {len(id2word):,} (no_below={P['no_below']}, no_above={P['no_above']})")
    print(f"  Eval vocab : {len(dict_eval):,}")
    vacios = sum(1 for c in corpus if len(c) == 0)
    if vacios:
        print(f"  WARNING: {vacios} docs with empty BoW after filtering.")
    # keep dict_eval token2id as attr for coherence

    # K search (resumable)
    ks = list(range(P["k_min"], P["k_max"] + 1, P["k_step"]))
    print(f"\n  Candidates: {ks}")
    from gensim.models.ldamodel import LdaModel
    from gensim.models.ldamulticore import LdaMulticore
    NC = (CFG["compute"]["n_cores"] if CFG["compute"]["n_cores"] else
          max(1, __import__("multiprocessing").cpu_count() - 1))
    # LdaModel monoproc is more stable/reproducible than LdaMulticore for these
    # corpus sizes and avoids joblib worker contention (which made the grid too slow).
    Cls = LdaModel
    ex = {} if Cls is LdaModel else {"workers": NC}
    print(f"  Class: {Cls.__name__} | cores: {NC}")

    res_path = RUTAS["results"] / "lda" / "lda_k_search.csv"
    resultados = []
    if res_path.exists():
        try:
            prev = pd.read_csv(res_path)
            resultados = prev.to_dict("records")
        except Exception:
            resultados = []
    done_set = {int(r["K"]) for r in resultados if "K" in r}

    for k in ks:
        if k in done_set:
            continue
        t_k = perf_counter()
        lda = Cls(corpus=corpus, num_topics=k, id2word=id2word,
                  alpha="symmetric", eta=None, passes=P["passes_lda"],
                  random_state=CFG["seed"], **ex)
        m = lda_metrics(lda, corpus, docs_tokens, dict_eval, k)
        m["elapsed_s"] = round(perf_counter() - t_k, 1)
        resultados.append(m)
        # save progress
        pd.DataFrame(resultados).sort_values("K").to_csv(res_path, index=False, encoding="utf-8-sig")
        print(f"  K={k:>3} -> Cv={m['cv']:.4f} NPMI={m['npmi']:+.4f} "
              f"TD={m['diversity']:.4f} perpl={m['perplexity']:.2f} "
              f"dom={m['dominance']:.3f} ent={m['entropy_norm']:.3f} "
              f"({_fmt(perf_counter() - t_k)})")

    df_k = pd.DataFrame(resultados).sort_values("K").reset_index(drop=True)
    guardar_csv(df_k, "lda_k_search.csv", subdir=RUTAS["results"] / "lda")

    mejor_cv = df_k.loc[df_k["cv"].idxmax()]
    print(f"\n  Best by C_v (REFERENCE ONLY): K={int(mejor_cv['K'])} "
          f"(Cv={mejor_cv['cv']:.4f})")
    print_separador("PHASE 1a COMPLETED")
    print(f"  Documents: {n_docs:,} | Candidates: {len(df_k)}")
    print(f"  Time: {_fmt(perf_counter() - t_total)}")
    print("  Next: fase1b (decide K, stability, alpha/eta)")
    return {"n_docs": n_docs, "n_candidates": len(df_k)}


if __name__ == "__main__":
    main()
