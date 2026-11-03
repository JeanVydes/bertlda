# -*- coding: utf-8 -*-
"""
phase1c_lda_final.py
====================
Phase 1c - train the single final LDA model using K/alpha/eta from phase1b.

Reads:  ck_corpus_base, ck_corpus_lda, ck_k_optimum, ck_lda_final_params
Writes: outputs/models/lda/lda_final.model
        outputs/results/lda/lda_theta.csv (document-topic matrix)
        outputs/results/lda/lda_topic_summary.csv
        outputs/results/lda/lda_ranking.csv
        outputs/results/lda/model_configuration.json
"""
from __future__ import annotations

import json
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

from config import CFG, RUTAS, COL_TITLE, COL_TEXT, COL_DOC_ID, COL_ANIO
from pipeline_base import (
    asegurar_carpetas, cargar_checkpoint, guardar_checkpoint,
    guardar_csv, guardar_json, checkpoint_existe, _fmt, print_separador,
)
from phase1a_lda_diagnostic import lda_metrics


def _fmt_t(s):
    m, r = divmod(s, 60)
    return f"{int(m):3d}m {r:05.2f}s"


def main() -> dict:
    asegurar_carpetas()
    print_separador("PHASE 1c — FINAL LDA MODEL")
    t_total = perf_counter()

    from gensim.models.ldamodel import LdaModel
    from gensim.models.ldamulticore import LdaMulticore

    for n in ["ck_corpus_base", "ck_corpus_lda"]:
        if not checkpoint_existe(n):
            raise FileNotFoundError(f"Missing {n}.pkl - run previous phases.")

    base = cargar_checkpoint("ck_corpus_base")
    df = base["df"].reset_index(drop=True)
    P = base["params"]

    ck_lda = cargar_checkpoint("ck_corpus_lda")
    docs_tokens = (ck_lda.get("docs_tokens") or ck_lda.get("docs_tok")
                   or ck_lda.get("docs_tokensens"))
    if docs_tokens is None:
        raise RuntimeError("ck_corpus_lda missing docs_tokens key")
    id2word = ck_lda["id2word"]
    corpus = ck_lda["corpus"]

    # K/alpha/eta
    k_default = None
    try:
        k_default = cargar_checkpoint("ck_k_optimum")
    except FileNotFoundError:
        pass
    params = {}
    try:
        params = cargar_checkpoint("ck_lda_final_params")
    except FileNotFoundError:
        pass
    # supports either int checkpoint or dict checkpoint
    if isinstance(k_default, dict):
        params = {**params, **k_default}
    elif k_default is not None:
        params["k"] = k_default
    if "k" not in params:
        params["k"] = int(input("Enter K (failed to load ck_k_optimum): ")) if False else 10
    k_final = int(params["k"])
    alpha_final = params.get("alpha", "symmetric")
    eta_final = params.get("eta", "symmetric")

    print(f"  {len(df):,} documents | K={k_final} | alpha={alpha_final} | eta={eta_final}")

    # Override support via env-ish (CLI args handled by simple argv parse)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--alpha", type=str, default=None)
    parser.add_argument("--eta", type=str, default=None)
    args, _ = parser.parse_known_args()
    def _cast(v):
        if v is None:
            return None
        try:
            return float(v)
        except ValueError:
            return v
    if args.k is not None:
        k_final = args.k
    if args.alpha is not None:
        alpha_final = _cast(args.alpha)
    if args.eta is not None:
        eta_final = _cast(args.eta)

    # train
    print(f"\n-- TRAINING FINAL LDA (K={k_final}, passes={CFG['lda']['passes_final']}) --")
    t0 = perf_counter()
    ncfg = CFG["compute"]["n_cores"]
    import multiprocessing
    if ncfg is None:
        ncfg = max(1, multiprocessing.cpu_count() - 1)
    USE_MULTI = False
    if USE_MULTI and ncfg > 1:
        from gensim.models.ldamulticore import LdaMulticore
        lda = LdaMulticore(corpus=corpus, num_topics=k_final, id2word=id2word,
                           alpha=alpha_final, eta=eta_final,
                           passes=CFG["lda"]["passes_final"], random_state=CFG["seed"],
                           workers=ncfg)
    else:
        lda = LdaModel(corpus=corpus, num_topics=k_final, id2word=id2word,
                       alpha=alpha_final, eta=eta_final,
                       passes=CFG["lda"]["passes_final"], random_state=CFG["seed"])
    model_path = RUTAS["models"] / "lda" / "lda_final.model"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    lda.save(str(model_path))
    print(f"  Model -> {model_path}")
    print(f"  Training time: {_fmt_t(perf_counter() - t0)}")

    # THETA matrix aligned by doc_id
    print("\n-- THETA MATRIX --")
    t0 = perf_counter()
    theta = np.zeros((len(corpus), k_final))
    for i, c in enumerate(corpus):
        for t, p in lda.get_document_topics(c, minimum_probability=0):
            theta[i, t] = p
    df_theta = pd.DataFrame({
        COL_DOC_ID: df[COL_DOC_ID].values,
        COL_TITLE: df[COL_TITLE].values,
        COL_ANIO: df[COL_ANIO].values,
        "dominant_topic": (np.argmax(theta, axis=1) + 1),
        "confidence": np.round(np.max(theta, axis=1), 4),
    })
    if "Country_normalized" in df.columns:
        df_theta["country_normalized"] = df["Country_normalized"].values
    elif "pais_normalizado" in df.columns:
        df_theta["country_normalized"] = df["pais_normalizado"].values
    for i in range(k_final):
        df_theta[f"topic_{i + 1}"] = np.round(theta[:, i], 6)
    print(f"  Shape: {df_theta.shape} | time: {_fmt_t(perf_counter() - t0)}")
    guardar_csv(df_theta, "lda_theta.csv", subdir=RUTAS["results"] / "lda")

    # TOPIC SUMMARY
    print("\n-- TOPIC SUMMARY --")
    conteo = df_theta["dominant_topic"].value_counts().to_dict()
    summary = []
    has_country = "country_normalized" in df_theta.columns
    if has_country:
        total_por_pais = (df_theta.dropna(subset=["country_normalized"])
                          .groupby("country_normalized").size())
    for i in range(k_final):
        tid = i + 1
        kw = ", ".join(w for w, _ in lda.show_topic(i, topn=20))
        fila = {"topic_id": tid, "articles_count": conteo.get(tid, 0),
                "keywords_20": kw}
        if has_country:
            sub = df_theta[df_theta["dominant_topic"] == tid]
            conteo_pais = sub["country_normalized"].value_counts()
            with np.errstate(divide="ignore", invalid="ignore"):
                esp = (conteo_pais / total_por_pais.reindex(conteo_pais.index)).dropna()
            fila["leading_country"] = esp.idxmax() if len(esp) else "N/A"
            fila["leading_country_specialization"] = round(float(esp.max()), 3) if len(esp) else float("nan")
        summary.append(fila)
        lc = fila.get("leading_country", "")
        top3 = ", ".join(kw.split(", ")[:3])
        print(f"  T{tid:02d} ({conteo.get(tid, 0):5d} docs)  {str(lc):<20}  {top3}")

    df_summary = pd.DataFrame(summary)
    guardar_csv(df_summary, "lda_topic_summary.csv", subdir=RUTAS["results"] / "lda")

    # Ranking for RIF
    df_rk = pd.DataFrame(summary)[["topic_id", "articles_count", "keywords_20"]]
    df_rk.columns = ["rank", "frequency", "label"]
    df_rk["label"] = df_rk.apply(
        lambda r: f"T{r['rank']}: " + ", ".join(
            x.strip() for x in str(r["label"]).split(",")[:3]), axis=1)
    df_rk = df_rk.sort_values("frequency", ascending=False).reset_index(drop=True)
    df_rk["rank"] = range(1, len(df_rk) + 1)
    guardar_csv(df_rk, "lda_ranking.csv", subdir=RUTAS["results"] / "lda")

    # model_configuration.json
    entorno = {}
    try:
        import platform, datetime
        entorno = {"fecha": datetime.datetime.now().isoformat(timespec="seconds"),
                   "python": sys.version.split()[0], "plataforma": platform.platform(),
                   "seed": CFG["seed"]}
    except Exception:
        pass
    cfg_json = {
        "fecha": entorno.get("fecha"),
        "k_final": k_final, "alpha": str(alpha_final), "eta": str(eta_final),
        "passes": CFG["lda"]["passes_final"], "seed": CFG["seed"],
        "n_docs": len(corpus), "vocab": len(id2word),
    }
    guardar_json(cfg_json, "model_configuration_lda")

    print_separador("PHASE 1c COMPLETED")
    print(f"  K={k_final} | alpha={alpha_final} | eta={eta_final}")
    print(f"  Docs: {len(corpus):,} | Time: {_fmt_t(perf_counter() - t_total)}")
    print("  Next: phase3 (visualizations) or phase4 (trends)")
    return {"k_final": k_final, "alpha_final": alpha_final, "eta_final": eta_final}


if __name__ == "__main__":
    main()
