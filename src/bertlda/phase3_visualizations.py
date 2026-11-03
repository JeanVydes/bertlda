# -*- coding: utf-8 -*-
"""
phase3_visualizations.py
========================
Phase 3 - visualizations for LDA (+ BERTopic if available).

Outputs (PNG + SVG, in outputs/visualizations/):
    LDA_Similitud_{Coseno,Hellinger,JensenShannon}.png/.svg
    LDA_TopWords_Heatmap.png/.svg
    LDA_Topic_x_Year.png/.svg
    LDA_Topic_x_Country.png/.svg  (if a country column is present)
    BERTopic_* (if bertopic_final model exists)
"""
from __future__ import annotations

import os
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    _DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _DIR = os.getcwd()
sys.path.insert(0, _DIR)

from config import CFG, RUTAS, COL_DOC_ID, COL_ANIO
from pipeline_base import (
    asegurar_carpetas, cargar_checkpoint, checkpoint_existe,
    guardar_csv, _fmt, print_separador,
)

DPI = CFG["viz"]["dpi"]
FORMATOS = CFG["viz"]["formatos"]
TOP_WORDS = CFG["viz"]["top_words"]
TOP_PAISES = CFG["viz"]["top_paises"]
MIN_DOCS_ANIO = 20


def _guardar_fig(fig, nombre):
    for fmt in FORMATOS:
        fig.savefig(RUTAS["visualizations"] / f"{nombre}.{fmt}", dpi=DPI,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {nombre}.{{{','.join(FORMATOS)}}}")


def _viz_lda_similitudes(lda):
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import entropy
    K = lda.num_topics
    labels = [f"T{i + 1}" for i in range(K)]
    tm = lda.get_topics()
    cos = 1 - squareform(pdist(tm, metric="cosine"))
    def hellinger(p, q):
        return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))
    hell = np.zeros((K, K)); js = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            hell[i, j] = 1 - hellinger(tm[i], tm[j])
            m = 0.5 * (tm[i] + tm[j])
            jsd = 0.5 * entropy(tm[i], m) + 0.5 * entropy(tm[j], m)
            js[i, j] = 1 - np.clip(jsd, 0, 1)
    np.fill_diagonal(hell, 1); np.fill_diagonal(js, 1)
    for nombre, mat in [("LDA_Similarity_Cosine", cos),
                        ("LDA_Similitud_Hellinger", hell),
                        ("LDA_Similitud_JensenShannon", js)]:
        fig, ax = plt.subplots(figsize=(max(6, K * 0.45), max(5, K * 0.4)))
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(K)); ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticks(range(K)); ax.set_yticklabels(labels, fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.75)
        fig.tight_layout()
        _guardar_fig(fig, nombre)


def _viz_lda_topwords(lda):
    K = lda.num_topics
    rows = {}
    todas = set()
    for i in range(K):
        top = lda.show_topic(i, topn=TOP_WORDS)
        rows[i] = dict(top)
        todas |= {w for w, _ in top}
    vocab = sorted(todas)
    M = np.array([[rows[i].get(w, 0) for w in vocab] for i in range(K)])
    fig, ax = plt.subplots(figsize=(max(10, len(vocab) * 0.22),
                                    max(5, K * 0.4)))
    im = ax.imshow(M, aspect="auto", cmap="Reds")
    ax.set_xticks(range(len(vocab)))
    ax.set_xticklabels(vocab, rotation=90, fontsize=6)
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"T{i + 1}" for i in range(K)], fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Probability")
    fig.tight_layout()
    _guardar_fig(fig, "LDA_TopWords_Heatmap")


def _heatmap_topico_anio(df, col_topic, col_anio, etiquetas, nombre_salida):
    d = df[[col_anio, col_topic]].dropna()
    d = d[d[col_topic] != -1]
    d[col_anio] = d[col_anio].astype(int)
    total = d.groupby(col_anio).size()
    validos = [a for a in total[total >= MIN_DOCS_ANIO].index.tolist()
               if a != int(total.index.max())]
    if len(validos) < 3:
        print(f"    Skipped {nombre_salida}: only {len(validos)} valid years.")
        return
    tabla = (d[d[col_anio].isin(validos)]
             .groupby([col_anio, col_topic]).size().unstack(fill_value=0))
    tabla = tabla.reindex(validos, fill_value=0)
    prop = tabla.div(tabla.sum(axis=1), axis=0) * 100
    orden = prop.mean().sort_values(ascending=False).index
    prop = prop[orden[:CFG["viz"]["max_topics_heatmap"]]]
    rows = [str(etiquetas.get(t, t))[:38] for t in prop.columns]
    fig, ax = plt.subplots(figsize=(max(8, len(validos) * 0.55),
                                    max(4, len(rows) * 0.32)))
    im = ax.imshow(prop.T.values, aspect="auto", cmap="Reds")
    ax.set_xticks(range(len(validos)))
    ax.set_xticklabels(validos, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    ax.set_xlabel("Year")
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label("Annual prevalence (%)", fontsize=9)
    fig.tight_layout()
    _guardar_fig(fig, nombre_salida)


def _heatmap_topico_pais(df, col_topic, col_pais, etiquetas, nombre_salida):
    d = df[[col_pais, col_topic]].dropna()
    d = d[d[col_topic] != -1]
    top_paises = d[col_pais].value_counts().head(TOP_PAISES).index.tolist()
    d = d[d[col_pais].isin(top_paises)]
    if d.empty:
        print("    Skipped: not enough country data.")
        return
    tabla = d.groupby([col_pais, col_topic]).size().unstack(fill_value=0)
    prop = tabla.div(tabla.sum(axis=1), axis=0) * 100
    prop = prop.reindex(top_paises)
    orden = prop.mean().sort_values(ascending=False).index
    prop = prop[orden[:CFG["viz"]["max_topics_heatmap"]]]
    cols = [str(etiquetas.get(t, t))[:30] for t in prop.columns]
    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 0.5),
                                    max(4, len(top_paises) * 0.35)))
    im = ax.imshow(prop.values, aspect="auto", cmap="Reds")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(top_paises)))
    ax.set_yticklabels(top_paises, fontsize=9)
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label("Share of country's documents (%)", fontsize=9)
    fig.tight_layout()
    _guardar_fig(fig, nombre_salida)


def main() -> dict:
    asegurar_carpetas()
    print_separador("PHASE 3 — VISUALIZATIONS")
    t0 = perf_counter()
    generated = []

    # LDA
    print("\n-- LDA --")
    model_path = RUTAS["models"] / "lda" / "lda_final.model"
    theta_path = RUTAS["results"] / "lda" / "lda_theta.csv"
    summary_path = RUTAS["results"] / "lda" / "lda_topic_summary.csv"
    try:
        from gensim.models.ldamodel import LdaModel
        from gensim.models.ldamulticore import LdaMulticore
        try:
            lda = LdaMulticore.load(str(model_path))
        except Exception:
            lda = LdaModel.load(str(model_path))
        _viz_lda_similitudes(lda); generated += ["LDA_Similarity_Cosine",
                                                  "LDA_Similitud_Hellinger",
                                                  "LDA_Similitud_JensenShannon"]
        _viz_lda_topwords(lda); generated.append("LDA_TopWords_Heatmap")
    except Exception as e:
        print(f"  LDA visualizations skipped: {e}")

    etiquetas = {}
    if summary_path.exists():
        try:
            res = pd.read_csv(summary_path)
            for _, r in res.iterrows():
                kw = str(r.get("keywords_20", "")).split(",")[:3]
                etiquetas[int(r["topic_id"])] = ", ".join(k.strip() for k in kw)
        except Exception:
            pass

    if theta_path.exists():
        theta = pd.read_csv(theta_path)
        if "anio" in theta.columns and "dominant_topic" in theta.columns:
            _heatmap_topico_anio(theta, "dominant_topic", "anio", etiquetas,
                                "LDA_Topic_x_Year")
            generated.append("LDA_Topic_x_Year")
        country_col = None
        for cand in ("country_normalized", "Country_normalized"):
            if cand in theta.columns:
                country_col = cand
                break
        if country_col and "dominant_topic" in theta.columns:
            _heatmap_topico_pais(theta, "dominant_topic", country_col, etiquetas,
                                "LDA_Topic_x_Country")
            generated.append("LDA_Topic_x_Country")

    # BERTopic (optional)
    print("\n-- BERTopic --")
    bt_model_path = RUTAS["models"] / "bertopic" / "bertopic_final"
    bt_info_path = RUTAS["results"] / "bertopic" / "bertopic_topic_info.csv"
    bt_docs_path = RUTAS["results"] / "bertopic" / "resultados_bertopic.csv"
    if bt_model_path.exists() and bt_docs_path.exists():
        try:
            from bertopic import BERTopic
            topic_model = BERTopic.load(str(bt_model_path))
            info = pd.read_csv(bt_info_path)
            docs_df = pd.read_csv(bt_docs_path)
            # topwords heatmap
            if "Topic" in info.columns and "Representation" in info.columns:
                val = info[info["Topic"] != -1].sort_values("Count", ascending=False).head(
                    CFG["viz"]["max_topics_heatmap"])
                todas = set(); rows = {}
                for _, r in val.iterrows():
                    rep = r["Representation"]
                    pal = (rep if isinstance(rep, list)
                           else str(rep).strip("[]").replace("'", "").split(","))
                    pal = [p.strip() for p in pal if p.strip()][:TOP_WORDS]
                    rows[r["Topic"]] = pal; todas |= set(pal)
                vocab = sorted(todas)
                M = np.array([[1.0 if w in rows[t] else 0.0 for w in vocab]
                              for t in val["Topic"]])
                nombres = [str(n)[:38] for n in val["Name"]]
                fig, ax = plt.subplots(figsize=(max(10, len(vocab) * 0.18),
                                                max(5, len(nombres) * 0.32)))
                ax.imshow(M, aspect="auto", cmap="Reds")
                ax.set_xticks(range(len(vocab)))
                ax.set_xticklabels(vocab, rotation=90, fontsize=6)
                ax.set_yticks(range(len(nombres)))
                ax.set_yticklabels(nombres, fontsize=8)
                fig.tight_layout()
                _guardar_fig(fig, "BERTopic_TopWords_Heatmap")
                generated.append("BERTopic_TopWords_Heatmap")

            bt_etiquetas = dict(zip(info["Topic"], info["Name"])) if "Topic" in info.columns else {}
            col_topic = "topic_reduced" if "topic_reduced" in docs_df.columns else "topic_original"
            if "anio" in docs_df.columns:
                _heatmap_topico_anio(docs_df, col_topic, "anio", bt_etiquetas,
                                    "BERTopic_Topic_x_Year")
                generated.append("BERTopic_Topic_x_Year")
            if "country" in docs_df.columns:
                _heatmap_topico_pais(docs_df, col_topic, "country", bt_etiquetas,
                                    "BERTopic_Topic_x_Country")
                generated.append("BERTopic_Topic_x_Country")
        except Exception as e:
            print(f"  BERTopic visualizations failed: {e}")
    else:
        print("  No BERTopic model found; skipping.")

    print_separador("PHASE 3 COMPLETED")
    print(f"  Generated: {len(generated)}")
    for g in generated:
        print(f"    - {g}")
    print(f"  Time: {_fmt(perf_counter() - t0)}")
    print(f"  Folder: {RUTAS['visualizations']}")
    return generated


if __name__ == "__main__":
    main()
