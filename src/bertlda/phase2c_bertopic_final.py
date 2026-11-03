# -*- coding: utf-8 -*-
"""
phase2c_bertopic_final.py
========================
Phase 2c - final BERTopic model built from a manually selected grid combination.

Prerequisite: edit config.CFG['seleccion_bertopic'] with values copied verbatim
from outputs/results/bertopic/etapa2_hdbscan.csv (produced by phase2b).

Reads:  ck_corpus_base, ck_docs_embeddings, ck_embeddings_<slug>, etapa2_hdbscan.csv
Writes: outputs/models/bertopic/bertopic_final/
        outputs/results/bertopic/resultados_bertopic.csv (doc-topic assignment)
        outputs/results/bertopic/bertopic_topic_info.csv
        outputs/results/bertopic/bertopic_ranking.csv
        outputs/results/bertopic/model_configuration_bertopic.json
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

from config import CFG, RUTAS
from pipeline_base import (
    asegurar_carpetas, cargar_checkpoint, guardar_checkpoint,
    guardar_csv, guardar_json, _fmt, print_separador,
)
from text_processing import clean_for_embeddings


def _guard():
    for m in ("bertopic", "umap", "hdbscan", "sentence_transformers"):
        try:
            __import__(m)
        except ImportError:
            raise ImportError(f"Missing dependency: {m}. Install: uv add {m}")


def main() -> dict:
    _guard()
    import numpy as _np
    from bertopic import BERTopic
    from umap import UMAP
    import hdbscan as _hdbscan
    from bertopic.vectorizers import ClassTfidfTransformer
    import scipy.sparse as sp
    from sentence_transformers import SentenceTransformer

    asegurar_carpetas()
    print_separador("PHASE 2c — FINAL BERTOPIC MODEL")
    t_total = perf_counter()

    base = cargar_checkpoint("ck_corpus_base")
    df = base["df"].reset_index(drop=True)
    P = base["params"]
    normalizar_mayus = base.get("normalizar_mayus", False)
    n_docs = len(df)
    print(f"  {n_docs:,} documents [{P['escala']}]")

    # Docs
    ck_docs = "ck_docs_embeddings"
    from pipeline_base import checkpoint_existe
    docs = None
    if checkpoint_existe(ck_docs):
        d = cargar_checkpoint(ck_docs)
        if isinstance(d, list) and len(d) == n_docs:
            docs = d
    if docs is None:
        docs = [clean_for_embeddings(t, normalizar_mayus) for t in df["texto"].tolist()]
        guardar_checkpoint(docs, ck_docs)

    # Embeddings (configured model)
    modelo_emb = CFG["embedding_model"]
    slug = modelo_emb.split("/")[-1].lower()
    slug = "".join(c if c.isalnum() else "_" for c in slug)
    ck_emb = f"ck_embeddings_{slug}"
    embeddings = None
    if checkpoint_existe(ck_emb):
        d = cargar_checkpoint(ck_emb)
        if d.get("modelo") == modelo_emb and d.get("n") == n_docs:
            embeddings = d["embeddings"]
    if embeddings is None:
        dev = CFG["compute"]["embeddings_device"]
        if dev == "auto":
            try:
                import torch
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                dev = "cpu"
        st = SentenceTransformer(modelo_emb, device=dev)
        st.max_seq_length = CFG["max_seq_length"]
        chunk = P["chunk_size"]
        parciales = []
        for i in range(0, n_docs, chunk):
            e = st.encode(docs[i:i + chunk], batch_size=CFG["compute"]["batch_size"],
                          show_progress_bar=False, convert_to_numpy=True,
                          normalize_embeddings=CFG["normalizar_embeddings"])
            parciales.append(np.asarray(e, dtype=np.float32))
            print(f"    {min(i+chunk, n_docs):,}/{n_docs:,}")
        embeddings = np.vstack(parciales)
        guardar_checkpoint({"modelo": modelo_emb, "n": n_docs, "embeddings": embeddings}, ck_emb)

    # Evaluation structures
    from sklearn.feature_extraction.text import CountVectorizer
    from config import (CFG as _CFG,)
    vcfg = _CFG["vectorizer"]
    # Use functional stopwords (layer 1) for both vectorizers
    import nltk
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords as _sw
    stops = set(_sw.words("english"))
    min_df = max(vcfg["min_df_min"], int(round(n_docs * vcfg["min_df_frac"])))
    vectorizer_eval = CountVectorizer(stop_words=sorted(stops),
                                     ngram_range=tuple(vcfg["ngram_range"]),
                                     min_df=min_df, max_features=vcfg["max_features"],
                                     lowercase=True)
    vectorizer_eval.fit(docs)
    vectorizer_ctfidf = CountVectorizer(vocabulary=vectorizer_eval.vocabulary_,
                                       ngram_range=tuple(vcfg["ngram_range"]),
                                       lowercase=True)

    # Stage D - manual selection
    print("\n-- MANUAL SELECTION (from etapa2_hdbscan.csv) --")
    grid_path = RUTAS["results"] / "bertopic" / "etapa2_hdbscan.csv"
    if not grid_path.exists():
        raise FileNotFoundError(f"{grid_path} not found. Run phase2b first.")
    sel = CFG["seleccion_bertopic"]
    required = ["n_neighbors", "n_components", "min_cluster_size", "min_samples",
                "metodo", "epsilon"]
    if any(sel.get(k) is None for k in required):
        raise ValueError("Fill CFG['seleccion_bertopic'] in config.py with a row from etapa2_hdbscan.csv")
    df_e2 = pd.read_csv(grid_path)
    nn = int(sel["n_neighbors"]); nc = int(sel["n_components"])
    mcs = int(sel["min_cluster_size"]); ms = int(sel["min_samples"])
    metodo = str(sel["metodo"]).lower().strip(); eps = float(sel["epsilon"])
    mask = (
        (df_e2["n_neighbors"].astype(int) == nn)
        & (df_e2["n_components"].astype(int) == nc)
        & (df_e2["mcs"].astype(int) == mcs)
        & (df_e2["ms"].astype(int) == ms)
        & (df_e2["metodo"].astype(str).str.lower().str.strip() == metodo)
        & (df_e2["eps"].astype(float).sub(eps).abs() < 1e-9)
    )
    cand = df_e2.loc[mask]
    if cand.empty:
        raise ValueError("Selected combination not found in etapa2_hdbscan.csv.")
    r = cand.sort_values("score", ascending=False).iloc[0]
    print(f"  nn={nn} nc={nc} mcs={mcs} ms={ms} {metodo} eps={eps:.2f}")
    print(f"  Grid: {int(r.get('n_topics', 0))} topics | outliers={float(r.get('outlier_rate', 0))*100:.1f}% "
          f"| score={float(r['score']):.5f}")

    # Stage E - stability of selected config (Umap with multiple seeds)
    print("\n-- STABILITY OF SELECTED CONFIG --")
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    seeds = CFG["busqueda"]["seeds_stability"]
    etiquetas, ks = [], []
    for s in seeds:
        try:
            red_s = UMAP(n_neighbors=nn, n_components=nc, min_dist=0.0,
                         metric="cosine", random_state=s).fit_transform(embeddings)
            cl = _hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                                  cluster_selection_method=metodo,
                                  cluster_selection_epsilon=eps,
                                  metric="euclidean", prediction_data=True)
            lab = cl.fit_predict(np.asarray(red_s, dtype=np.float64))
            etiquetas.append(lab); ks.append(len(set(lab.tolist()) - {-1}))
        except Exception as e:
            print(f"    seed {s} failed: {e}")
    aris, nmis = [], []
    for i in range(len(etiquetas)):
        for j in range(i + 1, len(etiquetas)):
            aris.append(adjusted_rand_score(etiquetas[i], etiquetas[j]))
            nmis.append(normalized_mutual_info_score(etiquetas[i], etiquetas[j]))
    stability = {
        "ARI_medio": round(float(np.mean(aris)), 4) if aris else None,
        "ARI_min": round(float(np.min(aris)), 4) if aris else None,
        "NMI_medio": round(float(np.mean(nmis)), 4) if nmis else None,
        "K_medio": round(float(np.mean(ks)), 2) if ks else None,
        "K_sd": round(float(np.std(ks)), 2) if ks else None,
    }
    print(f"  ARI={stability['ARI_medio']} | NMI={stability['NMI_medio']} | "
          f"K={stability['K_medio']}±{stability['K_sd']}")

    # Stage F - final model
    print("\n-- FINAL MODEL --")
    def _limpiar_ctfidf(m):
        if sp.issparse(m):
            m = m.tocsr()
            m.data = np.nan_to_num(m.data, nan=0.0, posinf=0.0, neginf=0.0)
            return m
        return np.nan_to_num(np.asarray(m, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    _fit_orig = ClassTfidfTransformer.fit_transform
    _tr_orig = ClassTfidfTransformer.transform

    def _safe_fit(self, X, y=None):
        return _limpiar_ctfidf(_fit_orig(self, X, y))
    def _safe_tr(self, X):
        return _limpiar_ctfidf(_tr_orig(self, X))
    ClassTfidfTransformer.fit_transform = _safe_fit
    ClassTfidfTransformer.transform = _safe_tr

    umap_model = UMAP(n_neighbors=nn, n_components=nc, min_dist=0.0, metric="cosine")
    hdbscan_model = _hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                                     cluster_selection_method=metodo,
                                     cluster_selection_epsilon=eps,
                                     metric="euclidean", prediction_data=True)
    ctfidf_model = ClassTfidfTransformer(
        reduce_frequent_words=CFG["ctfidf_reduce_frequent_words"])
    topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model,
                           vectorizer_model=vectorizer_ctfidf,
                           ctfidf_model=ctfidf_model, language="english",
                           calculate_probabilities=False, verbose=False)

    t0 = perf_counter()
    topics_orig, _ = topic_model.fit_transform(docs, embeddings=embeddings)
    topics_orig = list(topics_orig)
    n_out_orig = int(sum(1 for t in topics_orig if t == -1))
    print(f"  Topics: {len(set(topics_orig) - {-1})} | outliers: {n_out_orig} "
          f"({n_out_orig / n_docs * 100:.1f}%)")
    _limpiar_ctfidf(topic_model) if hasattr(topic_model, "c_tf_idf_") else None

    topics_red = topics_orig[:]
    if CFG["outliers"]["reasignar"] and n_out_orig > 0:
        th = CFG["outliers"]["threshold"]
        print(f"\n  Reassigning outliers (threshold={th})")
        try:
            topics_red = list(topic_model.reduce_outliers(
                docs, topics_orig, strategy=CFG["outliers"]["estrategia"],
                threshold=th))
            topic_model.update_topics(docs, topics=topics_red,
                                     vectorizer_model=vectorizer_ctfidf,
                                     ctfidf_model=ctfidf_model)
            n_out_red = int(sum(1 for t in topics_red if t == -1))
            print(f"  Outliers {n_out_orig} -> {n_out_red} "
                  f"({n_out_orig - n_out_red} reassigned)")
        except Exception as e:
            print(f"  Reassignment failed ({e}); keeping original.")
            topics_red = topics_orig

    # Save model (local) — BERTopic save
    model_dir = RUTAS["models_bertopic"] / "bertopic_final"
    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        topic_model.save(str(model_dir), serialization="safetensors",
                        save_ctfidf=True, save_embedding_model=CFG["embedding_model"])
    except Exception as e:
        print(f"  safetensors unavailable ({e}); falling back to pickle.")
        topic_model.save(str(model_dir), serialization="pickle")
    print(f"  Saved model: {model_dir}")

    info = topic_model.get_topic_info()
    nombres = dict(zip(info["Topic"] if "Topic" in info.columns else info.index, info["Name"]))

    # Export
    documentos = pd.DataFrame({
        "doc_id": df["doc_id"].values if "doc_id" in df.columns else range(n_docs),
        "topic_original": topics_orig,
        "topic_reduced": topics_red,
        "topic_label": [nombres.get(t, "Outlier") for t in topics_red],
        "reassigned": [a == -1 and b != -1 for a, b in zip(topics_orig, topics_red)],
        "anio": df["Year"].values if "Year" in df.columns else None,
        "country": df["pais_normalizado"].values if "pais_normalizado" in df.columns else None,
        "title": df["Title"].values if "Title" in df.columns else None,
    })
    guardar_csv(info, "bertopic_topic_info.csv", subdir=RUTAS["results"] / "bertopic")
    guardar_csv(documentos, "resultados_bertopic.csv", subdir=RUTAS["results"] / "bertopic")

    # ranking for RIF
    val = info[info["Topic"] != -1].copy() if "Topic" in info.columns else info.copy()
    if "Count" in val.columns:
        val["etiqueta"] = val.apply(
            lambda r: f"BT{r['Topic']}: " + ", ".join(
                str(p).strip() for p in
                (r["Representation"] if isinstance(r["Representation"], list)
                 else str(r["Representation"]).strip("[]").replace("'", "").split(","))[:3]),
            axis=1)
        rk = val[["etiqueta", "Count"]].copy()
        rk.columns = ["etiqueta", "frecuencia"]
        rk = rk.sort_values("frecuencia", ascending=False).reset_index(drop=True)
        rk["rank"] = range(1, len(rk) + 1)
        guardar_csv(rk, "bertopic_ranking.csv", subdir=RUTAS["results"] / "bertopic")

    guardar_checkpoint(topics_red, "bert_topics")
    guardar_checkpoint(topics_orig, "bert_topics_original")

    cfg_json = {
        "fecha": None,
        "corpus": {"n_docs": n_docs, "normalizacion_casing": normalizar_mayus},
        "embedding_model": modelo_emb,
        "umap": {"n_neighbors": nn, "n_components": nc, "min_dist": 0.0,
                 "metric": "cosine", "random_state": CFG["seed"]},
        "hdbscan": {"min_cluster_size": mcs, "min_samples": ms,
                    "cluster_selection_method": metodo,
                    "cluster_selection_epsilon": eps},
        "outliers": {**CFG["outliers"],
                     "n_original": n_out_orig,
                     "n_final": int(sum(1 for t in topics_red if t == -1))},
        "stability": {"seeds": seeds, "ARI_medio": stability["ARI_medio"]},
        "n_topics_final": int(len(set(topics_red) - {-1})),
        "seleccion": {"modo": "manual", "score_grilla": float(r["score"])},
    }
    guardar_json(cfg_json, "model_configuration_bertopic")

    print_separador("PHASE 2c COMPLETED")
    print(f"  Topics: {cfg_json['n_topics_final']} | outliers final: "
          f"{cfg_json['outliers']['n_final']}")
    print(f"  Time: {_fmt(perf_counter() - t_total)}")
    print("  Next: phase3 (visualizations) / phase4 (trends)")
    return {"n_topics_final": cfg_json["n_topics_final"]}


if __name__ == "__main__":
    main()
