# -*- coding: utf-8 -*-
"""
phase2a_embeddings_compare.py
=============================
Phase 2a (optional) - compare candidate embedding models.

Reads:  ck_corpus_base.pkl
Writes: outputs/results/bertopic/embeddings_comparison.csv
        outputs/checkpoints/ck_embeddings_<slug>.pkl (one per model)
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

from config import CFG, RUTAS
from pipeline_base import (
    asegurar_carpetas, cargar_checkpoint, guardar_checkpoint,
    guardar_csv, _fmt, print_separador,
)
from text_processing import clean_for_embeddings


def _guard():
    for m in ("sentence_transformers", "umap", "hdbscan", "sklearn"):
        try:
            __import__(m)
        except ImportError:
            raise ImportError(f"Missing: {m}. Install: uv add {m.split('_')[0]}")


def main() -> dict:
    _guard()
    import umap
    import hdbscan as _hd
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from gensim.corpora import Dictionary

    asegurar_carpetas()
    print_separador("PHASE 2a — EMBEDDINGS COMPARISON")
    t_total = perf_counter()

    base = cargar_checkpoint("ck_corpus_base")
    df = base["df"].reset_index(drop=True)
    P = base["params"]
    normalizar_mayus = base.get("normalizar_mayus", False)
    N_MUESTRA = None
    if N_MUESTRA and len(df) > N_MUESTRA:
        idx = np.random.default_rng(CFG["seed"]).choice(len(df), N_MUESTRA, replace=False)
        df_m = df.iloc[sorted(idx)].reset_index(drop=True)
    else:
        df_m = df.reset_index(drop=True)
    n_docs = len(df_m)
    print(f"  {n_docs:,} documents")

    docs = [clean_for_embeddings(t, normalizar_mayus) for t in df_m["texto"].tolist()]
    from nltk.corpus import stopwords as _sw
    import nltk
    nltk.download("stopwords", quiet=True)
    stops = set(_sw.words("english"))
    vcfg = CFG["vectorizer"]
    min_df = max(vcfg["min_df_min"], int(round(n_docs * vcfg["min_df_frac"])))
    vec = CountVectorizer(stop_words=sorted(stops), ngram_range=tuple(vcfg["ngram_range"]),
                         min_df=min_df, max_features=vcfg["max_features"], lowercase=True)
    vec.fit(docs)
    X_counts = vec.transform(docs)
    vocab = np.array(vec.get_feature_names_out())
    analyzer = vec.build_analyzer()
    docs_tokensens = [analyzer(d) for d in docs]
    dict_eval = Dictionary(docs_tokensens)
    dict_eval.filter_extremes(no_below=2, no_above=0.9)

    def evaluar(emb):
        cfg = P["umap_grid"][0] if P["umap_grid"] else {"n_neighbors": 15, "n_components": 5, "min_dist": 0.0}
        reducido = umap.UMAP(random_state=CFG["seed"], **cfg).fit_transform(emb)
        cl = _hd.HDBSCAN(**P["hdbscan_ref"])
        labels = cl.fit_predict(np.asarray(reducido, dtype=np.float64))
        from gensim.models.coherencemodel import CoherenceModel
        from sklearn.metrics import silhouette_score
        from sklearn.metrics.pairwise import cosine_similarity
        validos = sorted(set(labels.tolist()) - {-1})
        n_out = int((labels == -1).sum())
        m = {"n_topics": len(validos), "outlier_rate": round(n_out / max(n_docs, 1), 4)}
        if len(validos) >= 2:
            tam = np.array([(labels == t).sum() for t in validos], dtype=float)
            m["dominance"] = round(float(tam.max() / tam.sum()), 4)
            palabras = []
            for t in validos:
                fila = np.asarray(X_counts[labels == t].sum(axis=0)).ravel()
                top = [w for w in vocab[np.argsort(-fila)[:10]] if w in dict_eval.token2id]
                if len(top) >= 2:
                    palabras.append(top)
            if len(palabras) >= 2:
                for clave, medida in [("cv", "c_v"), ("npmi", "c_npmi")]:
                    try:
                        m[clave] = round(float(CoherenceModel(topics=palabras, texts=docs_tokensens,
                                                             dictionary=dict_eval, coherence=medida,
                                                             processes=1).get_coherence()), 4)
                    except Exception:
                        m[clave] = 0.0
                plano = [w for g in palabras for w in g]
                m["diversity"] = round(len(set(plano)) / len(plano), 4)
            perfiles = [np.asarray(X_counts[labels == t].sum(axis=0)).ravel() for t in validos]
            Pm = np.vstack(perfiles).astype(float) + 1e-12
            Pm = Pm / Pm.sum(axis=1, keepdims=True)
            S = cosine_similarity(Pm)
            tri = S[np.triu_indices_from(S, k=1)]
            m["redundancy"] = round(float((tri > CFG["umbral_redundancia"]).mean()) if tri.size else 0.0, 4)
            mask = labels != -1
            try:
                m["separation"] = round(float(silhouette_score(np.asarray(reducido)[mask], labels[mask])), 4)
            except Exception:
                m["separation"] = 0.0
            w = CFG["pesos_score"]
            R = CFG["restricciones"]
            ok = (m["n_topics"] >= R["min_n_topics"] and m["outlier_rate"] <= R["max_outlier_rate"]
                  and m.get("dominance", 1) <= R["max_dominance"])
            m["valid"] = ok
            m["score"] = round(float(
                w["cv"] * max(m.get("cv", 0), 0)
                + w["npmi"] * max((m.get("npmi", 0) + 1) / 2, 0)
                + w["diversity"] * m.get("diversity", 0)
                + w["redundancy"] * (1 - m.get("redundancy", 0))
                + w["separation"] * max((m.get("separation", 0) + 1) / 2, 0)
                + w["coverage"] * (1 - m["outlier_rate"])
            ), 5) if ok else 0.0
        else:
            m.update({"cv": 0.0, "npmi": 0.0, "diversity": 0.0, "redundancy": 1.0,
                      "separation": 0.0, "dominance": 1.0, "score": 0.0, "valid": False})
        return m

    dev = CFG["compute"]["embeddings_device"]
    if dev == "auto":
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            dev = "cpu"
    print(f"  Device: {dev}")

    rows = []
    for nombre, ruta_modelo in CFG.get("embedding_candidatos", []):
        print(f"\n  MODEL: {nombre}")
        slug = "".join(c if c.isalnum() else "_" for c in nombre.lower())
        ck_emb = f"ck_embeddings_{slug}"
        emb = None
        if (RUTAS["checkpoints"] / f"{ck_emb}.pkl").exists():
            d = cargar_checkpoint(ck_emb)
            if d.get("modelo") == ruta_modelo and d.get("n") == n_docs:
                emb = d["embeddings"]
        if emb is None:
            st = SentenceTransformer(ruta_modelo, device=dev)
            st.max_seq_length = CFG["max_seq_length"]
            emb = st.encode(docs, batch_size=CFG["compute"]["batch_size"],
                            show_progress_bar=True, convert_to_numpy=True,
                            normalize_embeddings=CFG["normalizar_embeddings"])
            emb = np.asarray(emb, dtype=np.float32)
            guardar_checkpoint({"modelo": ruta_modelo, "n": n_docs, "embeddings": emb}, ck_emb)
            del st
        m = evaluar(emb)
        rows.append({"modelo": nombre, "ruta": ruta_modelo, "dim": int(emb.shape[1]), **m})
        print(f"    -> {m['n_topics']} topics | out {m['outlier_rate'] * 100:.1f}% "
              f"| Cv={m.get('cv', 0):.4f} | score={m['score']:.4f}")
        del emb

    dfr = pd.DataFrame(rows)
    guardar_csv(dfr, "embeddings_comparison.csv", subdir=RUTAS["results"] / "bertopic")
    if "score" in dfr and dfr["score"].notna().any():
        gan = dfr.sort_values("score", ascending=False).iloc[0]
        print(f"\n  WINNER BY SCORE: {gan['modelo']} ({gan['score']:.4f})")
        print(f"  -> Set CFG['embedding_model'] = '{gan['ruta']}' in config.py")
    print_separador("PHASE 2a COMPLETED")
    print(f"  Time: {_fmt(perf_counter() - t_total)}")
    return {"n_models": len(rows)}


if __name__ == "__main__":
    main()
