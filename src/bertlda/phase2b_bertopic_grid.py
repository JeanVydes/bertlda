# -*- coding: utf-8 -*-
"""
phase2b_bertopic_grid.py
========================
Phase 2b - BERTopic exploration: UMAP grid (Phase 1) + HDBSCAN coarse-to-fine.

Reads:  ck_corpus_base.pkl (and optionally ck_docs_embeddings.pkl)
Writes: outputs/results/bertopic/etapa1_umap.csv
        outputs/results/bertopic/etapa2_hdbscan.csv
        outputs/checkpoints/ck_docs_embeddings.pkl
        outputs/checkpoints/ck_embeddings_<slug>.pkl (one per embedding model)
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
from text_processing import clean_for_embeddings, get_params


def _guard_imports():
    missing = []
    for mod in ("umap", "hdbscan", "sentence_transformers", "gensim"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise ImportError(
            "Missing dependencies for Phase 2b: " + ", ".join(missing) +
            "\nInstall with: uv add " + " ".join(missing)
        )


def main() -> dict:
    _guard_imports()

    import umap
    import hdbscan
    import json as _json
    from sentence_transformers import SentenceTransformer
    from gensim.corpora import Dictionary

    asegurar_carpetas()
    print_separador("PHASE 2b — BERTopic GRID EXPLORATION")
    t_total = perf_counter()

    # -- Corpus --
    base = cargar_checkpoint("ck_corpus_base")
    df = base["df"].reset_index(drop=True)
    P = base["params"]
    normalizar_mayus = base.get("normalizar_mayus", False)
    n_docs = len(df)
    print(f"  {n_docs:,} documents [{P['escala']}]")

    # -- Texts --
    from joblib import Parallel, delayed
    docs = None
    ck_docs = "ck_docs_embeddings"
    if checkpoint_existe(ck_docs) if False else False:
        pass
    from pipeline_base import checkpoint_existe
    if checkpoint_existe(ck_docs):
        docs = cargar_checkpoint(ck_docs)
        if not isinstance(docs, list) or len(docs) != n_docs:
            docs = None
    if docs is None:
        docs = [clean_for_embeddings(t, normalizar_mayus) for t in df["texto"].tolist()]
        guardar_checkpoint(docs, ck_docs)
    print(f"  Prepared {len(docs):,} texts")

    # -- Embeddings (single configured model) --
    modelo_emb = CFG["embedding_model"]
    slug = modelo_emb.split("/")[-1].lower().replace("-", "_").replace(".", "_")
    slug = "".join(c if c.isalnum() or c == "_" else "_" for c in slug)
    ck_emb = f"ck_embeddings_{slug}"
    embeddings = None
    if checkpoint_existe(ck_emb):
        d = cargar_checkpoint(ck_emb)
        if d.get("modelo") == modelo_emb and d.get("n") == n_docs:
            embeddings = d["embeddings"]
            print(f"  Embeddings checkpoint: {embeddings.shape}")
    if embeddings is None:
        dev = CFG["compute"]["embeddings_device"]
        if dev == "auto":
            try:
                import torch
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                dev = "cpu"
        print(f"  Device: {dev}")
        st = SentenceTransformer(modelo_emb, device=dev)
        st.max_seq_length = CFG["max_seq_length"]
        chunk = P["chunk_size"]
        parciales = []
        for i in range(0, n_docs, chunk):
            e = st.encode(docs[i:i + chunk], batch_size=CFG["compute"]["batch_size"],
                          show_progress_bar=False, convert_to_numpy=True,
                          normalize_embeddings=CFG["normalizar_embeddings"])
            parciales.append(np.asarray(e, dtype=np.float32))
            print(f"    {min(i + chunk, n_docs):,}/{n_docs:,}")
        embeddings = np.vstack(parciales)
        guardar_checkpoint({"modelo": modelo_emb, "n": n_docs, "embeddings": embeddings}, ck_emb)
        print(f"  Embeddings: {embeddings.shape}")
        del st

    # -- Evaluation structures --
    from config import RUTAS as _R
    stop_words = build_stopwords() if False else None
    # vectorizer eval (document level)
    from sklearn.feature_extraction.text import CountVectorizer
    stops = build_stopwords()
    vcfg = CFG["vectorizer"]
    min_df = max(vcfg["min_df_min"], int(round(n_docs * vcfg["min_df_frac"])))
    vectorizer_eval = CountVectorizer(
        stop_words=sorted(stops), ngram_range=tuple(vcfg["ngram_range"]),
        min_df=min_df, max_features=vcfg["max_features"], lowercase=True,
    )
    vectorizer_eval.fit(docs)
    X_counts = vectorizer_eval.transform(docs)
    vocab = np.array(vectorizer_eval.get_feature_names_out())
    analyzer = vectorizer_eval.build_analyzer()
    docs_tokensens = [analyzer(d) for d in docs]
    dict_eval = Dictionary(docs_tokensens)
    dict_eval.filter_extremes(no_below=2, no_above=0.9)
    print(f"  Vectorizer: {len(vocab):,} terms (min_df={min_df})")

    def _metricas(labels, reducido):
        from gensim.models.coherencemodel import CoherenceModel
        from sklearn.metrics import silhouette_score
        from sklearn.metrics.pairwise import cosine_similarity
        labels = np.asarray(labels)
        validos = sorted(set(labels.tolist()) - {-1})
        n_top = len(validos)
        n_out = int((labels == -1).sum())
        m = {"n_topics": n_top, "n_outliers": n_out,
             "outlier_rate": round(n_out / max(n_docs, 1), 4)}
        if n_top < 2:
            m.update({"cv": 0.0, "npmi": 0.0, "diversity": 0.0,
                      "redundancy": 1.0, "separation": 0.0,
                      "dominance": 1.0, "score": 0.0, "valid": False})
            return m
        tam = np.array([(labels == t).sum() for t in validos], dtype=float)
        m["dominance"] = round(float(tam.max() / tam.sum()), 4)
        palabras = []
        for t in validos:
            fila = np.asarray(X_counts[labels == t].sum(axis=0)).ravel()
            top = vocab[np.argsort(-fila)[:10]]
            top = [w for w in top if w in dict_eval.token2id]
            if len(top) >= 2:
                palabras.append(list(top))
        if len(palabras) >= 2:
            for clave, medida in [("cv", "c_v"), ("npmi", "c_npmi")]:
                try:
                    m[clave] = round(float(CoherenceModel(
                        topics=palabras, texts=docs_tokensens, dictionary=dict_eval,
                        coherence=medida, processes=1).get_coherence()), 4)
                except Exception:
                    m[clave] = 0.0
            plano = [w for g in palabras for w in g]
            m["diversity"] = round(len(set(plano)) / len(plano), 4)
        else:
            m.update({"cv": 0.0, "npmi": 0.0, "diversity": 0.0})
        perfiles = []
        for t in validos:
            fila = np.asarray(X_counts[labels == t].sum(axis=0)).ravel()
            perfiles.append(fila)
        Pm = np.vstack(perfiles).astype(float)
        Pm = Pm / (Pm.sum(axis=1, keepdims=True) + 1e-12)
        S = cosine_similarity(Pm) if False else None
        from sklearn.metrics.pairwise import cosine_similarity as cs
        S = cs(Pm)
        tri = S[np.triu_indices_from(S, k=1)]
        umbral = CFG["umbral_redundancia"]
        m["redundancy"] = round(float((tri > umbral).mean()) if tri.size else 0.0, 4)
        mask = labels != -1
        try:
            m["separation"] = round(float(silhouette_score(
                np.asarray(reducido)[mask], labels[mask])), 4)
        except Exception:
            m["separation"] = 0.0
        R = CFG["restricciones"]
        ok = (m["n_topics"] >= R["min_n_topics"]
              and m["outlier_rate"] <= R["max_outlier_rate"]
              and m["dominance"] <= R["max_dominance"])
        m["valid"] = ok
        if ok:
            w = CFG["pesos_score"]
            m["score"] = round(float(
                w["cv"] * max(m["cv"], 0)
                + w["npmi"] * max((m["npmi"] + 1) / 2, 0)
                + w["diversity"] * m["diversity"]
                + w["redundancy"] * (1 - m["redundancy"])
                + w["separation"] * max((m["separation"] + 1) / 2, 0)
                + w["coverage"] * (1 - m["outlier_rate"])
            ), 5)
        else:
            m["score"] = 0.0
        return m

    def cumple(m):
        R = CFG["restricciones"]
        return (m["n_topics"] >= R["min_n_topics"]
                and m["outlier_rate"] <= R["max_outlier_rate"]
                and m["dominance"] <= R["max_dominance"])

    # Stage 1 - UMAP
    print(f"\n-- STAGE 1: UMAP EXPLORATION ({len(P['umap_grid'])} combos) --")
    ck_e1 = None
    from pipeline_base import CHECKPOINT_DIR
    e1_path = CHECKPOINT_DIR / "ck_etapa1_umap.pkl"
    res_e1 = []
    if e1_path.exists():
        import pickle
        res_e1 = pickle.load(open(e1_path, "rb"))
    done_set = {(r["n_neighbors"], r["n_components"]) for r in res_e1}
    proyecciones = {}
    for cfg in P["umap_grid"]:
        clave = (cfg["n_neighbors"], cfg["n_components"])
        ck_proj = CHECKPOINT_DIR / f"ck_proj_{clave[0]}_{clave[1]}.pkl"
        try:
            red = umap.UMAP(n_neighbors=cfg["n_neighbors"],
                            n_components=cfg["n_components"],
                            min_dist=cfg["min_dist"], metric="cosine",
                            random_state=CFG["seed"]).fit_transform(embeddings)
            red = np.asarray(red, dtype=np.float32)
            import pickle
            with open(ck_proj, "wb") as f:
                pickle.dump(red, f)
            proyecciones[clave] = red
        except Exception as e:
            print(f"  UMAP fit failed {clave}: {e}")
            continue
        if clave in done_set:
            continue
        t_u = perf_counter()
        try:
            cl = hdbscan.HDBSCAN(**P["hdbscan_ref"])
            labels = cl.fit_predict(np.asarray(red, dtype=np.float64))
            m = _metricas(labels, red)
            ok = m["valid"]
            fila = {**cfg, **m, "valida": ok,
                    "motivo": None if ok else "restricciones",
                    "score": m["score"], "elapsed_s": round(perf_counter() - t_u, 1)}
            res_e1.append(fila)
            import pickle
            with open(e1_path, "wb") as f:
                pickle.dump(res_e1, f)
            print(f"  nn={cfg['n_neighbors']:<3} nc={cfg['n_components']:<3} "
                  f"-> {m['n_topics']:>3} topics | out {m['outlier_rate'] * 100:5.1f}% "
                  f"| Cv={m.get('cv', 0):.3f} | score {m['score']:.4f} "
                  f"{'OK' if ok else 'REJECTED'}")
        except Exception as e:
            print(f"  {clave} ERROR: {e}")

    df_e1 = pd.DataFrame(res_e1).sort_values("score", ascending=False)
    guardar_csv(df_e1, "etapa1_umap.csv", subdir=RUTAS["results"] / "bertopic")

    valid_rows = df_e1[df_e1["valida"]] if "valida" in df_e1 else df_e1
    if valid_rows.empty:
        print("\n  NO configuration passed the constraints.")
        valid_rows = df_e1.sort_values("outlier_rate")
    n_umap = min(CFG["busqueda"]["n_top_refinamiento"],
                 P.get("n_top_umap_hdbscan", CFG["busqueda"]["n_top_refinamiento"]))
    top_umap = valid_rows.sort_values("score", ascending=False).head(n_umap)
    print("\n  Best UMAP:", ", ".join(
        f"nn={int(r['n_neighbors'])}/nc={int(r['n_components'])}" for _, r in top_umap.iterrows()
    ))

    # Stage 2 - HDBSCAN coarse-to-fine
    print("\n-- STAGE 2: HDBSCAN COARSE-TO-FINE --")
    e2a_path = CHECKPOINT_DIR / "ck_etapa2a_hdbscan_coarse.pkl"
    e2b_path = CHECKPOINT_DIR / "ck_etapa2b_hdbscan_fino.pkl"
    res_e2a = pickle.load(open(e2a_path, "rb")) if e2a_path.exists() else []
    res_e2b = pickle.load(open(e2b_path, "rb")) if e2b_path.exists() else []

    metodo_base = CFG["busqueda"]["hdbscan_coarse_method"]
    eps_base = float(CFG["busqueda"]["hdbscan_coarse_epsilon"])
    from text_processing import ms_para
    combos_a = []
    done_set_a = {(r["n_neighbors"], r["n_components"], r["mcs"], r["ms"]) for r in res_e2a}
    for _, ru in top_umap.iterrows():
        nn, nc = int(ru["n_neighbors"]), int(ru["n_components"])
        for mcs in P["mcs_grid"]:
            for ms in ms_para(mcs):
                k = (nn, nc, mcs, ms)
                if k not in done_set_a:
                    combos_a.append(k)

    for (nn, nc, mcs, ms) in combos_a:
        red = proyecciones.get((nn, nc))
        if red is None:
            cp = CHECKPOINT_DIR / f"ck_proj_{nn}_{nc}.pkl"
            if cp.exists():
                red = pickle.load(open(cp, "rb"))
            else:
                continue
        t_h = perf_counter()
        try:
            cl = hdbscan.HDBSCAN(min_cluster_size=int(mcs), min_samples=int(ms),
                                 cluster_selection_method=metodo_base,
                                 cluster_selection_epsilon=eps_base,
                                 metric="euclidean", prediction_data=True)
            labels = cl.fit_predict(np.asarray(red, dtype=np.float64))
            m = _metricas(labels, red)
            ok = m["valid"]
            fila = {"fase": "coarse", "n_neighbors": nn, "n_components": nc,
                    "mcs": mcs, "ms": ms, "metodo": metodo_base, "eps": eps_base,
                    **m, "valida": ok,
                    "motivo": None if ok else "restricciones",
                    "score": m["score"], "elapsed_s": round(perf_counter() - t_h, 1)}
            res_e2a.append(fila)
            with open(e2a_path, "wb") as f:
                pickle.dump(res_e2a, f)
            if ok:
                print(f"  A nn{nn} nc{nc} mcs{mcs:<5} ms{ms:<4} -> "
                      f"{m['n_topics']:>3} topics | out {m['outlier_rate'] * 100:5.1f}% "
                      f"| score {m['score']:.4f}")
        except Exception as e:
            print(f"  A nn{nn} nc{nc} mcs={mcs} ms={ms} ERROR: {e}")

    df_e2a = pd.DataFrame(res_e2a)
    valid_a = df_e2a[df_e2a["valida"]] if ("valida" in df_e2a and len(df_e2a)) else df_e2a
    if valid_a.empty:
        raise RuntimeError("No coarse configuration passed the constraints.")
    top_base = valid_a.sort_values("score", ascending=False).head(
        int(P.get("n_top_hdbscan_fino", 5)))
    done_set_b = {(r["n_neighbors"], r["n_components"], r["mcs"], r["ms"],
                 r["metodo"], r["eps"]) for r in res_e2b}
    combos_b = []
    for _, r in top_base.iterrows():
        nn, nc = int(r["n_neighbors"]), int(r["n_components"])
        mcs, ms = int(r["mcs"]), int(r["ms"])
        for metodo in P["metodos_grid"]:
            for eps in P["eps_grid"]:
                eps = float(eps)
                if metodo == metodo_base and abs(eps - eps_base) < 1e-12:
                    continue
                k = (nn, nc, mcs, ms, metodo, eps)
                if k not in done_set_b:
                    combos_b.append(k)

    for (nn, nc, mcs, ms, metodo, eps) in combos_b:
        red = proyecciones.get((nn, nc))
        if red is None:
            cp = CHECKPOINT_DIR / f"ck_proj_{nn}_{nc}.pkl"
            if cp.exists():
                red = pickle.load(open(cp, "rb"))
            else:
                continue
        t_h = perf_counter()
        try:
            cl = hdbscan.HDBSCAN(min_cluster_size=int(mcs), min_samples=int(ms),
                                 cluster_selection_method=metodo,
                                 cluster_selection_epsilon=float(eps),
                                 metric="euclidean", prediction_data=True)
            labels = cl.fit_predict(np.asarray(red, dtype=np.float64))
            m = _metricas(labels, red)
            ok = m["valid"]
            fila = {"fase": "fine", "n_neighbors": nn, "n_components": nc,
                    "mcs": mcs, "ms": ms, "metodo": metodo, "eps": eps,
                    **m, "valida": ok,
                    "motivo": None if ok else "restricciones",
                    "score": m["score"], "elapsed_s": round(perf_counter() - t_h, 1)}
            res_e2b.append(fila)
            with open(e2b_path, "wb") as f:
                pickle.dump(res_e2b, f)
            if ok:
                print(f"  B nn{nn} nc{nc} mcs{mcs:<5} ms{ms:<4} {metodo:<4} "
                      f"eps{eps:.2f} -> {m['n_topics']:>3} topics | "
                      f"out {m['outlier_rate'] * 100:5.1f}% | score {m['score']:.4f}")
        except Exception as e:
            print(f"  B nn{nn} nc{nc} mcs={mcs} ms={ms} {metodo} eps={eps} ERROR: {e}")

    df_e2 = pd.concat([pd.DataFrame(res_e2a), pd.DataFrame(res_e2b)],
                       ignore_index=True, sort=False).sort_values("score", ascending=False)
    guardar_csv(df_e2, "etapa2_hdbscan.csv", subdir=RUTAS["results"] / "bertopic")

    valid2 = df_e2[df_e2["valida"]] if ("valida" in df_e2 and len(df_e2)) else df_e2
    print(f"\n  Total combos: {len(df_e2):,} ({len(res_e2a):,} coarse + "
          f"{len(res_e2b):,} fine) | valid: {len(valid2):,}")
    print(f"  Best score: {valid2.iloc[0]['score']:.5f}")
    print_separador("PHASE 2b COMPLETED")
    print(f"  Time: {_fmt(perf_counter() - t_total)}")
    print("  Next: review etapa2_hdbscan.csv and set CFG['seleccion_bertopic'] in config.py,")
    print("        then run phase2c.")
    return {"n_combos": len(df_e2), "n_valid": len(valid2)}


if __name__ == "__main__":
    main()
