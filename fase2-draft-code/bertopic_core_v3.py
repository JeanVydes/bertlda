# -*- coding: utf-8 -*-
"""
bertopic_core_v3.py
===================
Núcleo compartido del pipeline BERTopic.

Este módulo NO se ejecuta directamente. Centraliza las funciones que deben ser
idénticas en:
  - Script02_Grid_BERTopic_v3.py
  - Script02_Final_BERTopic_v3.py
  - Script02b_Embeddings_Compare.py

Objetivo: evitar dependencias entre scripts ejecutables y prevenir divergencias
entre copias de las métricas, restricciones o score multicriterio.
"""

import numpy as np

from config import CFG

SEED = CFG["seed"]


class UMAPPrecomputado:
    """Envuelve una proyección UMAP ya calculada para reutilizarla en BERTopic."""

    def __init__(self, reducido):
        if hasattr(reducido, "get"):
            reducido = reducido.get()
        self.reduced = np.asarray(reducido, dtype=np.float32)
        self.n_components = self.reduced.shape[1]
        self.embedding_ = self.reduced

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.reduced

    def transform(self, X):
        return self.reduced[:len(X)]


def get_umap(random_state=None, **kw):
    """
    Construye UMAP usando el backend definido en config.py.

    random_state:
      None -> usa CFG['seed'].
      entero -> permite estabilidad con semillas alternativas.
    """
    backend = CFG["compute"]["umap_backend"]
    rs = SEED if random_state is None else random_state

    if backend == "cuml":
        try:
            from cuml.manifold import UMAP as U  # type: ignore
            print("    AVISO: backend cuML — NO determinista, no publicable.")
            return U(**kw)
        except ImportError:
            pass

    from umap import UMAP as U
    return U(random_state=rs, **kw)


def get_hdbscan(**kw):
    """Construye HDBSCAN usando el backend definido en config.py."""
    backend = CFG["compute"]["hdbscan_backend"]

    if backend == "cuml":
        try:
            from cuml.cluster import HDBSCAN as H  # type: ignore
            return H(**kw)
        except ImportError:
            pass

    from hdbscan import HDBSCAN as H
    return H(**kw)


def get_vectorizers(docs, stops, n_docs):
    """
    Construye los DOS vectorizadores del pipeline. Son objetos distintos porque
    operan sobre unidades de análisis distintas:

      1) vectorizer_eval : se ajusta sobre los n_docs documentos individuales.
         Define el espacio léxico del estudio (min_df proporcional al corpus)
         y alimenta X_counts / vocab / docs_tok para coherencia y métricas.

      2) vectorizer_ctfidf : el que se pasa a BERTopic. BERTopic NO lo ajusta
         sobre los documentos: en _extract_topics agrupa por tópico
         (groupby("Topic").agg(" ".join)) y ajusta el vectorizador sobre K
         SUPER-DOCUMENTOS (uno por tópico). Con min_df=32 y K≈24 sklearn
         compara max_doc_count=K contra min_doc_count=32 y lanza
         "max_df corresponds to < documents than min_df".

    Fijar `vocabulary=` en el segundo es preferible a bajar min_df: cuando el
    vocabulario es fijo (fixed_vocabulary_=True) sklearn omite por completo el
    bloque de filtrado por frecuencia documental, y además garantiza que las
    representaciones c-TF-IDF vivan EXACTAMENTE en el mismo espacio léxico con
    el que se calcularon Cv/NPMI. Sin esto, coherencia y top-words hablan de
    vocabularios distintos y la comparación con LDA deja de ser limpia.

    Devuelve (vectorizer_eval, vectorizer_ctfidf, min_df).
    """
    from sklearn.feature_extraction.text import CountVectorizer

    vcfg = CFG["vectorizer"]
    min_df = max(vcfg["min_df_min"], int(round(n_docs * vcfg["min_df_frac"])))

    vectorizer_eval = CountVectorizer(
        stop_words=sorted(stops),
        ngram_range=vcfg["ngram_range"],
        min_df=min_df,
        max_features=vcfg["max_features"],
        lowercase=True,
    )
    vectorizer_eval.fit(docs)

    # Mismo analizador (ngram_range, lowercase), vocabulario congelado.
    # stop_words no se repite: ya están excluidas del vocabulario.
    vectorizer_ctfidf = CountVectorizer(
        vocabulary=vectorizer_eval.vocabulary_,
        ngram_range=vcfg["ngram_range"],
        lowercase=True,
    )

    return vectorizer_eval, vectorizer_ctfidf, min_df


def calcular_metricas(labels, reducido, docs_tok, dict_eval,
                      vectorizer, X_counts, vocab, n_docs):
    """
    Calcula las dimensiones de evaluación usadas por TODO el pipeline.

    `vectorizer` se conserva en la firma por compatibilidad aunque las
    estructuras necesarias ya llegan precalculadas.
    """
    from gensim.models.coherencemodel import CoherenceModel
    from sklearn.metrics import silhouette_score
    from sklearn.metrics.pairwise import cosine_similarity

    labels = np.asarray(labels)
    validos = sorted(set(labels.tolist()) - {-1})
    n_top = len(validos)
    n_out = int((labels == -1).sum())

    m = {
        "n_topics": n_top,
        "n_outliers": n_out,
        "outlier_rate": round(n_out / max(n_docs, 1), 4),
    }

    if n_top < 2:
        m.update({
            "cv": 0.0,
            "npmi": 0.0,
            "diversity": 0.0,
            "redundancy": 1.0,
            "separation": 0.0,
            "dominance": 1.0,
            "topic_min": 0,
            "entropy_norm": 0.0,
        })
        return m

    tam = np.array([(labels == t).sum() for t in validos], dtype=float)
    m["dominance"] = round(float(tam.max() / tam.sum()), 4)
    m["topic_min"] = int(tam.min())
    m["topic_max"] = int(tam.max())

    p = tam / tam.sum()
    m["entropy_norm"] = round(
        float(-(p * np.log(p + 1e-12)).sum() / np.log(max(n_top, 2))), 4
    )

    perfiles, palabras = [], []
    for t in validos:
        fila = np.asarray(X_counts[labels == t].sum(axis=0)).ravel()
        perfiles.append(fila)
        top = vocab[np.argsort(-fila)[:10]]
        top = [w for w in top if w in dict_eval.token2id]
        if len(top) >= 2:
            palabras.append(list(top))

    if len(palabras) >= 2:
        for clave, medida in [("cv", "c_v"), ("npmi", "c_npmi")]:
            try:
                m[clave] = round(float(CoherenceModel(
                    topics=palabras,
                    texts=docs_tok,
                    dictionary=dict_eval,
                    coherence=medida,
                    processes=1,
                ).get_coherence()), 4)
            except Exception:
                m[clave] = 0.0

        plano = [w for grupo in palabras for w in grupo]
        m["diversity"] = round(len(set(plano)) / len(plano), 4)
    else:
        m.update({"cv": 0.0, "npmi": 0.0, "diversity": 0.0})

    Pm = np.vstack(perfiles).astype(float)
    Pm = Pm / (Pm.sum(axis=1, keepdims=True) + 1e-12)
    S = cosine_similarity(Pm)
    tri = S[np.triu_indices_from(S, k=1)]

    umbral = CFG["umbral_redundancia"]
    m["redundancy"] = round(
        float((tri > umbral).mean()) if tri.size else 0.0, 4
    )
    m["sim_media_topicos"] = round(
        float(tri.mean()) if tri.size else 0.0, 4
    )

    mask = labels != -1
    try:
        m["separation"] = round(float(silhouette_score(
            np.asarray(reducido)[mask], labels[mask]
        )), 4)
    except Exception:
        m["separation"] = 0.0

    return m


def cumple_restricciones(m):
    """Filtro duro declarado a priori. Devuelve (bool, motivo)."""
    R = CFG["restricciones"]

    if m["n_topics"] < R["min_n_topics"]:
        return False, f"n_topics<{R['min_n_topics']}"

    if m["outlier_rate"] > R["max_outlier_rate"]:
        return False, f"outliers>{R['max_outlier_rate']:.0%}"

    if m.get("dominance", 1.0) > R["max_dominance"]:
        return False, f"dominancia>{R['max_dominance']:.0%}"

    return True, "ok"


def score(m):
    """
    Score multicriterio único para Grid, Final y comparación de embeddings.
    No incorpora bonus por número de tópicos.
    """
    w = CFG["pesos_score"]

    s = (
        w["cv"] * max(m.get("cv", 0), 0)
        + w["npmi"] * max((m.get("npmi", 0) + 1) / 2, 0)
        + w["diversity"] * m.get("diversity", 0)
        + w["redundancy"] * (1 - m.get("redundancy", 0))
        + w["separation"] * max((m.get("separation", 0) + 1) / 2, 0)
        + w["coverage"] * (1 - m.get("outlier_rate", 1))
    )
    return round(float(s), 5)
