# -*- coding: utf-8 -*-
"""
Script01_LDA_v3.py
===================
FASE DE DIAGNÓSTICO. Explora K y reporta métricas. NO entrena el modelo
final ni decide K por argmax(C_v) — esa decisión ocurre en Script01b.

CAMBIOS FRENTE A v2
-------------------
1. Ya NO entrena un modelo final (v2 lo hacía dos veces: aquí por argmax de
   C_v, y otra vez en Script01c — con criterios distintos, sin declararlo).
2. limpiar_para_lda corregido (lematiza antes de filtrar; procesador_v3).
3. Consume el corpus de Script00: mismo texto, mismo doc_id, misma
   normalización de casing que usa BERTopic. Antes cada script releía
   BD.xlsx por separado y podía construir un corpus ligeramente distinto.
4. K_min/K_max/K_step vienen de get_params(n_docs), no de una lista fija
   editada a mano.
5. Métricas multicriterio (C_v, NPMI, diversidad, dominancia, entropía),
   igual que BERTopic — no solo coherencia, para que ambos modelos se
   auditen con el mismo estándar (§10).
6. Se guarda docs_tok y el Dictionary de evaluación al mismo tiempo que el
   corpus, para que Script01b no tenga que reconstruirlos.

FASES
  A  Corpus (desde ck_corpus_base.pkl de Script00)
  B  Tokenización, bigramas, diccionario, corpus BoW
  C  Búsqueda de K (alpha/eta por defecto, coherencia + diversidad + tamaño)

SALIDAS
  03_Checkpoints/ck_corpus_lda.pkl     (docs_tok, dictionary, corpus, bigram)
  02_Resultados_Entrenamiento/LDA_K_Search.xlsx

SIGUIENTE PASO
  Script01b_Stability_v3.py  → decide K y busca alpha/eta
"""

import os
import sys
import pickle
import warnings
from time import perf_counter

try:
    _DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _DIR = os.getcwd()
sys.path.insert(0, _DIR)

import numpy as np
import pandas as pd

from config import CFG, rutas, crear_rutas, montar_drive, fijar_semillas, n_cores
import procesador_v3 as pr

warnings.filterwarnings("ignore")

SEED = CFG['seed']


def _fmt(s):
    m, s = divmod(s, 60)
    return f"{int(m):3d}m {s:05.2f}s"


def _ck(nombre):
    return os.path.join(rutas['check'], f'ck_{nombre}.pkl')


# =============================================================================
# MÉTRICAS (compartidas en espíritu con Script02_BERTopic_v3.calcular_metricas)
# =============================================================================
def metricas_lda(lda, corpus, docs_tok, dict_eval, k, top_n=10):
    from gensim.models.coherencemodel import CoherenceModel

    topics_words = [[w for w, _ in lda.show_topic(i, topn=top_n)]
                    for i in range(k)]

    m = {}
    for clave, medida in [('cv', 'c_v'), ('npmi', 'c_npmi')]:
        try:
            m[clave] = round(float(CoherenceModel(
                topics=topics_words, texts=docs_tok, dictionary=dict_eval,
                coherence=medida, processes=1).get_coherence()), 4)
        except Exception:
            m[clave] = 0.0

    plano = [w for grupo in topics_words for w in grupo]
    m['diversity'] = round(len(set(plano)) / len(plano), 4)

    m['perplexity'] = round(float(lda.log_perplexity(corpus)), 4)

    dominante = [int(np.argmax([p for _, p in
                                lda.get_document_topics(d, minimum_probability=0)]))
                for d in corpus]
    tam = np.array([dominante.count(i) for i in range(k)], dtype=float)
    tam = tam[tam > 0]
    m['dominance'] = round(float(tam.max() / tam.sum()), 4) if len(tam) else 1.0
    p = tam / tam.sum() if len(tam) else np.array([1.0])
    m['entropy_norm'] = round(float(-(p * np.log(p + 1e-12)).sum()
                                    / np.log(max(k, 2))), 4)
    m['topics_vacios'] = int(k - len(tam))
    return m


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':

    t_total = perf_counter()
    crear_rutas()
    montar_drive()
    fijar_semillas()
    NC = n_cores()

    print("\n" + "=" * 66)
    print("SCRIPT 01 — LDA: DIAGNÓSTICO Y BÚSQUEDA DE K")
    print(f"  SEED={SEED}")
    print("=" * 66)

    # ── FASE A — CORPUS ──────────────────────────────────────────────────────
    t0 = perf_counter()
    print("\n── FASE A: CORPUS ──")
    ck_base = os.path.join(rutas['check'], 'ck_corpus_base.pkl')
    if not os.path.exists(ck_base):
        raise FileNotFoundError(
            "ck_corpus_base.pkl no encontrado.\n"
            "Ejecuta Script00_Auditoria_Corpus.py primero.")
    base = pickle.load(open(ck_base, 'rb'))
    df = base['df'].reset_index(drop=True)
    P = base['params']
    n_docs = len(df)
    print(f"  {n_docs:,} documentos [{P['escala']}]")
    print(f"  Rejilla de K: {P['k_min']}-{P['k_max']} paso {P['k_step']}")

    # ── FASE B — TOKENIZACIÓN, BIGRAMAS, DICCIONARIO ─────────────────────────
    t0 = perf_counter()
    print("\n── FASE B: TOKENIZACIÓN Y CORPUS BoW ──")
    ck_lda = _ck('corpus_lda')
    if os.path.exists(ck_lda):
        d = pickle.load(open(ck_lda, 'rb'))
        if d.get('n_docs') == n_docs:
            docs_tok, id2word, corpus, dict_eval = (
                d['docs_tok'], d['id2word'], d['corpus'], d['dict_eval'])
            print("  Checkpoint recuperado.")
        else:
            docs_tok = None
    else:
        docs_tok = None

    if docs_tok is None:
        dominio = pickle.load(open(_ck('stopwords_dominio'), 'rb')) \
            if os.path.exists(_ck('stopwords_dominio')) else set()
        stops = pr.construir_stopwords(dominio=dominio)
        print(f"  Stopwords: {len(stops):,}")

        docs_tok = pr.procesar_lda(df['texto'].tolist(), stops, n_cores=NC)

        print("  Detectando bigramas...")
        from gensim.models.phrases import Phrases, Phraser
        bigram = Phrases(docs_tok, min_count=P['bigram_min_count'],
                         threshold=P['bigram_threshold'])
        bigram_phraser = Phraser(bigram)   # .freeze() equivalente: libera el modelo entrenable
        docs_tok = [bigram_phraser[d] for d in docs_tok]

        from gensim.corpora import Dictionary
        id2word = Dictionary(docs_tok)
        id2word.filter_extremes(no_below=P['no_below'], no_above=P['no_above'])
        corpus = [id2word.doc2bow(d) for d in docs_tok]

        # Diccionario de evaluación: igual criterio que BERTopic (no_below=2,
        # no_above=0.9), independiente del diccionario de entrenamiento.
        dict_eval = Dictionary(docs_tok)
        dict_eval.filter_extremes(no_below=2, no_above=0.9)

        pickle.dump({'docs_tok': docs_tok, 'id2word': id2word,
                     'corpus': corpus, 'dict_eval': dict_eval,
                     'n_docs': n_docs}, open(ck_lda, 'wb'))

    print(f"  Vocabulario (entrenamiento): {len(id2word):,} términos "
          f"(no_below={P['no_below']}, no_above={P['no_above']})")
    print(f"  Vocabulario (evaluación)   : {len(dict_eval):,} términos")
    vacios = sum(1 for c in corpus if len(c) == 0)
    if vacios:
        print(f"  AVISO: {vacios} documentos con BoW vacío tras filtrar "
              f"extremos (afectarán la estimación de θ).")
    tiempos = {'B - Tokenización/BoW': perf_counter() - t0}

    # ── FASE C — BÚSQUEDA DE K ───────────────────────────────────────────────
    t0 = perf_counter()
    print("\n── FASE C: BÚSQUEDA DE K ──")
    ks = list(range(P['k_min'], P['k_max'] + 1, P['k_step']))
    print(f"  Candidatos: {ks}")

    from gensim.models.ldamodel import LdaModel
    from gensim.models.ldamulticore import LdaMulticore
    Cls = LdaModel if (P['escala'] == 'MICRO' or NC == 1) else LdaMulticore
    ex = {} if Cls is LdaModel else {'workers': NC}
    print(f"  Clase: {Cls.__name__} | núcleos: {NC}")

    ck_search = os.path.join(rutas['sens'], 'ck_k_search.pkl')
    resultados = pickle.load(open(ck_search, 'rb')) if os.path.exists(ck_search) else []
    hechos = {r['K'] for r in resultados}

    for k in ks:
        if k in hechos:
            continue
        t_k = perf_counter()
        lda = Cls(corpus=corpus, num_topics=k, id2word=id2word,
                  alpha='symmetric', eta=None,
                  passes=P['passes_lda'], random_state=SEED, **ex)
        m = metricas_lda(lda, corpus, docs_tok, dict_eval, k)
        m['K'] = k
        m['tiempo_s'] = round(perf_counter() - t_k, 1)
        resultados.append(m)
        pickle.dump(resultados, open(ck_search, 'wb'))
        print(f"  K={k:>3} -> Cv={m['cv']:.4f} NPMI={m['npmi']:+.4f} "
              f"TD={m['diversity']:.4f} perpl={m['perplexity']:.2f} "
              f"dom={m['dominance']:.3f} ent={m['entropy_norm']:.3f} "
              f"({_fmt(m['tiempo_s'])})")

    df_k = pd.DataFrame(resultados).sort_values('K').reset_index(drop=True)
    xl = os.path.join(rutas['res_train'], 'LDA_K_Search.xlsx')
    df_k.to_excel(xl, index=False)
    print(f"\n  Excel -> {xl}")
    tiempos['C - Búsqueda K'] = perf_counter() - t0

    # ── Vista previa: el mejor por C_v es solo INFORMATIVO ──────────────────
    mejor_cv = df_k.loc[df_k['cv'].idxmax()]
    print(f"\n  Mejor por C_v (SOLO REFERENCIA, no es la decisión): "
          f"K={int(mejor_cv['K'])} (Cv={mejor_cv['cv']:.4f})")
    print("  argmax(C_v) tiende a preferir K grandes en corpus reales; la "
          "decisión final pondera también estabilidad, parsimonia e "
          "interpretabilidad. Ver Script01b.")

    print("\n" + "=" * 66)
    print("SCRIPT 01 COMPLETADO")
    for k_, v_ in tiempos.items():
        print(f"  {k_:<28} {_fmt(v_):>12}")
    print(f"  {'TOTAL':<28} {_fmt(perf_counter()-t_total):>12}")
    print("\n── SIGUIENTE PASO ──")
    print("  Script01b_Stability_v3.py  → decide K y busca alpha/eta")
    print("=" * 66)
