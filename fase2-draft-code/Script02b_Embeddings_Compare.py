# -*- coding: utf-8 -*-
"""
Script02b_Embeddings_Compare.py
===============================
Comparación controlada de modelos de embeddings (Modelo A/B/C, §31).

POR QUÉ EXISTE
  El pipeline v2 usaba SentenceTransformer('allenai/scibert_scivocab_uncased').
  SciBERT NO es un modelo sentence-transformers: es un BERT preentrenado con
  MLM. Al cargarlo así, la librería construye un Transformer + Pooling(mean)
  cuyo pooling NUNCA fue entrenado para similitud semántica. Es un resultado
  establecido (Reimers & Gurevych, 2019) que el mean-pooling de un BERT sin
  fine-tuning rinde peor en similitud que promediar vectores GloVe: el espacio
  resultante es anisotrópico y las vecindades son débiles.

  Consecuencia: UMAP recibe una geometría pobre, HDBSCAN no encuentra regiones
  densas separadas, y la tasa de outliers se dispara. Antes de tocar
  min_cluster_size o penalizaciones, hay que verificar si el problema está
  aguas arriba.

QUÉ HACE
  Codifica la MISMA muestra con varios modelos, aplica la MISMA configuración
  UMAP+HDBSCAN y la MISMA evaluación. La única variable es el modelo.

NOTA SOBRE EL CASING
  Si Script00 detectó corpus en MAYÚSCULAS, se normaliza antes de codificar.
  Sin ese paso la comparación sería injusta: un modelo 'cased' quedaría
  penalizado por un artefacto de exportación de WoS, no por su calidad.

PRERREQUISITO
  03_Checkpoints/ck_corpus_base.pkl   (Script00)

SALIDAS
  02_Resultados_Entrenamiento/Embeddings_Comparacion.xlsx
  03_Checkpoints/ck_embeddings_<slug>.pkl   (uno por modelo, reutilizable)
"""

import os
import re
import sys
import gc
import json
import pickle
from time import perf_counter

try:
    _DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _DIR = os.getcwd()
sys.path.insert(0, _DIR)

import numpy as np
import pandas as pd

from config import (CFG, rutas, crear_rutas, montar_drive,
                    fijar_semillas, registrar_entorno, n_cores)
import procesador_v3 as pr
from bertopic_core_v3 import calcular_metricas, cumple_restricciones, score

# =============================================================================
# CONFIGURACIÓN LOCAL
# =============================================================================
# Tamaño de la muestra. None = corpus completo.
# Con <= 5.000 documentos conviene usar el corpus entero: codificar es rápido
# y evita que la conclusión dependa del muestreo.
N_MUESTRA = None

SEED = CFG['seed']

# SIN rejilla propia. Se reutiliza P['umap_grid'] y P['hdbscan_ref'] que
# produce procesador_v3.get_params(n_docs) — la MISMA función que usa
# Script02 en su Etapa 1. Una sola fuente de verdad: si el corpus crece,
# ambos scripts escalan juntos sin tocar nada aquí.
#
# Por qué NO cruzar también la rejilla de HDBSCAN (como hace la Etapa 2):
# 8 UMAP x ~102 combinaciones HDBSCAN x ~2.5s de coherencia por combo son
# ~34 min POR MODELO — inviable x4 en una sesión T4 gratuita. Replicar solo
# la Etapa 1 (UMAP explorado contra un HDBSCAN de referencia fijo) es lo que
# de todas formas decide qué modelo pasa a Script02, así que es la
# comparación correcta: mismo criterio con el que luego se elige la
# configuración final, no una aproximación aparte.


def _slug(nombre):
    return re.sub(r'[^a-z0-9]+', '_', nombre.lower()).strip('_')


# =============================================================================
# MÉTRICAS
# =============================================================================
def metricas_espacio(emb):
    """
    Diagnóstico del espacio de embeddings ANTES de agrupar.

    similitud_media : coseno medio entre pares aleatorios. Un valor alto
        (>0.8) indica anisotropía: todo se parece a todo, que es
        exactamente la patología del mean-pooling sin fine-tuning.
    """
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(emb), size=min(2000, len(emb)), replace=False)
    X = emb[idx]
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    S = Xn @ Xn.T
    tri = S[np.triu_indices_from(S, k=1)]
    return {
        'sim_media':  round(float(tri.mean()), 4),
        'sim_std':    round(float(tri.std()), 4),
        'sim_p95':    round(float(np.percentile(tri, 95)), 4),
    }


def _evaluar_una(emb, cfg_umap, hdbscan_ref, docs_tok, dict_eval,
                 X_counts, vocab, n_docs):
    """Una configuración UMAP contra el HDBSCAN de referencia. Métricas
    idénticas a las de la grilla/final (provienen de bertopic_core_v3.py)."""
    from umap import UMAP
    import hdbscan

    reducido = UMAP(random_state=SEED, **cfg_umap).fit_transform(emb)
    cl = hdbscan.HDBSCAN(**hdbscan_ref)
    labels = cl.fit_predict(np.asarray(reducido, dtype=np.float64))

    m = calcular_metricas(labels, reducido, docs_tok, dict_eval,
                          None, X_counts, vocab, n_docs)
    ok, motivo = cumple_restricciones(m)
    m['valida'] = ok
    m['motivo'] = motivo
    m['score'] = score(m) if ok else 0.0
    return m, labels


def evaluar_modelo(emb, P, docs_tok, dict_eval, X_counts, vocab, n_docs):
    """
    Barre P['umap_grid'] contra P['hdbscan_ref'] (misma rejilla y misma
    referencia que la Etapa 1 de Script02, derivadas de get_params(n_docs)).
    Devuelve (mejor_fila, todas_las_filas).
    """
    filas = []
    for cfg in P['umap_grid']:
        t = perf_counter()
        try:
            m, _ = _evaluar_una(emb, cfg, P['hdbscan_ref'], docs_tok,
                                dict_eval, X_counts, vocab, n_docs)
            filas.append({**cfg, **m, 'tiempo_s': round(perf_counter() - t, 1)})
        except Exception as e:
            print(f"      cfg nn={cfg['n_neighbors']} nc={cfg['n_components']} "
                  f"falló: {e}")
    if not filas:
        return {'score': 0.0, 'n_topics': 0, 'valida': False}, []
    mejor = max(filas, key=lambda r: (r['valida'], r['score']))
    return mejor, filas


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':

    t_total = perf_counter()
    crear_rutas()
    montar_drive()
    fijar_semillas()

    print("\n" + "=" * 66)
    print("SCRIPT 02b — COMPARACIÓN DE MODELOS DE EMBEDDINGS")
    print("=" * 66)

    ck_base = os.path.join(rutas['check'], 'ck_corpus_base.pkl')
    if not os.path.exists(ck_base):
        raise FileNotFoundError(
            "ck_corpus_base.pkl no encontrado.\n"
            "Ejecuta Script00_Auditoria_Corpus.py primero.")
    base = pickle.load(open(ck_base, 'rb'))
    df   = base['df']
    P    = base['params']
    normalizar_mayus = base['normalizar_mayus']

    print(f"  Corpus: {len(df):,} documentos [{P['escala']}]")
    print(f"  Normalización de casing: {normalizar_mayus}")

    # ── Muestra ──────────────────────────────────────────────────────────────
    if N_MUESTRA and len(df) > N_MUESTRA:
        idx = np.random.default_rng(SEED).choice(len(df), N_MUESTRA, replace=False)
        df_m = df.iloc[sorted(idx)].reset_index(drop=True)
        print(f"  Muestra aleatoria: {len(df_m):,} documentos")
    else:
        df_m = df.reset_index(drop=True)
        print(f"  Se usa el corpus COMPLETO ({len(df_m):,} documentos)")

    n_docs = len(df_m)

    # ── Textos ───────────────────────────────────────────────────────────────
    print("\n── PREPROCESAMIENTO ──")
    docs = pr.procesar_embeddings(df_m['texto'].tolist(),
                                  normalizar_mayus=normalizar_mayus,
                                  n_cores=n_cores())
    print(f"  Ejemplo: {docs[0][:110]}...")

    # ── Estructuras de evaluación: SE CALCULAN UNA SOLA VEZ ──────────────────
    # (En v2 esto se reconstruía dentro de cada iteración del grid, ~49 veces.)
    print("\n── ESTRUCTURAS DE EVALUACIÓN (una sola vez) ──")
    from sklearn.feature_extraction.text import CountVectorizer
    from gensim.corpora import Dictionary

    stops = pr.construir_stopwords()
    vcfg = CFG['vectorizer']
    min_df = max(vcfg['min_df_min'], int(round(n_docs * vcfg['min_df_frac'])))
    vectorizer = CountVectorizer(
        stop_words=sorted(stops), ngram_range=vcfg['ngram_range'],
        min_df=min_df, max_features=vcfg['max_features'], lowercase=True,
    )
    vectorizer.fit(docs)
    X_counts = vectorizer.transform(docs)
    vocab = np.array(vectorizer.get_feature_names_out())
    analyzer = vectorizer.build_analyzer()
    docs_tok = [analyzer(d) for d in docs]
    dict_eval = Dictionary(docs_tok)
    dict_eval.filter_extremes(no_below=2, no_above=0.9)
    print(f"  Vocabulario vectorizer: {len(vectorizer.vocabulary_):,} "
          f"(min_df={min_df})")
    print(f"  Diccionario coherencia: {len(dict_eval):,} términos")

    # ── Dispositivo ──────────────────────────────────────────────────────────
    dev = CFG['compute']['embeddings_device']
    if dev == 'auto':
        try:
            import torch
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            dev = 'cpu'
    print(f"\n  Dispositivo embeddings: {dev}")

    # ── Bucle de modelos ─────────────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer

    filas, detalle = [], []
    for nombre, ruta_modelo in CFG['embedding_candidatos']:
        print("\n" + "-" * 66)
        print(f"MODELO: {nombre}")
        print(f"  {ruta_modelo}")
        slug = _slug(nombre)
        ck_emb = os.path.join(rutas['check'], f'ck_embeddings_{slug}.pkl')

        try:
            # Embeddings (con checkpoint por modelo)
            if os.path.exists(ck_emb):
                d = pickle.load(open(ck_emb, 'rb'))
                if d.get('modelo') == ruta_modelo and d.get('n') == n_docs:
                    emb = d['embeddings']
                    t_emb = d.get('t_emb', 0.0)
                    print(f"  Checkpoint reutilizado -> {emb.shape}")
                else:
                    print("  Checkpoint no coincide (modelo/n distinto): se recalcula.")
                    emb = None
            else:
                emb = None

            if emb is None:
                t = perf_counter()
                st = SentenceTransformer(ruta_modelo, device=dev)
                st.max_seq_length = CFG['max_seq_length']
                low = getattr(getattr(st, 'tokenizer', None), 'do_lower_case', None)
                print(f"  do_lower_case del tokenizador: {low}")
                emb = st.encode(
                    docs,
                    batch_size=CFG['compute']['batch_size'],
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    normalize_embeddings=CFG['normalizar_embeddings'],
                )
                t_emb = perf_counter() - t
                emb = np.asarray(emb, dtype=np.float32)
                pickle.dump({'modelo': ruta_modelo, 'n': n_docs,
                             'embeddings': emb, 't_emb': t_emb,
                             'normalizar_mayus': normalizar_mayus},
                            open(ck_emb, 'wb'))
                print(f"  Embeddings {emb.shape} en {t_emb:.1f}s -> {ck_emb}")
                del st
                gc.collect()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            # Diagnóstico del espacio
            esp = metricas_espacio(emb)
            print(f"  Similitud coseno media entre pares: {esp['sim_media']:.4f} "
                  f"(sd={esp['sim_std']:.4f})")
            if esp['sim_media'] > 0.80:
                print("    AVISO: espacio anisotrópico — todo se parece a todo.")

            # Etapa 1 del propio corpus: mismo umap_grid/hdbscan_ref que
            # usará Script02, así la comparación es directamente la que
            # decide qué modelo pasa a la búsqueda final.
            print(f"  Barriendo {len(P['umap_grid'])} configuraciones "
                  f"(Etapa 1, hdbscan_ref mcs={P['hdbscan_ref']['min_cluster_size']})...")
            res, todas = evaluar_modelo(emb, P, docs_tok, dict_eval,
                                        X_counts, vocab, n_docs)
            for r in todas:
                detalle.append({'modelo': nombre, **r})
            print(f"  MEJOR: nn={res.get('n_neighbors')} "
                  f"nc={res.get('n_components')} -> "
                  f"{res['n_topics']} tópicos | "
                  f"outliers {res.get('outlier_rate',1)*100:.1f}% | "
                  f"Cv={res.get('cv',0):.4f} NPMI={res.get('npmi',0):.4f} "
                  f"sep={res.get('separation',0):+.4f} | "
                  f"SCORE={res.get('score',0):.4f}"
                  + ("" if res.get('valida') else
                     f"  [ninguna config válida: {res.get('motivo','')}]"))

            filas.append({'modelo': nombre, 'ruta': ruta_modelo,
                          'dim': int(emb.shape[1]),
                          't_embeddings_s': round(t_emb, 1),
                          **esp, **res})
            del emb
            gc.collect()

        except Exception as e:
            print(f"  ERROR con {nombre}: {type(e).__name__}: {e}")
            filas.append({'modelo': nombre, 'ruta': ruta_modelo, 'error': str(e)})

    # ── Resultados ───────────────────────────────────────────────────────────
    dfr = pd.DataFrame(filas)
    xl = os.path.join(rutas['res_train'], 'Embeddings_Comparacion.xlsx')
    with pd.ExcelWriter(xl, engine='xlsxwriter') as w:
        dfr.to_excel(w, sheet_name='Comparacion', index=False)
        pd.DataFrame(detalle).to_excel(w, sheet_name='Detalle_Rejilla', index=False)
        pd.DataFrame([{
            'umap_grid': json.dumps(P['umap_grid']),
            'hdbscan_ref': json.dumps({k: v for k, v in P['hdbscan_ref'].items()
                                       if isinstance(v, (int, float, str, bool))}),
            'n_docs': n_docs,
            'normalizar_mayus': normalizar_mayus,
            'seed': SEED,
        }]).to_excel(w, sheet_name='Configuracion', index=False)
        pd.DataFrame([registrar_entorno()]).T.reset_index().to_excel(
            w, sheet_name='Entorno', index=False, header=['clave', 'valor'])

    print("\n" + "=" * 66)
    print("RESULTADOS")
    cols = [c for c in ['modelo', 'score', 'n_topics', 'outlier_rate', 'cv',
                        'npmi', 'diversity', 'redundancy', 'separation',
                        'dominance', 'sim_media', 't_embeddings_s']
            if c in dfr.columns]
    print(dfr[cols].to_string(index=False))

    if 'score' in dfr.columns and dfr['score'].notna().any():
        gan = dfr.sort_values('score', ascending=False).iloc[0]
        print(f"\n  GANADOR POR SCORE: {gan['modelo']}  ({gan['score']:.4f})")
        print(f"    -> config.py:  CFG['embedding_model'] = '{gan['ruta']}'")

    print("\n  REGLA DE DECISIÓN")
    print("  1) Filtro duro: debe existir AL MENOS una configuración que cumpla")
    print("     CFG['restricciones'] (outliers, n_topics, dominancia).")
    print("  2) Entre los que pasan, gana el SCORE más alto — el mismo score")
    print("     multicriterio compartido por Grid/Final, así el criterio es uno solo en")
    print("     todo el pipeline.")
    print("  3) sim_media NO decide: es diagnóstico. Un valor >0.80 explica POR")
    print("     QUÉ un modelo falla (espacio anisotrópico), no lo descalifica.")
    print("  4) Si dos modelos quedan a menos de ~0.01 de score, desempata por")
    print("     NPMI y separación, que son las menos sensibles al vocabulario.")
    print(f"\n  Excel -> {xl}")
    print(f"  Tiempo total: {perf_counter()-t_total:.1f}s")
    print("\n── SIGUIENTE PASO ──")
    print("  Copia el modelo ganador a CFG['embedding_model'] en config.py")
    print("  y ejecuta Script02_Grid_BERTopic_v3.py")
    print("=" * 66)
