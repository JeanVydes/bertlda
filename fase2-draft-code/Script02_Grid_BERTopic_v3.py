# -*- coding: utf-8 -*-
"""
Script02_Grid_BERTopic_v3.py
=======================
BERTopic: exploración UMAP + grilla HDBSCAN coarse-to-fine. NO construye el modelo final.

PRERREQUISITO
  03_Checkpoints/ck_corpus_base.pkl   (Script00)

CAMBIOS FRENTE A v2
-------------------
1. docs_tok y el Dictionary de coherencia se calculan UNA SOLA VEZ.
   En v2 se reconstruían dentro de calcular_metricas_bertopic(), es decir
   en cada una de las ~49 combinaciones del grid.
2. UMAP y HDBSCAN en CPU con random_state -> resultados deterministas.
   cuML no fija semilla y no es reproducible; queda como opción explícita.
3. La proyección UMAP de cada configuración se CACHEA durante la etapa 1.
   En v2 la mejor configuración se reajustaba después, de modo que la
   búsqueda HDBSCAN corría sobre un espacio distinto del que justificó
   elegir esa configuración.
4. Selección en dos niveles: restricciones duras (descartan) + score
   multicriterio (puntúa). Sin bonus por número de tópicos, que imponía
   un K a priori y contradecía §11.
5. Se añaden NPMI, redundancia y separación (silhouette), ausentes en v2.
6. Etapa 2 HDBSCAN usa búsqueda coarse-to-fine: primero mcs × min_samples
   con eom/epsilon=0; método y epsilon se refinan solo en los mejores candidatos.
7. Etapa 3 de estabilidad con varias semillas (ARI/NMI).
8. Misma configuración c-TF-IDF en búsqueda y modelo final. En v2 el
   modelo final usaba reduce_frequent_words=True y la búsqueda no.
9. reduce_outliers con threshold > 0 y exportación de AMBAS asignaciones.
10. probs recalculadas tras update_topics (en v2 quedaban desincronizadas).
11. model_configuration.json con semillas, versiones y métricas.

FASES
  A  Corpus + textos
  B  Embeddings (checkpoint etiquetado con el nombre del modelo)
  C  Estructuras de evaluación (una vez)
  D  Etapa 1 — exploración UMAP
  E  Etapa 2 — HDBSCAN coarse-to-fine + exportación de la grilla
"""

import os
import re
import sys
import json
import pickle
import warnings
from copy import deepcopy
from time import perf_counter

try:
    _DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _DIR = os.getcwd()
sys.path.insert(0, _DIR)

import numpy as np
import pandas as pd

from config import (CFG, rutas, crear_rutas, montar_drive,
                    fijar_semillas, registrar_entorno, n_cores,
                    guardar_modelo_bertopic)
import procesador_v3 as pr
from bertopic_core_v3 import (
    UMAPPrecomputado, get_umap, get_hdbscan, get_vectorizers,
    calcular_metricas, cumple_restricciones, score,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

SEED = CFG['seed']


# =============================================================================
# UTILIDADES
# =============================================================================
def _fmt(s):
    m, s = divmod(s, 60)
    return f"{int(m):3d}m {s:05.2f}s"


def _ck(nombre):
    return os.path.join(rutas['check'], f'ck_{nombre}.pkl')


def _guardar(nombre, obj):
    pickle.dump(obj, open(_ck(nombre), 'wb'))
    print(f"    checkpoint -> ck_{nombre}.pkl")


def _cargar(nombre):
    p = _ck(nombre)
    return pickle.load(open(p, 'rb')) if os.path.exists(p) else None


def _slug(nombre):
    return re.sub(r'[^a-z0-9]+', '_', str(nombre).lower()).strip('_')


# Funciones BERTopic compartidas importadas desde bertopic_core_v3.py

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
    print("SCRIPT 02 — BERTopic v3")
    print(f"  SEED={SEED} | UMAP={CFG['compute']['umap_backend']} "
          f"| HDBSCAN={CFG['compute']['hdbscan_backend']}")
    print("=" * 66)

    # ── FASE A — CORPUS ──────────────────────────────────────────────────────
    t0 = perf_counter()
    print("\n── FASE A: CORPUS ──")
    base = _cargar('corpus_base')
    if base is None:
        raise FileNotFoundError(
            "ck_corpus_base.pkl no encontrado.\n"
            "Ejecuta Script00_Auditoria_Corpus.py primero.")
    df_corpus = base['df'].reset_index(drop=True)
    P = base['params']
    normalizar_mayus = base['normalizar_mayus']
    n_docs = len(df_corpus)
    print(f"  {n_docs:,} documentos [{P['escala']}]")

    docs = _cargar('docs_embeddings')
    if docs is not None and len(docs) == n_docs:
        print("  Textos recuperados de checkpoint.")
    else:
        docs = pr.procesar_embeddings(df_corpus['texto'].tolist(),
                                      normalizar_mayus=normalizar_mayus,
                                      n_cores=NC)
        _guardar('docs_embeddings', docs)
    tiempos = {'A - Corpus': perf_counter() - t0}

    # ── FASE B — EMBEDDINGS ──────────────────────────────────────────────────
    t0 = perf_counter()
    modelo_emb = CFG['embedding_model']
    slug = _slug(modelo_emb.split('/')[-1])
    print(f"\n── FASE B: EMBEDDINGS ──\n  Modelo: {modelo_emb}")

    ck_emb = _ck(f'embeddings_{slug}')
    embeddings = None
    if os.path.exists(ck_emb):
        d = pickle.load(open(ck_emb, 'rb'))
        # El checkpoint guarda el nombre del modelo: en v2 no lo hacía y
        # cambiar de modelo recargaba silenciosamente vectores antiguos.
        if d.get('modelo') == modelo_emb and d.get('n') == n_docs:
            embeddings = d['embeddings']
            print(f"  Checkpoint válido -> {embeddings.shape}")
        else:
            print("  Checkpoint de OTRO modelo o tamaño: se recalcula.")

    if embeddings is None:
        from sentence_transformers import SentenceTransformer
        dev = CFG['compute']['embeddings_device']
        if dev == 'auto':
            try:
                import torch
                dev = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                dev = 'cpu'
        print(f"  Dispositivo: {dev}")
        st = SentenceTransformer(modelo_emb, device=dev)
        st.max_seq_length = CFG['max_seq_length']

        chunk = P['chunk_size']
        parciales, ck_parcial = [], _ck(f'embeddings_parcial_{slug}')
        inicio = 0
        if os.path.exists(ck_parcial):
            d = pickle.load(open(ck_parcial, 'rb'))
            if d.get('modelo') == modelo_emb:
                parciales, inicio = d['parciales'], d['inicio']
                print(f"  Reanudando desde el documento {inicio:,}")

        for i in range(inicio, n_docs, chunk):
            e = st.encode(docs[i:i + chunk],
                          batch_size=CFG['compute']['batch_size'],
                          show_progress_bar=False, convert_to_numpy=True,
                          normalize_embeddings=CFG['normalizar_embeddings'])
            parciales.append(np.asarray(e, dtype=np.float32))
            # Checkpoint incremental: si Colab se corta no se pierde todo.
            pickle.dump({'modelo': modelo_emb, 'parciales': parciales,
                         'inicio': min(i + chunk, n_docs)},
                        open(ck_parcial, 'wb'))
            print(f"    {min(i+chunk, n_docs):,}/{n_docs:,}")

        embeddings = np.vstack(parciales)
        pickle.dump({'modelo': modelo_emb, 'n': n_docs,
                     'embeddings': embeddings}, open(ck_emb, 'wb'))
        if os.path.exists(ck_parcial):
            os.remove(ck_parcial)
        print(f"  Embeddings -> {embeddings.shape}")
    tiempos['B - Embeddings'] = perf_counter() - t0

    # ── FASE C — ESTRUCTURAS DE EVALUACIÓN (UNA SOLA VEZ) ────────────────────
    t0 = perf_counter()
    print("\n── FASE C: ESTRUCTURAS DE EVALUACIÓN ──")
    from gensim.corpora import Dictionary

    dominio = _cargar('stopwords_dominio') or set()
    stops = pr.construir_stopwords(dominio=dominio)
    print(f"  Stopwords: {len(stops):,} (dominio aplicado: {len(dominio)})")

    vcfg = CFG['vectorizer']
    # El Grid NO instancia BERTopic (clusteriza con UMAP+HDBSCAN directo), así
    # que solo necesita el vectorizador de evaluación. Se construye con la
    # MISMA función que el script final para que el espacio léxico sea idéntico.
    vectorizer_model, _vectorizer_ctfidf, min_df = get_vectorizers(
        docs, stops, n_docs)
    X_counts = vectorizer_model.transform(docs)
    vocab = np.array(vectorizer_model.get_feature_names_out())
    analyzer = vectorizer_model.build_analyzer()
    docs_tok = [analyzer(d) for d in docs]
    dict_eval = Dictionary(docs_tok)
    dict_eval.filter_extremes(no_below=2, no_above=0.9)
    print(f"  Vectorizer: {len(vocab):,} términos (min_df={min_df})")
    print(f"  Diccionario coherencia: {len(dict_eval):,}")
    print(f"  (En v2 esto se recalculaba en CADA combinación del grid.)")
    tiempos['C - Estructuras'] = perf_counter() - t0

    EV = dict(docs_tok=docs_tok, dict_eval=dict_eval,
              vectorizer=vectorizer_model, X_counts=X_counts,
              vocab=vocab, n_docs=n_docs)

    # ── FASE D — ETAPA 1: EXPLORACIÓN UMAP ───────────────────────────────────
    t0 = perf_counter()
    print("\n── FASE D: ETAPA 1 — EXPLORACIÓN UMAP ──")
    print(f"  {len(P['umap_grid'])} configuraciones | min_dist=0.0 (clustering)")

    ck_e1 = os.path.join(rutas['check_grid'], 'ck_etapa1_umap.pkl')
    res_e1 = pickle.load(open(ck_e1, 'rb')) if os.path.exists(ck_e1) else []
    hechos = {(r['n_neighbors'], r['n_components']) for r in res_e1}
    proyecciones = {}

    for cfg in P['umap_grid']:
        clave = (cfg['n_neighbors'], cfg['n_components'])
        ck_proj = os.path.join(rutas['check_grid'],
                               f"ck_proj_{clave[0]}_{clave[1]}.pkl")
        if os.path.exists(ck_proj):
            proyecciones[clave] = pickle.load(open(ck_proj, 'rb'))
        if clave in hechos:
            continue
        t_u = perf_counter()
        try:
            if clave not in proyecciones:
                red = get_umap(n_neighbors=cfg['n_neighbors'],
                               n_components=cfg['n_components'],
                               min_dist=cfg['min_dist'],
                               metric='cosine').fit_transform(embeddings)
                red = np.asarray(red, dtype=np.float32)
                # Se CACHEA para que la etapa 2 use exactamente este espacio.
                pickle.dump(red, open(ck_proj, 'wb'))
                proyecciones[clave] = red
            red = proyecciones[clave]

            cl = get_hdbscan(**P['hdbscan_ref'])
            labels = cl.fit_predict(np.asarray(red, dtype=np.float64))
            m = calcular_metricas(labels, red, **EV)
            ok, motivo = cumple_restricciones(m)
            fila = {**cfg, **m, 'valida': ok, 'motivo': motivo,
                    'score': score(m) if ok else 0.0,
                    'tiempo_s': round(perf_counter() - t_u, 1)}
            res_e1.append(fila)
            pickle.dump(res_e1, open(ck_e1, 'wb'))
            print(f"  nn={cfg['n_neighbors']:<3} nc={cfg['n_components']:<3} "
                  f"-> {m['n_topics']:>3} tóp | out {m['outlier_rate']*100:5.1f}% "
                  f"| Cv={m.get('cv',0):.3f} sep={m.get('separation',0):+.3f} "
                  f"| {'score %.4f' % fila['score'] if ok else 'DESCARTADA: '+motivo}")
        except Exception as e:
            print(f"  nn={cfg['n_neighbors']} nc={cfg['n_components']} ERROR: {e}")

    df_e1 = pd.DataFrame(res_e1)
    validas = df_e1[df_e1['valida']] if 'valida' in df_e1 else df_e1
    if validas.empty:
        print("\n  NINGUNA configuración pasó las restricciones.")
        print("  Revisa CFG['restricciones'] o el modelo de embeddings "
              "(ver Script02b).")
        validas = df_e1.sort_values('outlier_rate')
    df_e1 = df_e1.sort_values('score', ascending=False)
    df_e1.to_excel(os.path.join(rutas['res_train'],
                                'Etapa1_UMAP.xlsx'), index=False)
    # Para corpus grandes no tiene sentido propagar 3 UMAP a todo el barrido
    # HDBSCAN: cada ajuste es mucho más costoso. get_params() limita
    # adaptativamente a 3 (<30k) o 2 (>=30k), sin superar el máximo de CFG.
    n_umap_e2 = min(CFG['busqueda']['n_top_refinamiento'],
                    P.get('n_top_umap_hdbscan',
                          CFG['busqueda']['n_top_refinamiento']))
    top_umap = (validas.sort_values('score', ascending=False)
                .head(n_umap_e2))
    print(f"\n  Mejores UMAP: " + ", ".join(
        f"nn={int(r['n_neighbors'])}/nc={int(r['n_components'])}"
        for _, r in top_umap.iterrows()))
    tiempos['D - Etapa 1 UMAP'] = perf_counter() - t0

    # ── FASE E — ETAPA 2: HDBSCAN COARSE-TO-FINE ────────────────────────────
    t0 = perf_counter()
    print("\n── FASE E: ETAPA 2 — HDBSCAN COARSE-TO-FINE ──")

    # IMPORTANTE: se usan checkpoints nuevos para no mezclar resultados de la
    # antigua rejilla exhaustiva (324 combinaciones) con esta estrategia.
    ck_e2a = os.path.join(rutas['check_grid'], 'ck_etapa2a_hdbscan_coarse.pkl')
    ck_e2b = os.path.join(rutas['check_grid'], 'ck_etapa2b_hdbscan_fino.pkl')
    res_e2a = pickle.load(open(ck_e2a, 'rb')) if os.path.exists(ck_e2a) else []
    res_e2b = pickle.load(open(ck_e2b, 'rb')) if os.path.exists(ck_e2b) else []

    metodo_base = CFG['busqueda'].get('hdbscan_coarse_method', 'eom')
    eps_base = float(CFG['busqueda'].get('hdbscan_coarse_epsilon', 0.0))

    # ── E1. Barrido estructural: UMAP × mcs × min_samples ───────────────────
    hechos_a = {(r['n_neighbors'], r['n_components'], r['mcs'], r['ms'])
                for r in res_e2a}
    combos_a = []
    for _, ru in top_umap.iterrows():
        clave = (int(ru['n_neighbors']), int(ru['n_components']))
        for mcs in P['mcs_grid']:
            for ms in pr.ms_para(mcs):
                k = (*clave, mcs, ms)
                if k not in hechos_a:
                    combos_a.append(k)

    total_a = (len(top_umap) *
               sum(len(pr.ms_para(mcs)) for mcs in P['mcs_grid']))
    print(f"  E1 estructural: {total_a} combinaciones máximas "
          f"({len(combos_a)} pendientes) | {metodo_base}, eps={eps_base:.2f}")

    for (nn, nc, mcs, ms) in combos_a:
        clave = (nn, nc)
        if clave not in proyecciones:
            ck_proj = os.path.join(rutas['check_grid'], f"ck_proj_{nn}_{nc}.pkl")
            if not os.path.exists(ck_proj):
                continue
            proyecciones[clave] = pickle.load(open(ck_proj, 'rb'))
        red = proyecciones[clave]
        t_h = perf_counter()
        try:
            cl = get_hdbscan(min_cluster_size=int(mcs), min_samples=int(ms),
                             cluster_selection_method=metodo_base,
                             cluster_selection_epsilon=eps_base,
                             metric='euclidean', prediction_data=True)
            labels = cl.fit_predict(np.asarray(red, dtype=np.float64))
            m = calcular_metricas(labels, red, **EV)
            ok, motivo = cumple_restricciones(m)
            fila = {'fase_busqueda': 'coarse',
                    'n_neighbors': nn, 'n_components': nc, 'mcs': mcs,
                    'ms': ms, 'metodo': metodo_base, 'eps': eps_base, **m,
                    'valida': ok, 'motivo': motivo,
                    'score': score(m) if ok else 0.0,
                    'tiempo_s': round(perf_counter() - t_h, 1)}
            res_e2a.append(fila)
            pickle.dump(res_e2a, open(ck_e2a, 'wb'))
            if ok:
                print(f"    A nn{nn} nc{nc} mcs{mcs:<5} ms{ms:<4} "
                      f"-> {m['n_topics']:>3} tóp | "
                      f"out {m['outlier_rate']*100:5.1f}% | "
                      f"score {fila['score']:.4f}")
        except Exception as e:
            print(f"    A nn{nn} nc{nc} mcs={mcs} ms={ms} ERROR: {e}")

    df_e2a = pd.DataFrame(res_e2a)
    val_a = df_e2a[df_e2a['valida']] if 'valida' in df_e2a else df_e2a
    if val_a.empty:
        raise RuntimeError(
            "Ninguna configuración HDBSCAN del barrido estructural pasó "
            "las restricciones. Revisa CFG['restricciones'] o embeddings.")

    # ── E2. Refinamiento local: método × epsilon solo en los mejores ─────────
    n_top_fino = int(P.get('n_top_hdbscan_fino', 5))
    top_base = (val_a.sort_values('score', ascending=False)
                .head(n_top_fino))

    hechos_b = {(r['n_neighbors'], r['n_components'], r['mcs'], r['ms'],
                 r['metodo'], r['eps']) for r in res_e2b}
    combos_b = []
    for _, r in top_base.iterrows():
        nn, nc = int(r['n_neighbors']), int(r['n_components'])
        mcs, ms = int(r['mcs']), int(r['ms'])
        for metodo in P['metodos_grid']:
            for eps in P['eps_grid']:
                eps = float(eps)
                # La configuración eom/0 ya fue calculada en E1.
                if metodo == metodo_base and abs(eps - eps_base) < 1e-12:
                    continue
                k = (nn, nc, mcs, ms, metodo, eps)
                if k not in hechos_b:
                    combos_b.append(k)

    alternativas_por_base = (len(P['metodos_grid']) * len(P['eps_grid']) - 1)
    total_b = len(top_base) * max(alternativas_por_base, 0)
    print(f"  E2 refinamiento: top {len(top_base)} × "
          f"{alternativas_por_base} alternativas = {total_b} máximas "
          f"({len(combos_b)} pendientes)")

    for (nn, nc, mcs, ms, metodo, eps) in combos_b:
        clave = (nn, nc)
        if clave not in proyecciones:
            ck_proj = os.path.join(rutas['check_grid'], f"ck_proj_{nn}_{nc}.pkl")
            if not os.path.exists(ck_proj):
                continue
            proyecciones[clave] = pickle.load(open(ck_proj, 'rb'))
        red = proyecciones[clave]
        t_h = perf_counter()
        try:
            cl = get_hdbscan(min_cluster_size=int(mcs), min_samples=int(ms),
                             cluster_selection_method=metodo,
                             cluster_selection_epsilon=float(eps),
                             metric='euclidean', prediction_data=True)
            labels = cl.fit_predict(np.asarray(red, dtype=np.float64))
            m = calcular_metricas(labels, red, **EV)
            ok, motivo = cumple_restricciones(m)
            fila = {'fase_busqueda': 'fine',
                    'n_neighbors': nn, 'n_components': nc, 'mcs': mcs,
                    'ms': ms, 'metodo': metodo, 'eps': eps, **m,
                    'valida': ok, 'motivo': motivo,
                    'score': score(m) if ok else 0.0,
                    'tiempo_s': round(perf_counter() - t_h, 1)}
            res_e2b.append(fila)
            pickle.dump(res_e2b, open(ck_e2b, 'wb'))
            if ok:
                print(f"    B nn{nn} nc{nc} mcs{mcs:<5} ms{ms:<4} "
                      f"{metodo:<4} eps{eps:.2f} -> "
                      f"{m['n_topics']:>3} tóp | "
                      f"out {m['outlier_rate']*100:5.1f}% | "
                      f"score {fila['score']:.4f}")
        except Exception as e:
            print(f"    B nn{nn} nc{nc} mcs={mcs} ms={ms} "
                  f"{metodo} eps={eps} ERROR: {e}")

    # Las configuraciones base siguen siendo candidatas; el refinamiento no las
    # reemplaza. Se combinan ambas fases y se selecciona por el mismo score.
    df_e2b = pd.DataFrame(res_e2b)
    df_e2 = pd.concat([df_e2a, df_e2b], ignore_index=True, sort=False)
    df_e2 = df_e2.sort_values('score', ascending=False)
    df_e2.to_excel(os.path.join(rutas['res_train'],
                                'Etapa2_HDBSCAN.xlsx'), index=False)

    val2 = df_e2[df_e2['valida']] if 'valida' in df_e2 else df_e2
    if val2.empty:
        raise RuntimeError(
            "Ninguna configuración HDBSCAN pasó las restricciones.\n"
            "Opciones: (a) relajar CFG['restricciones'], "
            "(b) cambiar el modelo de embeddings (Script02b).")
    val2 = val2.sort_values('score', ascending=False)

    print(f"  Total evaluado en etapa 2: {len(df_e2):,} configuraciones "
          f"({len(df_e2a):,} coarse + {len(df_e2b):,} fine)")
    print(f"  Mejor score etapa 2: {val2.iloc[0]['score']:.5f}")
    tiempos['E - Etapa 2 HDBSCAN'] = perf_counter() - t0


    # ── FIN DEL SCRIPT DE GRILLA ─────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("SCRIPT 02 GRID COMPLETADO")
    print(f"  Configuraciones HDBSCAN evaluadas: {len(df_e2):,}")
    print(f"  Configuraciones válidas: {len(val2):,}")
    print(f"  Mejor score observado: {val2.iloc[0]['score']:.5f}")
    print("\n  NO se ha construido ningún modelo final.")
    print("  Revise: 02_Resultados_Entrenamiento/Etapa2_HDBSCAN.xlsx")
    print("  Seleccione manualmente una fila y copie sus parámetros en")
    print("  CFG['seleccion_bertopic'] de config.py.")
    print("  Después ejecute: Script02_Final_BERTopic_v3.py")
    print("=" * 66)
