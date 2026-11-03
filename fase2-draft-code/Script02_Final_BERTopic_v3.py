# -*- coding: utf-8 -*-
"""
Script02_Final_BERTopic_v3.py
=======================
BERTopic FINAL a partir de una combinación seleccionada manualmente de la grilla.

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
  D  Selección manual + validación contra Etapa2_HDBSCAN.xlsx
  E  Estabilidad de la configuración seleccionada
  F  Modelo final + outliers + exportación
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
    # DOS vectorizadores: uno a nivel documento (evaluación) y otro con
    # vocabulario congelado para c-TF-IDF, que BERTopic ajusta sobre K
    # super-documentos y no sobre los n_docs. Ver bertopic_core_v3.
    vectorizer_model, vectorizer_ctfidf, min_df = get_vectorizers(
        docs, stops, n_docs)
    X_counts = vectorizer_model.transform(docs)
    vocab = np.array(vectorizer_model.get_feature_names_out())
    analyzer = vectorizer_model.build_analyzer()
    docs_tok = [analyzer(d) for d in docs]
    dict_eval = Dictionary(docs_tok)
    dict_eval.filter_extremes(no_below=2, no_above=0.9)
    print(f"  Vectorizer evaluación: {len(vocab):,} términos (min_df={min_df})")
    print(f"  Vectorizer c-TF-IDF  : vocabulario congelado ({len(vocab):,})")
    print(f"  Diccionario coherencia: {len(dict_eval):,}")
    print(f"  (En v2 esto se recalculaba en CADA combinación del grid.)")
    tiempos['C - Estructuras'] = perf_counter() - t0

    EV = dict(docs_tok=docs_tok, dict_eval=dict_eval,
              vectorizer=vectorizer_model, X_counts=X_counts,
              vocab=vocab, n_docs=n_docs)

    # ── FASE D — SELECCIÓN MANUAL Y VALIDACIÓN ─────────────────────────────
    t0 = perf_counter()
    print("\n── FASE D: SELECCIÓN MANUAL DEL MODELO ──")

    sel = CFG.get('seleccion_bertopic', {})
    requeridos = ['n_neighbors', 'n_components', 'min_cluster_size',
                  'min_samples', 'metodo', 'epsilon']
    faltan = [k for k in requeridos if sel.get(k) is None]
    if faltan:
        raise ValueError(
            "Falta completar CFG['seleccion_bertopic'] en config.py: "
            + ", ".join(faltan)
            + "\nEjecute primero Script02_Grid_BERTopic_v3.py, revise "
              "Etapa2_HDBSCAN.xlsx y copie una combinación evaluada.")

    ruta_grid = os.path.join(rutas['res_train'], 'Etapa2_HDBSCAN.xlsx')
    if not os.path.exists(ruta_grid):
        raise FileNotFoundError(
            f"No existe {ruta_grid}. Ejecute primero Script02_Grid_BERTopic_v3.py")

    df_e2 = pd.read_excel(ruta_grid)
    nn = int(sel['n_neighbors'])
    nc = int(sel['n_components'])
    mcs = int(sel['min_cluster_size'])
    ms = int(sel['min_samples'])
    metodo = str(sel['metodo']).lower().strip()
    eps = float(sel['epsilon'])

    mask = (
        (df_e2['n_neighbors'].astype(int) == nn) &
        (df_e2['n_components'].astype(int) == nc) &
        (df_e2['mcs'].astype(int) == mcs) &
        (df_e2['ms'].astype(int) == ms) &
        (df_e2['metodo'].astype(str).str.lower().str.strip() == metodo) &
        (np.isclose(df_e2['eps'].astype(float), eps, atol=1e-12))
    )
    cand = df_e2.loc[mask].copy()
    if cand.empty:
        raise ValueError(
            "La combinación manual NO aparece en Etapa2_HDBSCAN.xlsx. "
            "Copie exactamente los parámetros de una fila evaluada.")

    r = cand.sort_values('score', ascending=False).iloc[0]
    if 'valida' in cand.columns and not bool(r['valida']):
        raise ValueError(
            f"La combinación seleccionada fue descartada por: {r.get('motivo','restricciones')}.")

    print(f"  Seleccionada: nn={nn}, nc={nc}, mcs={mcs}, ms={ms}, "
          f"{metodo}, eps={eps:.2f}")
    print(f"  Grilla: {int(r['n_topics'])} tópicos | "
          f"outliers={float(r['outlier_rate'])*100:.1f}% | "
          f"Cv={float(r.get('cv',0)):.4f} | score={float(r['score']):.5f}")
    print("  La selección es MANUAL; el score se conserva como evidencia, no como regla automática.")
    tiempos['D - Selección manual'] = perf_counter() - t0

    mejor = pd.Series({
        'n_neighbors': nn, 'n_components': nc, 'mcs': mcs, 'ms': ms,
        'metodo': metodo, 'eps': eps, 'score': float(r['score'])
    })

    # ── FASE E — ESTABILIDAD DE LA CONFIGURACIÓN ELEGIDA ────────────────────
    t0 = perf_counter()
    print("\n── FASE E: ESTABILIDAD DE LA CONFIGURACIÓN ELEGIDA ──")
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from umap import UMAP

    semillas = CFG['busqueda']['semillas_estabilidad']
    etiquetas = []
    ks = []
    for s in semillas:
        try:
            red_s = UMAP(n_neighbors=nn, n_components=nc, min_dist=0.0,
                         metric='cosine', random_state=s).fit_transform(embeddings)
            cl = get_hdbscan(min_cluster_size=mcs, min_samples=ms,
                             cluster_selection_method=metodo,
                             cluster_selection_epsilon=eps,
                             metric='euclidean', prediction_data=True)
            lab = cl.fit_predict(np.asarray(red_s, dtype=np.float64))
            etiquetas.append(lab)
            ks.append(len(set(lab.tolist()) - {-1}))
        except Exception as e:
            print(f"    semilla {s} falló: {e}")

    aris, nmis = [], []
    for i in range(len(etiquetas)):
        for j in range(i + 1, len(etiquetas)):
            aris.append(adjusted_rand_score(etiquetas[i], etiquetas[j]))
            nmis.append(normalized_mutual_info_score(etiquetas[i], etiquetas[j]))

    estabilidad = {
        'ARI_medio': round(float(np.mean(aris)), 4) if aris else np.nan,
        'ARI_min': round(float(np.min(aris)), 4) if aris else np.nan,
        'NMI_medio': round(float(np.mean(nmis)), 4) if nmis else np.nan,
        'K_medio': round(float(np.mean(ks)), 2) if ks else np.nan,
        'K_sd': round(float(np.std(ks)), 2) if ks else np.nan,
    }
    print(f"  ARI={estabilidad['ARI_medio']} | NMI={estabilidad['NMI_medio']} | "
          f"K={estabilidad['K_medio']}±{estabilidad['K_sd']}")

    pd.DataFrame([{**{'n_neighbors':nn,'n_components':nc,'mcs':mcs,'ms':ms,
                       'metodo':metodo,'eps':eps,'score_grilla':float(r['score'])},
                    **estabilidad}]).to_excel(
        os.path.join(rutas['res_train'], 'Etapa3_Estabilidad_SeleccionManual.xlsx'),
        index=False)
    mejor['ARI_medio'] = estabilidad['ARI_medio']
    tiempos['E - Estabilidad manual'] = perf_counter() - t0

    # ── FASE F — MODELO FINAL ────────────────────────────────────────────────
    t0 = perf_counter()
    print("\n── FASE F: MODELO FINAL ──")
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    import scipy.sparse as sp

    def _limpiar_ctfidf(m):
        """Elimina inf/nan de una matriz c-TF-IDF (sparse o densa)."""
        if sp.issparse(m):
            m = m.tocsr()
            m.data = np.nan_to_num(m.data, nan=0.0, posinf=0.0, neginf=0.0)
            return m
        return np.nan_to_num(
            np.asarray(m, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    # Guardamos los métodos originales ANTES de parchear para no romper
    # la cadena de herencia con super() dentro de __main__.
    _ctfidf_fit_transform_orig = ClassTfidfTransformer.fit_transform
    _ctfidf_transform_orig     = ClassTfidfTransformer.transform

    def _safe_fit_transform(self, X, y=None):
        return _limpiar_ctfidf(_ctfidf_fit_transform_orig(self, X, y))

    def _safe_transform(self, X):
        return _limpiar_ctfidf(_ctfidf_transform_orig(self, X))

    ClassTfidfTransformer.fit_transform = _safe_fit_transform
    ClassTfidfTransformer.transform     = _safe_transform

    # Clase alias para mantener la firma en model_configuration.json
    _SafeClassTfidfTransformer = ClassTfidfTransformer

    def _sanear_ctfidf(model):
        """Limpia c_tf_idf_ residual tras fit_transform o update_topics."""
        if not hasattr(model, 'c_tf_idf_') or model.c_tf_idf_ is None:
            return
        m = model.c_tf_idf_
        if sp.issparse(m):
            m = m.tocsr()
            m.data = np.nan_to_num(m.data, nan=0.0, posinf=0.0, neginf=0.0)
            model.c_tf_idf_ = m
        else:
            model.c_tf_idf_ = np.nan_to_num(
                np.asarray(m, dtype=np.float64),
                nan=0.0, posinf=0.0, neginf=0.0)

    umap_model = get_umap(n_neighbors=int(mejor['n_neighbors']),
                          n_components=int(mejor['n_components']),
                          min_dist=0.0, metric='cosine')
    hdbscan_model = get_hdbscan(min_cluster_size=int(mejor['mcs']),
                                min_samples=int(mejor['ms']),
                                cluster_selection_method=mejor['metodo'],
                                cluster_selection_epsilon=float(mejor['eps']),
                                metric='euclidean', prediction_data=True)
    # MISMA configuración c-TF-IDF que en la búsqueda (bug corregido de v2).
    # Se usa _SafeClassTfidfTransformer para evitar inf/nan con vocab congelado.
    ctfidf_model = _SafeClassTfidfTransformer(
        reduce_frequent_words=CFG['ctfidf_reduce_frequent_words'])

    topic_model = BERTopic(
        umap_model=umap_model, hdbscan_model=hdbscan_model,
        vectorizer_model=deepcopy(vectorizer_ctfidf),
        ctfidf_model=ctfidf_model, language='english',
        calculate_probabilities=False,   # costoso y solo se usaba un max()
        verbose=False)

    topics_orig, _ = topic_model.fit_transform(docs, embeddings=embeddings)
    _sanear_ctfidf(topic_model)   # evita inf/nan en c_tf_idf_ con vocab congelado
    topics_orig = list(topics_orig)
    n_out_orig = int(sum(1 for t in topics_orig if t == -1))
    print(f"  Tópicos: {len(set(topics_orig) - {-1})} | "
          f"outliers: {n_out_orig} ({n_out_orig/n_docs*100:.1f}%)")

    # ── Reasignación de outliers ─────────────────────────────────────────────
    topics_red = topics_orig
    if CFG['outliers']['reasignar'] and n_out_orig > 0:
        th = CFG['outliers']['threshold']
        print(f"\n  Reasignación de outliers (threshold={th})")
        print("  v2 usaba threshold=0.0, que forzaba a reasignar el 100%.")
        try:
            topics_red = list(topic_model.reduce_outliers(
                docs, topics_orig,
                strategy=CFG['outliers']['estrategia'], threshold=th))
            topic_model.update_topics(docs, topics=topics_red,
                                      vectorizer_model=deepcopy(vectorizer_ctfidf),
                                      ctfidf_model=ctfidf_model)
            _sanear_ctfidf(topic_model)   # update_topics recalcula c_tf_idf_
            n_out_red = int(sum(1 for t in topics_red if t == -1))
            print(f"  Outliers {n_out_orig} -> {n_out_red} "
                  f"({n_out_orig - n_out_red} reasignados, "
                  f"{n_out_red/n_docs*100:.1f}% restante)")
        except Exception as e:
            print(f"  Reasignación falló ({e}); se conserva la original.")
            topics_red = topics_orig

    # Guardado verificado: escribe en disco local y copia a Drive comprobando
    # tamaño y carga real. Un save() directo sobre Drive se pierde en silencio
    # (ver docstring de config.guardar_modelo_bertopic).
    guardar_modelo_bertopic(topic_model, 'bertopic_final', modelo_emb)
    info = topic_model.get_topic_info()

    # ── Exportación ──────────────────────────────────────────────────────────
    # Se conservan AMBAS asignaciones y el doc_id estable de Script00.
    nombres = dict(zip(info['Topic'], info['Name']))
    df_docs = pd.DataFrame({
        'doc_id':          df_corpus['doc_id'].values,
        'Topic_original':  topics_orig,
        'Topic_reduced':   topics_red,
        'Topic_label':     [nombres.get(t, 'Outlier') for t in topics_red],
        'reasignado':      [a == -1 and b != -1
                            for a, b in zip(topics_orig, topics_red)],
        'anio':            df_corpus['anio'].values,
        'Country':         df_corpus['Country_normalized'].values,
        CFG['col_titulo']: df_corpus[CFG['col_titulo']].values,
    })

    ruta_xl = os.path.join(rutas['res_bert'], 'RESULTADOS_BERTOPIC_COMPLETO.xlsx')
    with pd.ExcelWriter(ruta_xl, engine='xlsxwriter') as w:
        info.to_excel(w, sheet_name='Topic_Info', index=False)
        df_docs.to_excel(w, sheet_name='Doc_Topic_Assignment', index=False)
        df_e2.sort_values('score', ascending=False).to_excel(
            w, sheet_name='Etapa2_HDBSCAN', index=False)
        pd.DataFrame([{**{'n_neighbors':nn,'n_components':nc,'mcs':mcs,'ms':ms,
                           'metodo':metodo,'eps':eps,'score_grilla':float(r['score'])},
                        **estabilidad}]).to_excel(
            w, sheet_name='Estabilidad_Seleccion', index=False)
    print(f"  Excel -> {ruta_xl}")

    # Ranking para RIF (usa la asignación reducida; se documenta cuál)
    val = info[info['Topic'] != -1].copy()
    val['etiqueta'] = val.apply(
        lambda r: f"BT{r['Topic']}: " + ", ".join(
            str(p).strip() for p in
            (r['Representation'] if isinstance(r['Representation'], list)
             else str(r['Representation']).strip("[]").replace("'", "").split(","))[:3]),
        axis=1)
    rk = val[['etiqueta', 'Count']].copy()
    rk.columns = ['etiqueta', 'frecuencia']
    rk = rk.sort_values('frecuencia', ascending=False).reset_index(drop=True)
    rk['rank'] = range(1, len(rk) + 1)
    rk.to_excel(os.path.join(rutas['res_bert'],
                             'BERTopic_Ranking_para_RIF.xlsx'), index=False)

    _guardar('bert_topics', topics_red)
    _guardar('bert_topics_original', topics_orig)

    # ── model_configuration.json (§26) ───────────────────────────────────────
    cfg_json = {
        'fecha': registrar_entorno()['fecha'],
        'entorno': registrar_entorno(),
        'corpus': {'n_docs': n_docs, 'escala': P['escala'],
                   'normalizacion_casing': normalizar_mayus},
        'embedding_model': modelo_emb,
        'umap': {'n_neighbors': int(mejor['n_neighbors']),
                 'n_components': int(mejor['n_components']),
                 'min_dist': 0.0, 'metric': 'cosine',
                 'backend': CFG['compute']['umap_backend'],
                 'random_state': SEED},
        'hdbscan': {'min_cluster_size': int(mejor['mcs']),
                    'min_samples': int(mejor['ms']),
                    'cluster_selection_method': mejor['metodo'],
                    'cluster_selection_epsilon': float(mejor['eps']),
                    'backend': CFG['compute']['hdbscan_backend']},
        'vectorizer': {'ngram_range': list(vcfg['ngram_range']),
                       'min_df': min_df, 'n_stopwords': len(stops),
                       'n_terminos': int(len(vocab)),
                       'ctfidf_vocabulario': 'congelado desde el vectorizer '
                                             'de evaluación (nivel documento)'},
        'ctfidf': {'reduce_frequent_words': CFG['ctfidf_reduce_frequent_words']},
        'restricciones': CFG['restricciones'],
        'pesos_score': CFG['pesos_score'],
        'outliers': {**CFG['outliers'],
                     'n_original': n_out_orig,
                     'n_final': int(sum(1 for t in topics_red if t == -1))},
        'seleccion': {'modo': 'manual', 'score_grilla': float(r['score'])},
        'estabilidad': {'semillas': semillas,
                        'ARI_medio': float(estabilidad['ARI_medio'])
                        if pd.notna(estabilidad['ARI_medio']) else None},
        'n_topics_final': int(len(set(topics_red) - {-1})),
    }
    ruta_json = os.path.join(rutas['res_bert'], 'model_configuration.json')
    json.dump(cfg_json, open(ruta_json, 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print(f"  model_configuration.json -> {ruta_json}")
    tiempos['F - Modelo final'] = perf_counter() - t0

    # ── VERIFICACIÓN DE DETERMINISMO ─────────────────────────────────────────
    print("\n── VERIFICACIÓN DE DETERMINISMO ──")
    if CFG['compute']['umap_backend'] == 'cpu':
        from umap import UMAP
        from sklearn.metrics import adjusted_rand_score as _ari
        r2 = UMAP(n_neighbors=int(mejor['n_neighbors']),
                  n_components=int(mejor['n_components']), min_dist=0.0,
                  metric='cosine', random_state=SEED).fit_transform(embeddings)
        l2 = get_hdbscan(min_cluster_size=int(mejor['mcs']),
                         min_samples=int(mejor['ms']),
                         cluster_selection_method=mejor['metodo'],
                         cluster_selection_epsilon=float(mejor['eps']),
                         metric='euclidean', prediction_data=True
                         ).fit_predict(np.asarray(r2, dtype=np.float64))
        a = _ari(topics_orig, l2)
        print(f"  ARI entre dos ajustes con la misma semilla: {a:.4f}")
        print("  OK: pipeline determinista." if a > 0.999 else
              "  AVISO: no determinista — revisa el backend.")
    else:
        print("  Backend cuML: determinismo NO garantizado.")

    # ── RESUMEN ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("SCRIPT 02 COMPLETADO")
    print(f"  Tópicos: {cfg_json['n_topics_final']} | "
          f"outliers finales: {cfg_json['outliers']['n_final']}")
    for k, v in tiempos.items():
        print(f"  {k:<28} {_fmt(v):>12}")
    print(f"  {'TOTAL':<28} {_fmt(perf_counter()-t_total):>12}")
    print("\n── SIGUIENTE PASO ──")
    print("  Script03_Viz_v3.py  |  Script04_RIF_v3.py")
    print("=" * 66)
