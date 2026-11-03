# -*- coding: utf-8 -*-
"""
Script03_Viz_v3.py
===================
Visualizaciones para LDA y BERTopic. Funciona con cualquiera de los dos,
o con ambos si existen sus resultados.

CAMBIOS FRENTE A v2
-------------------
1. PNG + SVG + PDF (v2 solo exportaba PNG; §13/§14/§20 piden los tres).
2. La proyección UMAP 2D de visualización se calcula UNA vez y se cachea.
   (v2 la recalculaba hasta 3 veces: panel UMAP, embeddings 2D, 3D.)
3. topics_over_time ya NO convierte años faltantes en año 0. Se usa la
   columna 'anio' ya limpia (NaN real) que exporta Script00 / Script02,
   y se excluye el último año por estar típicamente incompleto.
4. nr_bins fijo (10) sustituido por proporción ANUAL EXACTA — mismo cálculo
   que usa Script05, así ambas salidas son consistentes entre sí.
5. Heatmap tópico × país usando Country_normalized (Script00), en vez de
   nombres de país sin normalizar.
6. Se elimina visualize_distribution sobre un único documento (no aporta
   nada y no es una elección editorial defendible).
7. Si existe Trend_Analysis_*.xlsx (Script05), el heatmap tópico × año
   anota la clasificación de tendencia (↑/↓/→) de cada fila.

NOTA SOBRE LDA
  Los scripts LDA (01/01b/01c) todavía no se han migrado para consumir el
  checkpoint de Script00 (doc_id, Country_normalized). Mientras eso no
  ocurra, el heatmap tópico × país de LDA se omite automáticamente si el
  Excel de LDA no trae una columna de país reconocible, con un aviso claro
  en el log — no se inventa una columna ni se asume alineación posicional.

PRERREQUISITOS (al menos uno)
  01_Modelos_Entrenados/BERTopic/bertopic_final/  +  RESULTADOS_BERTOPIC_COMPLETO.xlsx
  01_Modelos_Entrenados/LDA/lda_final.model       +  RESULTADOS_LDA_COMPLETO.xlsx

SALIDAS (en 06_Visualizaciones/, formatos según CFG['viz']['formatos'])
  LDA_Similitud_{Coseno,Hellinger,JensenShannon}.*
  LDA_TopWords_Heatmap.*
  LDA_Topico_x_Anio.*
  LDA_Topico_x_Pais.*                (si hay columna de país)
  BERTopic_TopWords_Heatmap.*
  BERTopic_Topico_x_Anio.*
  BERTopic_Topico_x_Pais.*
  BERTopic_UMAP_Scatter.*
  BERTopic_UMAP_Scatter_3D.*
  BERTopic_Intertopic_Map.html       (nativo, interactivo)
  BERTopic_Jerarquia.html            (nativo, interactivo)
  BERTopic_Similitud_Heatmap.html    (nativo, interactivo)
"""

import os
import sys
import json
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registra projection='3d')
from matplotlib.colors import LinearSegmentedColormap

from config import CFG, rutas, crear_rutas, montar_drive

warnings.filterwarnings("ignore")

DPI = CFG['viz']['dpi']
FORMATOS = CFG['viz']['formatos']
TOP_WORDS = CFG['viz']['top_words']
MAX_TOPICS_HEATMAP = CFG['viz']['max_topics_heatmap']
TOP_PAISES = CFG['viz']['top_paises']
MIN_DOCS_ANIO = 20          # mismo umbral que Script05, por consistencia
EXCLUIR_ULTIMO_ANIO = True

CMAP_SIM = LinearSegmentedColormap.from_list('sim', ['#FFFFFF', '#2166AC'])
CMAP_HEAT = LinearSegmentedColormap.from_list('heat', ['#FFFFFF', '#B2182B'])


# =============================================================================
# UTILIDADES COMUNES
# =============================================================================
def _guardar_fig(fig, nombre, dpi=DPI):
    rutas_out = []
    for fmt in FORMATOS:
        p = os.path.join(rutas['viz'], f"{nombre}.{fmt}")
        fig.savefig(p, dpi=dpi, bbox_inches='tight')
        rutas_out.append(p)
    plt.close(fig)
    print(f"    -> {nombre}.{{{','.join(FORMATOS)}}}")
    return rutas_out


def _fmt(s):
    m, s = divmod(s, 60)
    return f"{int(m):3d}m {s:05.2f}s"


def cargar_tendencia(modelo):
    """Carga la clasificación de tendencia de Script05, si existe."""
    p = os.path.join(rutas['base'], '04_Resultados_Finales', 'Tendencias',
                     f'Trend_Analysis_{modelo}.xlsx')
    if not os.path.exists(p):
        return None
    df = pd.read_excel(p, sheet_name='Trend_Analysis')
    return dict(zip(df['Topic_ID'], df['Trend']))


def _simbolo_tendencia(t):
    if t is None:
        return ''
    if 'Increasing' in t:
        return ' ↑'
    if 'Declining' in t:
        return ' ↓'
    return ' →'


# =============================================================================
# TÓPICO × AÑO  (compartido LDA / BERTopic)
# =============================================================================
def heatmap_topico_anio(df, col_topic, col_anio, etiquetas, nombre_salida,
                        titulo, tendencia=None):
    """
    Proporción anual por tópico. Mismo cálculo que Script05 (sin nr_bins
    arbitrario, sin convertir años faltantes en 0).
    """
    d = df[[col_anio, col_topic]].dropna()
    d = d[d[col_topic] != -1]
    d[col_anio] = d[col_anio].astype(int)

    total = d.groupby(col_anio).size()
    validos = total[total >= MIN_DOCS_ANIO].index.tolist()
    if EXCLUIR_ULTIMO_ANIO and validos:
        validos = [a for a in validos if a != max(total.index)]
    if len(validos) < 3:
        print(f"    Omitido: solo {len(validos)} años con >= {MIN_DOCS_ANIO} "
              f"documentos.")
        return None

    tabla = (d[d[col_anio].isin(validos)]
             .groupby([col_anio, col_topic]).size()
             .unstack(fill_value=0))
    tabla = tabla.reindex(validos, fill_value=0)
    prop = tabla.div(tabla.sum(axis=1), axis=0) * 100

    # Ordenar tópicos por prevalencia media, limitar para legibilidad
    orden = prop.mean().sort_values(ascending=False).index
    prop = prop[orden[:MAX_TOPICS_HEATMAP]]

    filas = [str(etiquetas.get(t, t))[:38] +
            _simbolo_tendencia(tendencia.get(t) if tendencia else None)
            for t in prop.columns]

    fig, ax = plt.subplots(figsize=(max(8, len(validos) * 0.55),
                                    max(4, len(filas) * 0.32)))
    im = ax.imshow(prop.T.values, aspect='auto', cmap=CMAP_HEAT)
    ax.set_xticks(range(len(validos)))
    ax.set_xticklabels(validos, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(filas)))
    ax.set_yticklabels(filas, fontsize=8)
    ax.set_xlabel('Year')
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label('Annual prevalence (%)', fontsize=9)
    fig.tight_layout()
    _guardar_fig(fig, nombre_salida)
    return prop


# =============================================================================
# TÓPICO × PAÍS  (compartido LDA / BERTopic)
# =============================================================================
def heatmap_topico_pais(df, col_topic, col_pais, etiquetas, nombre_salida,
                        titulo, top_n=TOP_PAISES):
    d = df[[col_pais, col_topic]].dropna()
    d = d[d[col_topic] != -1]
    top_paises = d[col_pais].value_counts().head(top_n).index.tolist()
    d = d[d[col_pais].isin(top_paises)]
    if d.empty:
        print("    Omitido: sin datos de país suficientes.")
        return None

    tabla = d.groupby([col_pais, col_topic]).size().unstack(fill_value=0)
    prop = tabla.div(tabla.sum(axis=1), axis=0) * 100
    prop = prop.reindex(top_paises)

    orden = prop.mean().sort_values(ascending=False).index
    prop = prop[orden[:MAX_TOPICS_HEATMAP]]
    cols = [str(etiquetas.get(t, t))[:30] for t in prop.columns]

    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 0.5),
                                    max(4, len(top_paises) * 0.35)))
    im = ax.imshow(prop.values, aspect='auto', cmap=CMAP_HEAT)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(top_paises)))
    ax.set_yticklabels(top_paises, fontsize=9)
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label("Share of country's documents (%)", fontsize=9)
    fig.tight_layout()
    _guardar_fig(fig, nombre_salida)
    return prop


# =============================================================================
# LDA
# =============================================================================
def _distancias_lda(topics_matrix):
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import entropy

    cos = 1 - squareform(pdist(topics_matrix, metric='cosine'))

    def hellinger(p, q):
        return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))
    K = topics_matrix.shape[0]
    hell = np.zeros((K, K))
    js = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            hell[i, j] = 1 - hellinger(topics_matrix[i], topics_matrix[j])
            m = 0.5 * (topics_matrix[i] + topics_matrix[j])
            jsd = 0.5 * entropy(topics_matrix[i], m) + 0.5 * entropy(topics_matrix[j], m)
            js[i, j] = 1 - np.clip(jsd, 0, 1)
    np.fill_diagonal(hell, 1); np.fill_diagonal(js, 1)
    return cos, hell, js


def viz_lda_similitudes(lda_model):
    print("  Similitudes entre tópicos LDA...")
    K = lda_model.num_topics
    labels = [f"T{i+1}" for i in range(K)]
    tm = lda_model.get_topics()
    cos, hell, js = _distancias_lda(tm)
    for nombre, mat in [
        ('LDA_Similitud_Coseno', cos),
        ('LDA_Similitud_Hellinger', hell),
        ('LDA_Similitud_JensenShannon', js),
    ]:
        fig, ax = plt.subplots(figsize=(max(6, K * 0.45), max(5, K * 0.4)))
        im = ax.imshow(mat, cmap=CMAP_SIM, vmin=0, vmax=1)
        ax.set_xticks(range(K)); ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticks(range(K)); ax.set_yticklabels(labels, fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.75)
        fig.tight_layout()
        _guardar_fig(fig, nombre)
    return cos


def viz_lda_topwords(lda_model, top_n=TOP_WORDS):
    print("  Heatmap de palabras clave LDA...")
    K = lda_model.num_topics
    todas = set()
    filas = {}
    for i in range(K):
        top = lda_model.show_topic(i, topn=top_n)
        filas[i] = dict(top)
        todas |= {w for w, _ in top}
    vocab = sorted(todas)
    M = np.array([[filas[i].get(w, 0) for w in vocab] for i in range(K)])

    fig, ax = plt.subplots(figsize=(max(10, len(vocab) * 0.22), max(5, K * 0.4)))
    im = ax.imshow(M, aspect='auto', cmap=CMAP_HEAT)
    ax.set_xticks(range(len(vocab)))
    ax.set_xticklabels(vocab, rotation=90, fontsize=6)
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"T{i+1}" for i in range(K)], fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.7, label='Probability')
    fig.tight_layout()
    _guardar_fig(fig, 'LDA_TopWords_Heatmap')


def procesar_lda():
    ruta_mod = os.path.join(rutas['mod_lda'], 'lda_final.model')
    ruta_xl = os.path.join(rutas['res_lda'], 'RESULTADOS_LDA_COMPLETO.xlsx')
    if not (os.path.exists(ruta_mod) and os.path.exists(ruta_xl)):
        print("  No hay resultados de LDA completos; se omite.")
        return []

    from gensim.models.ldamodel import LdaModel
    from gensim.models.ldamulticore import LdaMulticore
    try:
        lda = LdaMulticore.load(ruta_mod)
    except Exception:
        lda = LdaModel.load(ruta_mod)
    print(f"  Modelo LDA cargado: K={lda.num_topics}")

    generados = []
    try:
        viz_lda_similitudes(lda)
        generados += ['LDA_Similitud_Coseno', 'LDA_Similitud_Hellinger',
                      'LDA_Similitud_JensenShannon']
    except Exception as e:
        print(f"  Similitudes LDA fallaron: {e}")

    try:
        viz_lda_topwords(lda)
        generados.append('LDA_TopWords_Heatmap')
    except Exception as e:
        print(f"  TopWords LDA falló: {e}")

    theta = pd.read_excel(ruta_xl, sheet_name='Matrix_Theta_Asignacion')
    etiquetas = {}
    try:
        res = pd.read_excel(ruta_xl, sheet_name='Resumen_Geografico')
        for _, r in res.iterrows():
            kw = str(r.get('Keywords_20', '')).split(',')[:3]
            etiquetas[int(r['Topic_ID'])] = ", ".join(k.strip() for k in kw)
    except Exception:
        pass

    col_anio = 'anio' if 'anio' in theta.columns else CFG['col_anio']
    if col_anio in theta.columns and 'Dominant_LDA_Topic' in theta.columns:
        print("  Tópico × Año (LDA)...")
        tend = cargar_tendencia('LDA')
        try:
            r = heatmap_topico_anio(theta, 'Dominant_LDA_Topic', col_anio,
                                    etiquetas, 'LDA_Topico_x_Anio',
                                    'Prevalencia anual por tópico (LDA)',
                                    tendencia=tend)
            if r is not None:
                generados.append('LDA_Topico_x_Anio')
        except Exception as e:
            print(f"  Tópico×Año LDA falló: {e}")
    else:
        print(f"  Sin columna de año utilizable ('{col_anio}'); se omite "
              f"Tópico×Año.")

    col_pais = None
    for cand in ['Country_normalized', CFG['col_pais']]:
        if cand in theta.columns:
            col_pais = cand
            break
    if col_pais and 'Dominant_LDA_Topic' in theta.columns:
        print(f"  Tópico × País (LDA, columna '{col_pais}')...")
        try:
            r = heatmap_topico_pais(theta, 'Dominant_LDA_Topic', col_pais,
                                    etiquetas, 'LDA_Topico_x_Pais',
                                    'Distribución de tópicos por país (LDA)')
            if r is not None:
                generados.append('LDA_Topico_x_Pais')
        except Exception as e:
            print(f"  Tópico×País LDA falló: {e}")
    else:
        print("  LDA aún no exporta país por documento (pendiente migrar "
              "Script01/01b/01c a v3, ver auditoría A.1.5). Se omite "
              "Tópico×País para LDA.")

    return generados


# =============================================================================
# BERTOPIC
# =============================================================================
def viz_bert_topwords(info, top_n=TOP_WORDS, max_topics=MAX_TOPICS_HEATMAP):
    print("  Heatmap de palabras clave BERTopic...")
    val = info[info['Topic'] != -1].sort_values('Count', ascending=False)
    val = val.head(max_topics)
    todas = set()
    filas = {}
    for _, r in val.iterrows():
        rep = r['Representation']
        pal = rep if isinstance(rep, list) else \
            str(rep).strip("[]").replace("'", "").split(",")
        pal = [p.strip() for p in pal if p.strip()][:top_n]
        filas[r['Topic']] = pal
        todas |= set(pal)
    vocab = sorted(todas)
    M = np.array([[1.0 if w in filas[t] else 0.0 for w in vocab]
                  for t in val['Topic']])
    nombres = [str(n)[:38] for n in val['Name']]

    fig, ax = plt.subplots(figsize=(max(10, len(vocab) * 0.18),
                                    max(5, len(nombres) * 0.32)))
    ax.imshow(M, aspect='auto', cmap=CMAP_HEAT)
    ax.set_xticks(range(len(vocab)))
    ax.set_xticklabels(vocab, rotation=90, fontsize=6)
    ax.set_yticks(range(len(nombres)))
    ax.set_yticklabels(nombres, fontsize=8)
    fig.tight_layout()
    _guardar_fig(fig, 'BERTopic_TopWords_Heatmap')


N_LEGEND_TOPICS = 20   # tope de entradas individuales en el legend (legibilidad)


def _etiqueta_corta(t, etiquetas, max_len=30):
    if etiquetas and t in etiquetas:
        return str(etiquetas[t])[:max_len]
    return f"Topic {t}"


def _colores_por_topico(topics_validos_ordenados):
    """Un color 'tab20' por tópico, ciclando si hay más de 20."""
    cmap = plt.get_cmap('tab20', 20)
    return {t: cmap(i % 20) for i, t in enumerate(topics_validos_ordenados)}


def _legend_topicos(ax, color_map, validos_por_tamano, etiquetas):
    """
    Legend con parches de color + nombre de tópico (inglés, truncado).
    Se limita a los N_LEGEND_TOPICS tópicos más grandes para que siga
    siendo legible; el resto queda representado solo por color en el
    gráfico (igual que hacía v2, que también topaba en 20).
    """
    top = validos_por_tamano[:N_LEGEND_TOPICS]
    handles = [mpatches.Patch(color=color_map[t],
                              label=_etiqueta_corta(t, etiquetas))
              for t in top]
    handles.append(mpatches.Patch(color='#D9D9D9', label='Outliers'))
    ax.legend(handles=handles, fontsize=6.5, ncol=1, loc='center left',
             bbox_to_anchor=(1.01, 0.5), framealpha=0.9, markerscale=1,
             borderaxespad=0.)
    if len(validos_por_tamano) > N_LEGEND_TOPICS:
        print(f"    Legend limited to the {N_LEGEND_TOPICS} largest "
              f"topics ({len(validos_por_tamano)} total in the plot).")


def viz_bert_umap_scatter(embeddings, topics, P, etiquetas=None):
    """
    Proyección 2D SOLO para visualizar (min_dist>0), separada del espacio
    de clustering. Se cachea: si ya existe no se recalcula (v2 la
    recalculaba hasta 3 veces).
    """
    ck = os.path.join(rutas['check'], 'ck_umap_2d_viz.pkl')
    if os.path.exists(ck):
        red2d = pickle.load(open(ck, 'rb'))
        print("  Proyección 2D de visualización recuperada de checkpoint.")
    else:
        print("  Calculando proyección 2D de visualización (una sola vez)...")
        from umap import UMAP
        cfg = P.get('umap_viz', dict(n_neighbors=15, n_components=2, min_dist=0.10))
        red2d = UMAP(random_state=CFG['seed'], metric='cosine',
                     **cfg).fit_transform(embeddings)
        pickle.dump(red2d, open(ck, 'wb'))

    topics = np.asarray(topics)
    out = topics == -1
    validos, counts = np.unique(topics[~out], return_counts=True)
    validos_por_tamano = validos[np.argsort(-counts)].tolist()
    color_map = _colores_por_topico(validos_por_tamano)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(red2d[out, 0], red2d[out, 1], s=5, c='#D9D9D9',
              alpha=0.4, linewidths=0)
    for t in validos_por_tamano:
        m = topics == t
        ax.scatter(red2d[m, 0], red2d[m, 1], s=6, color=color_map[t],
                  alpha=0.7, linewidths=0)

    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    _legend_topicos(ax, color_map, validos_por_tamano, etiquetas)
    fig.tight_layout()
    _guardar_fig(fig, 'BERTopic_UMAP_Scatter')


def viz_bert_umap_scatter_3d(embeddings, topics, P, etiquetas=None):
    """
    Igual que viz_bert_umap_scatter pero en 3D (proyección UMAP independiente,
    también cacheada). Recupera la visualización 3D que existía en v2 y que
    v3 había dejado de generar.
    """
    ck = os.path.join(rutas['check'], 'ck_umap_3d_viz.pkl')
    if os.path.exists(ck):
        red3d = pickle.load(open(ck, 'rb'))
        print("  Proyección 3D de visualización recuperada de checkpoint.")
    else:
        print("  Calculando proyección 3D de visualización (una sola vez)...")
        from umap import UMAP
        cfg = P.get('umap_viz_3d', dict(n_neighbors=15, n_components=3, min_dist=0.10))
        red3d = UMAP(random_state=CFG['seed'], metric='cosine',
                     **cfg).fit_transform(embeddings)
        pickle.dump(red3d, open(ck, 'wb'))

    topics = np.asarray(topics)
    out = topics == -1
    validos, counts = np.unique(topics[~out], return_counts=True)
    validos_por_tamano = validos[np.argsort(-counts)].tolist()
    color_map = _colores_por_topico(validos_por_tamano)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(red3d[out, 0], red3d[out, 1], red3d[out, 2], s=5,
              c='#D9D9D9', alpha=0.4, linewidths=0)
    for t in validos_por_tamano:
        m = topics == t
        ax.scatter(red3d[m, 0], red3d[m, 1], red3d[m, 2], s=6,
                  color=color_map[t], alpha=0.7, linewidths=0)

    ax.set_xlabel('UMAP 1', labelpad=6)
    ax.set_ylabel('UMAP 2', labelpad=6)
    ax.set_zlabel('UMAP 3', labelpad=6)
    ax.tick_params(labelsize=7)
    _legend_topicos(ax, color_map, validos_por_tamano, etiquetas)
    fig.tight_layout()
    _guardar_fig(fig, 'BERTopic_UMAP_Scatter_3D')


def viz_bert_nativas(topic_model, docs):
    """Visualizaciones interactivas nativas de BERTopic (HTML)."""
    generados = []
    intentos = [
        ('BERTopic_Intertopic_Map', lambda: topic_model.visualize_topics()),
        ('BERTopic_Jerarquia', lambda: topic_model.visualize_hierarchy()),
        ('BERTopic_Similitud_Heatmap', lambda: topic_model.visualize_heatmap()),
        ('BERTopic_Barchart', lambda: topic_model.visualize_barchart(
            top_n_topics=MAX_TOPICS_HEATMAP)),
    ]
    for nombre, fn in intentos:
        try:
            fig = fn()
            p = os.path.join(rutas['viz_bt'], f"{nombre}.html")
            fig.write_html(p)
            print(f"    -> {nombre}.html")
            generados.append(nombre)
        except Exception as e:
            print(f"  {nombre} falló: {e}")
    return generados


def procesar_bertopic():
    ruta_mod = os.path.join(rutas['mod_bert'], 'bertopic_final')
    ruta_xl = os.path.join(rutas['res_bert'], 'RESULTADOS_BERTOPIC_COMPLETO.xlsx')
    if not (os.path.exists(ruta_mod) and os.path.exists(ruta_xl)):
        print("  No hay resultados de BERTopic completos; se omite.")
        return []

    from bertopic import BERTopic
    topic_model = BERTopic.load(ruta_mod)
    info = pd.read_excel(ruta_xl, sheet_name='Topic_Info')
    docs_df = pd.read_excel(ruta_xl, sheet_name='Doc_Topic_Assignment')
    print(f"  Modelo BERTopic cargado: {(info['Topic'] != -1).sum()} tópicos")

    generados = []
    try:
        viz_bert_topwords(info)
        generados.append('BERTopic_TopWords_Heatmap')
    except Exception as e:
        print(f"  TopWords BERTopic falló: {e}")

    etiquetas = dict(zip(info['Topic'], info['Name']))
    col_topic = 'Topic_reduced' if 'Topic_reduced' in docs_df.columns else 'Topic'

    if 'anio' in docs_df.columns:
        print("  Tópico × Año (BERTopic)...")
        tend = cargar_tendencia('BERTopic')
        try:
            r = heatmap_topico_anio(docs_df, col_topic, 'anio', etiquetas,
                                    'BERTopic_Topico_x_Anio',
                                    'Prevalencia anual por tópico (BERTopic)',
                                    tendencia=tend)
            if r is not None:
                generados.append('BERTopic_Topico_x_Anio')
        except Exception as e:
            print(f"  Tópico×Año BERTopic falló: {e}")

    if 'Country' in docs_df.columns:
        print("  Tópico × País (BERTopic)...")
        try:
            r = heatmap_topico_pais(docs_df, col_topic, 'Country', etiquetas,
                                    'BERTopic_Topico_x_Pais',
                                    'Distribución de tópicos por país (BERTopic)')
            if r is not None:
                generados.append('BERTopic_Topico_x_Pais')
        except Exception as e:
            print(f"  Tópico×País BERTopic falló: {e}")

    # UMAP scatter: requiere los embeddings del modelo elegido
    try:
        import re as _re
        slug = _re.sub(r'[^a-z0-9]+', '_',
                       CFG['embedding_model'].split('/')[-1].lower()).strip('_')
        ck_emb = os.path.join(rutas['check'], f'ck_embeddings_{slug}.pkl')
        if os.path.exists(ck_emb):
            d = pickle.load(open(ck_emb, 'rb'))
            base = pickle.load(open(os.path.join(rutas['check'],
                                                  'ck_corpus_base.pkl'), 'rb'))
            viz_bert_umap_scatter(d['embeddings'], docs_df[col_topic].values,
                                 base['params'], etiquetas=etiquetas)
            generados.append('BERTopic_UMAP_Scatter')
            viz_bert_umap_scatter_3d(d['embeddings'], docs_df[col_topic].values,
                                     base['params'], etiquetas=etiquetas)
            generados.append('BERTopic_UMAP_Scatter_3D')
        else:
            print("  Checkpoint de embeddings no encontrado; se omite el "
                 "scatter UMAP.")
    except Exception as e:
        print(f"  UMAP scatter falló: {e}")

    try:
        docs = pickle.load(open(os.path.join(rutas['check'],
                                             'ck_docs_embeddings.pkl'), 'rb'))
    except Exception:
        docs = None
    try:
        generados += viz_bert_nativas(topic_model, docs)
    except Exception as e:
        print(f"  Visualizaciones nativas fallaron: {e}")

    return generados


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':

    t0 = perf_counter()
    crear_rutas()
    montar_drive()

    print("\n" + "=" * 66)
    print("SCRIPT 03 — VISUALIZACIONES (LDA · BERTopic)  v3")
    print(f"  Formatos: {FORMATOS} | DPI: {DPI}")
    print("=" * 66)

    print("\n── LDA ──")
    gen_lda = procesar_lda()

    print("\n── BERTOPIC ──")
    gen_bt = procesar_bertopic()

    print("\n" + "=" * 66)
    print("SCRIPT 03 COMPLETADO")
    print(f"  LDA generados      : {len(gen_lda)}  {gen_lda}")
    print(f"  BERTopic generados : {len(gen_bt)}  {gen_bt}")
    if not gen_lda and not gen_bt:
        print("\n  No se generó ninguna visualización. Verifica que exista")
        print("  al menos un modelo entrenado (Script02_Final_BERTopic_v3.py o")
        print("  Script01c_ModeloFinal.py) antes de ejecutar este script.")
    print(f"  Tiempo: {perf_counter()-t0:.1f}s")
    print(f"  Carpeta: {rutas['viz']}")
    print("=" * 66)
