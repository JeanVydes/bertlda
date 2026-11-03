# -*- coding: utf-8 -*-
"""
procesador_v3.py
================
Módulo utilitario compartido del pipeline (LDA + BERTopic).
NO ejecutar directamente. Importar desde los scripts numerados.

CAMBIOS FRENTE A procesador_v2.py
---------------------------------
1. limpiar_para_lda: LEMATIZA y DESPUÉS filtra stopwords.
   En v2 el orden era inverso, así que 'studies' esquivaba el filtro de
   'study' y luego se lematizaba: el término entraba igualmente al corpus.
2. Se preservan dígitos y guiones internos: 'covid-19', 'co2', 'pm2.5'
   ya no se destruyen.
3. Umbral de longitud >=3 (antes >3) con lista blanca de siglas científicas.
4. RE_COPYRIGHT menos agresivo: ya no borra la oración completa que
   contenga 'open access' o 'creative commons'.
5. Stopwords en 3 capas explícitas y auditables (funcional / editorial /
   dominio empírico). Se elimina ENGLISH_STOP_WORDS de sklearn.
6. get_params(n_docs): parámetros como FUNCIÓN CONTINUA del tamaño del
   corpus. Los perfiles discretos MICRO/SMALL/MEDIUM/LARGE producían
   saltos absurdos (4.999 docs -> mcs<=180; 5.001 docs -> mcs<=300) y
   MICRO traía hdbscan_ref con min_cluster_size=3, que degeneraba el
   barrido UMAP completo. Las etiquetas se conservan solo como
   descriptor para los informes.
7. normalizar_casing(): corrige corpus exportados en MAYÚSCULAS.
"""

import re
import math
import multiprocessing

import nltk
from nltk.stem import WordNetLemmatizer
from joblib import Parallel, delayed

# =============================================================================
# 1. RECURSOS NLTK
# =============================================================================
def preparar_recursos_nltk():
    for nombre, ruta in {
        'wordnet':   'corpora/wordnet',
        'omw-1.4':   'corpora/omw-1.4',
        'stopwords': 'corpora/stopwords',
    }.items():
        try:
            nltk.data.find(ruta)
        except LookupError:
            try:
                nltk.download(nombre, quiet=True)
            except Exception:
                pass

preparar_recursos_nltk()
_lemmatizer = WordNetLemmatizer()

# =============================================================================
# 2. REGEX COMPILADOS
# =============================================================================
RE_URL   = re.compile(r'https?://\S+|www\.\S+')
RE_EMAIL = re.compile(r'\S+@\S+')
RE_SPACES = re.compile(r'\s+')

# Citas entre corchetes. Se conserva '(...1234...)' porque borraba datos
# legítimos del tipo "(n = 2400 muestras)" o "(1990-2024)".
RE_CITATION = re.compile(r'\[\s*\d+(?:\s*[,-]\s*\d+)*\s*\]')

# Boilerplate editorial. Frente a v2 se acotan los patrones: solo se
# elimina el fragmento final de copyright, no cualquier oración que
# mencione 'open access' o 'creative commons' (que en un corpus sobre
# ciencia abierta o políticas editoriales sería contenido real).
RE_COPYRIGHT = re.compile(
    r'(?:\(\s*c\s*\)|©)\s*\d{4}[^.]*?(?:\.|$)'
    r'|all\s+rights?\s+reserved[^.]*?(?:\.|$)'
    r'|exclusive\s+licen[sc]e\s+to[^.]*?(?:\.|$)'
    r'|published\s+by\s+(?:elsevier|springer|wiley|taylor|informa|mdpi|'
    r'oxford|cambridge|sage|emerald|frontiers)[^.]*?(?:\.|$)'
    r'|on\s+behalf\s+of\s+[^.]{0,80}?(?:\.|$)'
    r'|this\s+is\s+an\s+open\s+access\s+article[^.]*?(?:\.|$)',
    re.IGNORECASE,
)

# Para LDA: conserva letras, dígitos y guiones internos.
RE_NO_ALNUM   = re.compile(r'[^a-z0-9\-\s]')
RE_GUION_EXT  = re.compile(r'(^-+)|(-+$)')
RE_SOLO_DIGIT = re.compile(r'^[\d\-]+$')

# Siglas científicas de 2-3 letras que NO deben descartarse por longitud.
SIGLAS_VALIDAS = {
    'dna', 'rna', 'gdp', 'gis', 'iot', 'sst', 'mpa', 'eez', 'ghg', 'co2',
    'ndc', 'sdg', 'sme', 'gis', 'gps', 'led', 'uav', 'auv', 'rov', 'ai',
    'ml', 'gas', 'oil', 'air', 'ice', 'sea', 'bay', 'law', 'eco', 'usa',
    'eu', 'un', 'ph', 'uv', 'tv',
}

# =============================================================================
# 3. NORMALIZACIÓN DE CASING
# =============================================================================
def detectar_mayusculas(serie, umbral=0.80):
    """
    Fracción de documentos íntegramente en MAYÚSCULAS.
    Las exportaciones de Web of Science entregan TI y AB en mayúsculas;
    eso no es texto natural y perjudica a los modelos de embeddings
    'cased', cuyo tokenizador fragmenta las palabras en subpiezas que
    apenas aparecieron durante el preentrenamiento.
    """
    s = serie.astype(str)
    frac = float((s == s.str.upper()).mean())
    return frac, frac > umbral


# Scopus anexa el título en el idioma original entre corchetes:
#   "Coastal areas and the maritime economy; [Obalna podrucja i pomorsko...]"
# Es la vía por la que entran caracteres chinos y cirílicos al corpus.
RE_TITULO_TRADUCIDO = re.compile(r'\s*;\s*\[.*?\]\s*$', re.DOTALL)

def limpiar_titulo(t):
    """Elimina el título traducido anexado entre corchetes."""
    return RE_TITULO_TRADUCIDO.sub('', str(t)).strip()


_RE_INICIO_ORACION = re.compile(r'(^|(?<=[.!?])\s+)([a-z])')

def normalizar_casing(texto):
    """MAYÚSCULAS -> minúsculas con inicial de oración capitalizada."""
    t = str(texto).lower()
    return _RE_INICIO_ORACION.sub(lambda m: m.group(1) + m.group(2).upper(), t)

# =============================================================================
# 4. LEMATIZACIÓN
# =============================================================================
def _lematizar(word):
    """
    Lema nominal por defecto; si el nominal no cambia la palabra se prueba
    el verbal. En un corpus de abstracts predominan los sintagmas
    nominales, así que priorizar el sustantivo es preferible a la
    heurística de v2 (min por longitud), que producía 'meeting'->'meet'
    y 'building'->'build'.
    """
    n = _lemmatizer.lemmatize(word, pos='n')
    if n != word:
        return n
    return _lemmatizer.lemmatize(word, pos='v')

# =============================================================================
# 5. LIMPIEZA PARA LDA
# =============================================================================
def limpiar_para_lda(texto, stops, min_len=3):
    """
    Devuelve lista de tokens lematizados.
    ORDEN CORRECTO: normalizar -> tokenizar -> LEMATIZAR -> filtrar stopwords.
    """
    t = str(texto).lower()
    t = RE_URL.sub(' ', t)
    t = RE_EMAIL.sub(' ', t)
    t = RE_CITATION.sub(' ', t)
    t = RE_COPYRIGHT.sub(' ', t)
    t = RE_NO_ALNUM.sub(' ', t)

    out = []
    for w in t.split():
        w = RE_GUION_EXT.sub('', w)
        if not w or RE_SOLO_DIGIT.match(w):
            continue
        lema = _lematizar(w)                      # ← lematiza ANTES
        if lema in stops:                         # ← y filtra DESPUÉS
            continue
        if len(lema) < min_len and lema not in SIGLAS_VALIDAS:
            continue
        out.append(lema)
    return out

# =============================================================================
# 6. LIMPIEZA PARA BERTopic / embeddings
# =============================================================================
def limpiar_para_embeddings(texto, normalizar_mayus=False):
    """
    Limpieza MÍNIMA: sin stemming, sin lematización, sin stopwords.
    BERTopic usa embeddings semánticos; una limpieza agresiva destruiría
    la información que el modelo necesita (§4 del prompt de auditoría).
    """
    t = str(texto)
    t = RE_URL.sub(' ', t)
    t = RE_EMAIL.sub(' ', t)
    t = RE_CITATION.sub(' ', t)
    t = RE_COPYRIGHT.sub(' ', t)
    t = RE_SPACES.sub(' ', t).strip()
    if normalizar_mayus:
        t = normalizar_casing(t)
    return t

# =============================================================================
# 7. STOPWORDS EN 3 CAPAS
# =============================================================================
# CAPA 1 — funcionales. Solo NLTK inglés (179 términos).
# Se descarta ENGLISH_STOP_WORDS de sklearn (318): incluye 'system',
# 'computer', 'describe', 'find', 'detail', 'part', 'amount', 'move',
# 'call', 'name'... y sus propios autores desaconsejan usarla.
def capa1_funcionales():
    from nltk.corpus import stopwords
    try:
        return set(stopwords.words('english'))
    except Exception:
        return set()

# CAPA 2 — boilerplate editorial. Universal, no depende del dominio.
CAPA2_EDITORIAL = {
    'elsevier', 'springer', 'wiley', 'taylor', 'francis', 'informa',
    'oxford', 'cambridge', 'sage', 'emerald', 'frontiers', 'mdpi',
    'gmbh', 'llc', 'ltd', 'limited', 'press', 'publisher', 'publishing',
    'publication', 'copyright', 'reserved', 'licence', 'license',
    'licensee', 'behalf', 'author', 'authors', 'journal', 'article',
    'abstract', 'introduction', 'conclusion', 'discussion',
    'doi', 'issn', 'isbn', 'volume', 'issue',
}

# CAPA 3 — dominio. Se deriva EMPÍRICAMENTE en Script00 a partir de la
# frecuencia documental del corpus y se guarda en Excel para que sea
# auditable. Este set es solo la semilla manual (puede quedar vacío).
CAPA3_DOMINIO = set()

# Términos metodológicos genéricos. OJO: en v2 esta lista incluía
# 'model', 'control', 'structure', 'applied', 'science', 'physical',
# que pueden ser términos de dominio con valor discriminante. Se han
# retirado: si el corpus los usa como muletilla, la capa empírica los
# detectará por frecuencia documental.
CAPA2_METODOLOGICO = {
    'study', 'result', 'results', 'paper', 'research', 'analysis',
    'method', 'methods', 'approach', 'data', 'using', 'used', 'use',
    'show', 'shown', 'found', 'present', 'provide', 'based',
    'suggest', 'indicate', 'demonstrate', 'reveal', 'report',
    'describe', 'examine', 'investigate', 'evaluate', 'assess',
    'obtain', 'perform', 'conduct', 'propose', 'proposed',
    'however', 'therefore', 'thus', 'furthermore', 'moreover',
    'also', 'may', 'can', 'well', 'due', 'new', 'different',
    'fig', 'figure', 'table', 'section', 'et', 'al',
}


def construir_stopwords(dominio=None, incluir_metodologico=True,
                        df_stops_excel=None):
    """
    Ensambla las tres capas.

    dominio : set | None
        Capa 3, normalmente la que genera Script00 empíricamente.
    df_stops_excel : DataFrame | None
        StopWord.xlsx opcional, con una columna 'StopWord'.
    """
    stops = capa1_funcionales() | CAPA2_EDITORIAL
    if incluir_metodologico:
        stops |= CAPA2_METODOLOGICO
    if dominio:
        stops |= set(dominio)
    else:
        stops |= CAPA3_DOMINIO
    if df_stops_excel is not None:
        col = 'StopWord' if 'StopWord' in df_stops_excel.columns \
              else df_stops_excel.columns[0]
        stops |= set(df_stops_excel[col].astype(str)
                     .str.lower().str.strip().tolist())
    return {w for w in stops if w and w != 'nan'}


def derivar_stopwords_dominio(textos, umbral_df=0.60, min_len=4, top_n=80):
    """
    Capa 3 empírica: términos presentes en más del `umbral_df` de los
    documentos. Son los términos de la ecuación de búsqueda, que por
    construcción aparecen en casi todo el corpus y no discriminan.

    Devuelve lista de (termino, frecuencia_documental) para revisión
    MANUAL antes de aplicarse. No se aplica automáticamente.
    """
    from collections import Counter
    n = len(textos)
    df_count = Counter()
    for t in textos:
        toks = set(re.findall(r'[a-z]{%d,}' % min_len, str(t).lower()))
        df_count.update(toks)
    fuera = capa1_funcionales() | CAPA2_EDITORIAL | CAPA2_METODOLOGICO
    cand = [(w, c / n) for w, c in df_count.items()
            if c / n >= umbral_df and w not in fuera]
    cand.sort(key=lambda x: -x[1])
    return cand[:top_n]

# =============================================================================
# 8. PROCESAMIENTO PARALELO
# =============================================================================
def procesar_lda(docs_raw, stops, n_cores=None):
    """Tokens lematizados para LDA."""
    if n_cores is None:
        n_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"  Preprocesamiento LDA ({n_cores} núcleos)...")
    docs = Parallel(n_jobs=n_cores, backend='loky')(
        delayed(limpiar_para_lda)(d, stops) for d in docs_raw
    )
    vacios = sum(1 for d in docs if len(d) == 0)
    if vacios:
        print(f"  AVISO: {vacios} documentos quedaron vacíos tras limpieza LDA.")
    return docs


def procesar_embeddings(docs_raw, normalizar_mayus=False, n_cores=None):
    """Textos limpios (strings) para el modelo de embeddings."""
    if n_cores is None:
        n_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"  Preprocesamiento embeddings ({n_cores} núcleos)"
          f"{' + normalización de casing' if normalizar_mayus else ''}...")
    docs = Parallel(n_jobs=n_cores, backend='loky')(
        delayed(limpiar_para_embeddings)(d, normalizar_mayus) for d in docs_raw
    )
    vacios = sum(1 for d in docs if not d.strip())
    if vacios:
        print(f"  AVISO: {vacios} documentos vacíos tras limpieza.")
    return docs

# =============================================================================
# 9. PARÁMETROS COMO FUNCIÓN CONTINUA DEL TAMAÑO
# =============================================================================
def eps_grid_escalado(reducido, percentiles=(0, 5, 10), n_muestra=2000, seed=42):
    """
    Deriva valores de cluster_selection_epsilon a partir de la ESCALA REAL del
    espacio UMAP reducido, en vez de constantes absolutas.

    `cluster_selection_epsilon` es una distancia en las unidades de la
    proyección. Un mismo valor (0.05) puede ser enorme en un espacio y
    despreciable en otro, según n_components y el embedding de partida. En el
    corpus ECOT resultó despreciable: eps=0.10 no alteró ninguna partición.

    Se toma una muestra aleatoria de puntos, se calculan las distancias al
    vecino más cercano y se devuelven los percentiles pedidos. El percentil 0
    equivale a 0.0 (sin fusión), que es el valor por defecto del pipeline.

    Uso opcional, para un análisis de sensibilidad explícito:
        P['eps_grid'] = eps_grid_escalado(red)
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(reducido, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if len(X) > n_muestra:
        X = X[rng.choice(len(X), n_muestra, replace=False)]

    d, _ = NearestNeighbors(n_neighbors=2).fit(X).kneighbors(X)
    d1 = d[:, 1]                     # distancia al vecino más cercano

    valores = []
    for p in percentiles:
        v = 0.0 if p == 0 else float(np.percentile(d1, p))
        valores.append(round(v, 4))
    return sorted(set(valores))


def ms_para(mcs):
    """
    Valores de min_samples a explorar para un min_cluster_size dado.
    Se define a nivel de módulo (no como closure) para que el dict de
    parámetros siga siendo serializable con pickle.
    """
    # return sorted({1, max(1, mcs // 10), max(1, mcs // 4)})
    return sorted({
    1,
    max(1, mcs // 20),
    max(1, mcs // 10)
})


def etiqueta_escala(n_docs):
    """Solo descriptiva, para los informes. NO decide parámetros."""
    if n_docs < 5_000:
        return 'MICRO'
    if n_docs < 30_000:
        return 'SMALL'
    if n_docs <= 50_000:
        return 'MEDIUM'
    return 'LARGE'


def get_params(n_docs):
    """
    Parámetros derivados de forma continua de n_docs.

    min_cluster_size se expresa como PORCENTAJE del corpus, que es la
    forma correcta de razonarlo: 'un tópico debe agrupar al menos el 1%
    de los documentos' es un criterio defendible; 'min_cluster_size=50'
    no significa nada sin saber el tamaño del corpus.
    """
    n = int(n_docs)

    # ── HDBSCAN: rejilla como fracción del corpus ────────────────────────────
    # Rejilla REVISADA tras la evidencia de Etapa2_HDBSCAN (corpus ECOT, n≈21k).
    #
    # La rejilla anterior (0.02…0.10) resultó estar íntegramente en zona
    # degenerada: a partir del 1.5% del corpus HDBSCAN colapsa a 2-4 cúmulos
    # con dominancia ≥0.94, y las restricciones descartaban casi todo. Peor:
    # el 1% (mcs=212) está sobre una DISCONTINUIDAD — con min_samples=21 da
    # K=24, pero con min_samples=10 cae a K=3 y dominancia 0.95. Esa cercanía
    # al acantilado es lo que produce ARI≈0.24 y K=16±11 entre semillas.
    #
    # La única meseta con respuesta suave y monótona está en el 0.5%
    # (K = 49 → 46 → 42 al variar min_samples). Por eso la rejilla se recorta
    # por arriba y se extiende hacia abajo, que es la zona nunca explorada.
    fracs = (0.0025, 0.0035, 0.005, 0.0075, 0.01)
    mcs_ref = max(10, int(round(n * 0.005)))      # 0.5% = referencia (meseta)
    # mcs_ref SIEMPRE entra en la rejilla: en la versión anterior se usaba como
    # HDBSCAN de referencia en la Etapa 1 pero nunca se evaluaba formalmente en
    # la Etapa 2, y aun así podía acabar seleccionado.
    mcs_grid = sorted({max(10, int(round(n * f))) for f in fracs} | {mcs_ref})

    # min_samples relativo a mcs: ver ms_para() a nivel de módulo.
    # (En v2 bastaba con ms < mcs, lo que permitía mcs=300 con ms=1:
    #  cúmulos muy laxos.)

    # ── UMAP ─────────────────────────────────────────────────────────────────
    # n_neighbors: controla el equilibrio estructura local/global.
    nn_grid = sorted({10, 15, 30, min(50, max(15, int(math.sqrt(n) / 2)))})
    # n_components: más dimensiones para corpus grandes.
    if   n <  5_000: nc_grid = [5, 10]
    elif n < 30_000: nc_grid = [5, 10, 15]
    else:            nc_grid = [10, 15, 20]

    # min_dist SIEMPRE 0.0 para clustering. Valores mayores dispersan los
    # puntos dentro del cúmulo por diseño, lo que perjudica a HDBSCAN.
    # La proyección de VISUALIZACIÓN usa un UMAP aparte con min_dist>0.
    umap_grid = [dict(n_neighbors=nn, n_components=nc, min_dist=0.0)
                 for nn in nn_grid for nc in nc_grid]
    umap_ref  = dict(n_neighbors=15, n_components=min(10, nc_grid[-1]),
                     min_dist=0.0)

    # ── LDA ──────────────────────────────────────────────────────────────────
    k_max = int(min(50, max(25, round(math.sqrt(n) / 2))))
    params = dict(
        n_docs=n,
        escala=etiqueta_escala(n),

        # LDA
        k_min=4,
        k_max=40,
        k_step=1 if n < 30_000 else 2,
        passes_lda=30 if n < 30_000 else 20,
        passes_fijo=30 if n < 30_000 else 20,
        no_below=max(2, int(round(n * 0.002))),
        no_above=0.50,
        bigram_min_count=max(3, int(round(n * 0.003))),
        bigram_threshold=8,
        alpha_grid=['symmetric', 'asymmetric', 0.01, 0.1],
        eta_grid=['symmetric', 0.01, 0.05, 0.5],

        # Evaluación
        coherence_sample=None if n <= 10_000 else 10_000,

        # UMAP / HDBSCAN
        umap_grid=umap_grid,
        umap_ref=umap_ref,
        umap_viz=dict(n_neighbors=15, n_components=2, min_dist=0.10),
        mcs_grid=mcs_grid,
        mcs_ref=mcs_ref,
        hdbscan_ref=dict(min_cluster_size=mcs_ref,
                         min_samples=max(1, mcs_ref // 10),
                         cluster_selection_method='eom',
                         cluster_selection_epsilon=0.0,
                         metric='euclidean',
                         prediction_data=True),
        # cluster_selection_epsilon: se declara FIJO en 0.0.
        #
        # Motivo empírico: en Etapa2_HDBSCAN, eps ∈ {0.0, 0.05, 0.10} produjo
        # resultados IDÉNTICOS en 7 de las 8 combinaciones donde se barrió
        # (única excepción: mcs=106, ms=1, leaf → K 59 vs 57). El parámetro es
        # una distancia ABSOLUTA en el espacio UMAP reducido, y a esta escala
        # 0.10 queda muy por debajo de las distancias entre cúmulos, así que no
        # fusiona nada. Barrerlo triplicaba el costo de la Etapa 2 sin aportar
        # información: 38 filas para 22 combinaciones reales.
        #
        # Si se desea explorarlo, hay que escalarlo a los datos con
        # eps_grid_escalado(reducido) — nunca con constantes absolutas.
        eps_grid=[0.0],
        metodos_grid=['eom', 'leaf'],

        # Embeddings
        chunk_size=2_000 if n < 30_000 else 5_000,
        batch_size=16,
    )
    return params


if __name__ == "__main__":
    print("procesador_v3.py cargado correctamente.")
    print(f"  Núcleos disponibles: {multiprocessing.cpu_count()}")
    print("\n  Prueba de limpieza LDA (orden lematizar->filtrar):")
    stops = construir_stopwords()
    demo = "STUDIES ON COVID-19 AND CO2 SHOW THAT RESULTS ARE ANALYSED."
    print(f"    entrada : {demo}")
    print(f"    salida  : {limpiar_para_lda(demo, stops)}")
    print("\n  Perfiles por tamaño de corpus:")
    for n in [3_217, 8_000, 40_000, 80_000]:
        p = get_params(n)
        print(f"    {n:>7,} docs [{p['escala']:<6}] "
              f"K:{p['k_min']}-{p['k_max']} | mcs_ref={p['mcs_ref']} "
              f"| mcs_grid={p['mcs_grid']} | umap={len(p['umap_grid'])} combos")
