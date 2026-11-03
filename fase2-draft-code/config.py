# -*- coding: utf-8 -*-
"""
config.py
=========
Configuración central del pipeline de topic modeling (LDA + BERTopic).
TODOS los scripts importan de aquí. No dupliques parámetros en otros archivos.

Uso:
    from config import CFG, rutas, crear_rutas
"""

import os
import sys
import multiprocessing

# =============================================================================
# 1. DETECCIÓN DE ENTORNO
# =============================================================================
def detectar_entorno():
    """Devuelve 'colab' | 'local'."""
    try:
        import google.colab  # type: ignore  # noqa
        return 'colab'
    except ImportError:
        return 'local'

ENTORNO = detectar_entorno()

# ★★★ EDITA ESTAS DOS RUTAS ★★★
RUTA_COLAB = '/content/drive/MyDrive/BLUEE'
RUTA_LOCAL = r'C:\Users\Javier_de_la_Hoz\Documents\Articulos\BLUE_ECONOMY_2026\blue_economy\VERSION_20_08_2026'

# Permite sobreescribir con variable de entorno: PIPELINE_BASE=/ruta
RUTA_BASE = os.environ.get(
    'PIPELINE_BASE',
    RUTA_COLAB if ENTORNO == 'colab' else RUTA_LOCAL
)

# =============================================================================
# 2. CONFIGURACIÓN PRINCIPAL
# =============================================================================
CFG = {

    # ── Reproducibilidad ─────────────────────────────────────────────────────
    # UNA sola semilla para todo el pipeline (antes: 123 en LDA, 42 en BERTopic)
    "seed": 42,

    # ── Archivos de entrada ──────────────────────────────────────────────────
    "archivo_bd": "BD.xlsx",
    "archivo_sw": "StopWord.xlsx",      # opcional; "" para omitir

    # ── Columnas ─────────────────────────────────────────────────────────────
    # Obligatorias: el pipeline falla con mensaje claro si falta alguna.
    "col_titulo":   "TI",
    "col_abstract": "AB",
    "col_anio":     "PY",
    "col_pais":     "Country",

    # Opcionales: se usan si existen, nunca son requisito.
    #   SO = revista | LA = idioma original | DI = DOI | TC = citas
    "cols_opcionales": ["SO", "LA", "DI", "TC"],

    # ── Corpus ───────────────────────────────────────────────────────────────
    # Si más del X% de los documentos está íntegramente en MAYÚSCULAS
    # (artefacto de exportación WoS/Scopus), se normaliza el casing antes
    # de generar embeddings. Ver docstring de procesador_v3.normalizar_casing.
    "umbral_deteccion_mayusculas": 0.80,

    # Enriquecimiento OPCIONAL: recupera el casing original desde TI_raw/AB_raw
    # SOLO si esas columnas existen Y se verifica AB_raw.upper() == AB.
    # False = comportamiento portable estándar (recomendado por defecto).
    "recuperar_casing_desde_raw": False,

    # Mínimo de palabras (TI+AB) para que un documento entre al modelo.
    "min_palabras_doc": 20,

    # ── Cómputo ──────────────────────────────────────────────────────────────
    "compute": {
        # GPU solo para embeddings. UMAP/HDBSCAN SIEMPRE en CPU:
        # a esta escala la GPU no acelera y cuML no es determinista.
        "embeddings_device": "auto",     # 'auto' | 'cuda' | 'cpu'
        "umap_backend":      "cpu",      # 'cpu' | 'cuml'  (cuml NO publicable)
        "hdbscan_backend":   "cpu",      # 'cpu' | 'cuml'
        "deterministic":     True,       # fp32 + algoritmos deterministas
        "n_cores":           None,       # None = automático
        "batch_size":        16,
    },

    # ── Embeddings ───────────────────────────────────────────────────────────
    # Modelo en uso para el pipeline principal.
    "embedding_model": "sentence-transformers/allenai-specter",

    # Candidatos para Script02b (comparación Modelo A/B/C, §31).
    "embedding_candidatos": [
        ("SciBERT (actual)",  "allenai/scibert_scivocab_uncased"),
        ("all-mpnet-base-v2", "sentence-transformers/all-mpnet-base-v2"),
        ("SPECTER",           "sentence-transformers/allenai-specter"),
        ("BGE-base-en-v1.5",  "BAAI/bge-base-en-v1.5"),
    ],
    "normalizar_embeddings": True,
    "max_seq_length": 512,

    # ── Vectorizador / c-TF-IDF ──────────────────────────────────────────────
    # MISMA configuración en búsqueda y modelo final (bug corregido de v2).
    "vectorizer": {
        "ngram_range": (1, 2),
        "min_df_frac": 0.0015,      # proporcional al corpus; mínimo 3
        "min_df_min":  3,
        "max_features": 50_000,
    },
    "ctfidf_reduce_frequent_words": True,

    # ── Restricciones duras de selección de modelo ───────────────────────────
    # Nivel 1: descartan configuraciones. NO puntúan.
    # Declaradas a priori para poder justificarlas en el artículo.
    "restricciones": {
        "max_outlier_rate":   0.35,
        "min_n_topics":       5,
        "max_dominance":      0.40,
    },

    # ── Pesos del score (Nivel 2, solo sobre las que sobreviven) ─────────────
    # Sin bonus por número de tópicos (contradecía §11).
    "pesos_score": {
        "cv":           0.25,   # coherencia C_v
        "npmi":         0.20,   # coherencia NPMI
        "diversity":    0.15,   # topic diversity
        "redundancy":   0.15,   # 1 - redundancia (penaliza tópicos gemelos)
        "separation":   0.15,   # silhouette en espacio UMAP
        "coverage":     0.10,   # 1 - outlier_rate
    },
    "umbral_redundancia": 0.70,   # similitud c-TF-IDF que define "casi idéntico"

    # ── Búsqueda en 3 etapas (§9) ────────────────────────────────────────────
    "busqueda": {
        # Máximo global. Script02 lo combina con el límite adaptativo de
        # get_params(n_docs): 3 UMAP si n<30k; 2 UMAP si n>=30k.
        "n_top_refinamiento": 3,

        # HDBSCAN coarse-to-fine:
        #   A) barrido estructural: mcs × min_samples, con eom y epsilon=0
        #   B) solo los mejores candidatos prueban método/epsilon
        # El número efectivo de candidatos finos depende de n_docs.
        "hdbscan_coarse_method": "eom",
        "hdbscan_coarse_epsilon": 0.0,

        "semillas_estabilidad": [42, 43, 44, 45, 46],
        "n_top_estabilidad":  3,    # configs evaluadas en etapa 3
    },

    # ── Selección MANUAL del modelo BERTopic final ───────────────────────────
    # Se llena DESPUÉS de ejecutar Script02_Grid_BERTopic_v3.py y revisar
    # 02_Resultados_Entrenamiento/Etapa2_HDBSCAN.xlsx.
    # El script final NO elige automáticamente el mayor score: valida que esta
    # combinación exista en la grilla y que haya pasado las restricciones.
    "seleccion_bertopic": {
        "n_neighbors": 30,          # ej. 34
        "n_components": 10,         # ej. 5
        "min_cluster_size": 82,     # ej. 69
        "min_samples": 20,          # ej. 17
        "metodo": "eom",               # "eom" o "leaf"
        "epsilon": 0.00,              # ej. 0.0
    },

    # ── Outliers ─────────────────────────────────────────────────────────────
    # threshold > 0: NO forzar la reasignación del 100% (§8).
    # Se exportan SIEMPRE ambas asignaciones (original y reducida).
    "outliers": {
        "reasignar":  True,
        "estrategia": "c-tf-idf",
        "threshold":  0.10,
    },

    # ── LDA ──────────────────────────────────────────────────────────────────
    "lda": {
        "passes_final": 50,
        "k_candidatos": None,      # None = se decide en Script01b
        "n_corridas_estabilidad": 10,
    },

    # ── Visualización ────────────────────────────────────────────────────────
    "viz": {
        "dpi": 500,
        "formatos": ["png", "svg", "pdf"],   # §13/§14/§20
        "top_words": 20,
        "max_topics_heatmap": 25,
        "top_paises": 10,
    },
}

# =============================================================================
# 3. RUTAS DEL PROYECTO
# =============================================================================
rutas = {
    "base":       RUTA_BASE,
    "mod_lda":    os.path.join(RUTA_BASE, "01_Modelos_Entrenados", "LDA"),
    "mod_bert":   os.path.join(RUTA_BASE, "01_Modelos_Entrenados", "BERTopic"),
    "res_train":  os.path.join(RUTA_BASE, "02_Resultados_Entrenamiento"),
    "check":      os.path.join(RUTA_BASE, "03_Checkpoints"),
    "res_lda":    os.path.join(RUTA_BASE, "04_Resultados_Finales", "LDA"),
    "res_bert":   os.path.join(RUTA_BASE, "04_Resultados_Finales", "BERTopic"),
    "res_rif":    os.path.join(RUTA_BASE, "04_Resultados_Finales", "RIF"),
    "res_corpus": os.path.join(RUTA_BASE, "04_Resultados_Finales", "Corpus"),
    "sens":       os.path.join(RUTA_BASE, "05_Sensibilidad"),
    "check_grid": os.path.join(RUTA_BASE, "05_Sensibilidad", "Checkpoints"),
    "viz":        os.path.join(RUTA_BASE, "06_Visualizaciones"),
    "viz_bt":     os.path.join(RUTA_BASE, "06_Visualizaciones", "BERTopic_HTML"),
}


def crear_rutas():
    """Crea todas las carpetas del proyecto. Idempotente."""
    for k, v in rutas.items():
        if k != "base":
            os.makedirs(v, exist_ok=True)
    return rutas


def montar_drive():
    """Monta Google Drive si estamos en Colab. No-op en local."""
    if ENTORNO == 'colab':
        try:
            from google.colab import drive  # type: ignore
            drive.mount('/content/drive')
            print("  Drive montado.")
        except Exception as e:
            print(f"  Aviso: no se pudo montar Drive: {e}")


def n_cores():
    """Núcleos a usar. Colab tiene 2; deja uno libre en local."""
    if CFG["compute"]["n_cores"] is not None:
        return CFG["compute"]["n_cores"]
    if ENTORNO == 'colab':
        return 2
    return max(1, multiprocessing.cpu_count() - 1)


def fijar_semillas(seed=None):
    """Fija semillas globales (random, numpy, torch) y modo determinista."""
    seed = CFG["seed"] if seed is None else seed
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if CFG["compute"]["deterministic"]:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except ImportError:
        pass
    return seed


def guardar_modelo_bertopic(topic_model, nombre='bertopic_final',
                            modelo_emb=None, verbose=True):
    """
    Guarda un modelo BERTopic de forma segura y VERIFICADA.

    Por qué existe esta función
    ---------------------------
    El FUSE de `drive.mount` acepta escrituras de archivos grandes sin lanzar
    error, pero no llega a sincronizarlas: un `save()` directo sobre Drive con
    serialization='pickle' y save_embedding_model=True (~500 MB - 1 GB, porque
    incluye los pesos del SentenceTransformer) se pierde en silencio al cerrar
    la sesión de Colab. El script termina "bien" y la carpeta queda vacía.

    Estrategia
    ----------
    1. Serializa en 'safetensors' guardando SOLO EL NOMBRE del modelo de
       embeddings (no sus pesos): el artefacto baja a unos pocos MB. Es el
       formato portable recomendado por BERTopic y `BERTopic.load()` lo lee
       igual que un pickle. Si la versión instalada no lo soporta, cae a pickle.
    2. Escribe primero en el disco local de la VM (rápido y fiable).
    3. Copia a Drive y verifica: existencia, tamaño idéntico y carga real.
       Ante cualquier fallo lanza excepción en vez de fingir éxito.

    Devuelve la ruta final en Drive.
    """
    import shutil
    from bertopic import BERTopic

    def _peso(p):
        """Tamaño en MB de un archivo o de un directorio completo."""
        if os.path.isfile(p):
            return os.path.getsize(p) / 1e6
        return sum(os.path.getsize(os.path.join(r, f))
                   for r, _, fs in os.walk(p) for f in fs) / 1e6

    os.makedirs(rutas['mod_bert'], exist_ok=True)
    destino = os.path.join(rutas['mod_bert'], nombre)

    # En Colab: escritura intermedia en el disco de la VM. En local: directo.
    tmp_dir = '/content/_modelo_tmp' if ENTORNO == 'colab' else rutas['mod_bert']
    os.makedirs(tmp_dir, exist_ok=True)
    tmp = os.path.join(tmp_dir, nombre)
    if os.path.exists(tmp):
        shutil.rmtree(tmp) if os.path.isdir(tmp) else os.remove(tmp)

    emb = modelo_emb or CFG['embedding_model']
    serializacion = 'safetensors'
    try:
        topic_model.save(tmp, serialization='safetensors',
                         save_ctfidf=True, save_embedding_model=emb)
    except Exception as e:
        if verbose:
            print(f"  safetensors no disponible ({e}); se usa pickle.")
        # El intento de safetensors puede haber dejado 'tmp' creado como
        # carpeta (escribe varios archivos antes de fallar). pickle espera
        # escribir un único archivo en esa ruta, así que hay que limpiarla
        # de nuevo aquí o revienta con IsADirectoryError.
        if os.path.exists(tmp):
            shutil.rmtree(tmp) if os.path.isdir(tmp) else os.remove(tmp)
        serializacion = 'pickle'
        topic_model.save(tmp, serialization='pickle')

    if not os.path.exists(tmp):
        raise RuntimeError("save() no produjo ningún archivo.")
    mb = _peso(tmp)

    if os.path.abspath(tmp) != os.path.abspath(destino):
        if os.path.exists(destino):
            shutil.rmtree(destino) if os.path.isdir(destino) else os.remove(destino)
        (shutil.copytree if os.path.isdir(tmp) else shutil.copy2)(tmp, destino)

    if not os.path.exists(destino):
        raise RuntimeError(f"La copia no llegó a {destino}")
    mb_dest = _peso(destino)
    if abs(mb_dest - mb) > 0.01 * max(mb, 1):
        raise RuntimeError(
            f"Tamaños distintos (local {mb:.1f} MB vs destino {mb_dest:.1f} MB): "
            f"la sincronización falló.")

    BERTopic.load(destino)          # prueba de carga real desde el destino
    if verbose:
        print(f"  Modelo guardado y verificado [{serializacion}] "
              f"({mb_dest:.1f} MB) -> {destino}")
    return destino


def registrar_entorno():
    """Devuelve dict con versiones de librerías, para model_configuration.json."""
    import platform
    info = {
        "fecha": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "entorno": ENTORNO,
        "python": sys.version.split()[0],
        "plataforma": platform.platform(),
        "seed": CFG["seed"],
    }
    for lib in ["numpy", "pandas", "sklearn", "gensim", "bertopic",
                "umap", "hdbscan", "sentence_transformers", "torch"]:
        try:
            m = __import__(lib)
            info[lib] = getattr(m, "__version__", "desconocida")
        except Exception:
            info[lib] = "no instalada"
    try:
        import torch
        info["cuda_disponible"] = bool(torch.cuda.is_available())
        info["gpu"] = (torch.cuda.get_device_name(0)
                       if torch.cuda.is_available() else None)
    except Exception:
        info["cuda_disponible"] = False
        info["gpu"] = None
    return info


if __name__ == "__main__":
    print("=" * 62)
    print("config.py")
    print("=" * 62)
    print(f"  Entorno   : {ENTORNO}")
    print(f"  RUTA_BASE : {RUTA_BASE}")
    print(f"  Existe    : {os.path.exists(RUTA_BASE)}")
    print(f"  SEED      : {CFG['seed']}")
    print(f"  Núcleos   : {n_cores()}")
    print(f"  UMAP      : {CFG['compute']['umap_backend']} "
          f"| HDBSCAN: {CFG['compute']['hdbscan_backend']}")
    print("\n  Entorno registrado:")
    for k, v in registrar_entorno().items():
        print(f"    {k:24s} {v}")
