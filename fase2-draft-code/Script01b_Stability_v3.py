"""
Script01b_Stability_v3.py
==========================
Decide K con cuatro criterios y busca alpha/eta para el K elegido.

CAMBIOS FRENTE A v2
-------------------
1. CRITERIO DE PARSIMONIA CORREGIDO. v2 normalizaba la entropía por log(K)
   y llamaba a eso "parsimonia", pero la entropía normalizada tiende a
   crecer con K en corpus reales: el criterio premiaba K grandes, lo
   opuesto de lo que su nombre decía. Aquí la parsimonia penaliza
   directamente la distancia de K a K_min.
2. ESTABILIDAD REAL entre semillas (v2 solo la mencionaba en el nombre del
   script). Para los candidatos con mejor score se entrena con varias
   semillas, se alinean los tópicos entre corridas (asignación húngara
   sobre similitud coseno de las distribuciones palabra-tópico) y se mide
   ARI de la asignación documento->tópico dominante.
3. Paleta de color dinámica (v2 tenía COLORES_K = {15:..,19:..,23:..} y
   lanzaba KeyError si K_CANDIDATOS cambiaba). Se mantiene así: _paleta(n)
   genera la paleta según len(ks), sin diccionario fijo, por lo que elegir
   3, 4 o más candidatos (manual o automático) no requiere tocar nada aquí.
4. Alpha/eta se buscan DESPUÉS de fijar K (v2 los buscaba con el K del
   argmax de C_v de Script01, que ya no es la decisión final aquí).
5. Pesos declarados en config.py, no embebidos en el código.
6. SELECCIÓN DE CANDIDATOS A ESTABILIDAD, automática o manual
   (K_CANDIDATOS_MANUAL). Por defecto (None) se conserva el comportamiento
   automático: top N_TOP_ESTABILIDAD por score_parcial (plateau +
   parsimonia + interpretabilidad). Si se fija una lista, se usa esa lista
   tal cual y se ignora N_TOP_ESTABILIDAD; se valida contra la rejilla ya
   evaluada en Script01 para fallar temprano y con mensaje claro si se pide
   un K que nunca se corrió. El resto del pipeline (score_total sobre todos
   los K, gráficos, checkpoints) no necesita cambios: ya itera sobre lo que
   efectivamente terminó en `estab`, sea cual sea su tamaño.
7. DETECCIÓN FORMAL DEL PLATEAU (Kneedle, Satopaa et al. 2011) sobre la
   curva C_v vs K, como punto de referencia citable en el texto
   metodológico -- reemplaza la identificación visual/post-hoc. Requiere
   `pip install kneed`; si no está instalado, se omite sin detener el
   pipeline (solo se pierde el punto de referencia formal).
8. ANÁLISIS DE SENSIBILIDAD DE PESOS del score compuesto: recalcula el K
   ganador bajo 3 esquemas alternativos de pesos (iguales, estabilidad
   dominante, parsimonia dominante) además del declarado, y avisa si el
   K final depende de la ponderación elegida. Exporta
   LDA_Sensibilidad_Pesos.xlsx.
9. SEMILLAS DE ESTABILIDAD leídas de CFG['lda']['n_corridas_estabilidad']
   (antes: 5 semillas hardcodeadas en este script, desincronizadas del
   valor de config.py, que ya declaraba 10 sin que nada lo usara). El
   checkpoint ck_k_estabilidad.pkl ahora guarda cuántas semillas se
   usaron por K y se invalida/recalcula si el número pedido cambia -- de
   lo contrario, subir de 5 a 10 semillas habría reutilizado en silencio
   resultados calculados con las 5 anteriores. Los gráficos comparativos
   (boxplot, líneas por semilla) excluyen cualquier K cuyo checkpoint
   tenga un número de semillas distinto al actual.

SALIDAS
  05_Sensibilidad/LDA_K_Decision.xlsx
  05_Sensibilidad/LDA_Sensibilidad_Pesos.xlsx  (si no se usó OVERRIDE_K)
  05_Sensibilidad/LDA_Alpha_Eta_Search.xlsx
  06_Visualizaciones/LDA_K_vs_Criterios.(png|svg|pdf)
  06_Visualizaciones/LDA_Stability_BoxPlot.(png|svg|pdf)      [en inglés]
  06_Visualizaciones/LDA_Stability_CV_Seeds.(png|svg|pdf)     [en inglés]
  06_Visualizaciones/LDA_Stability_Radar.(png|svg|pdf)        [en inglés]
  03_Checkpoints/ck_k_optimo.pkl
  03_Checkpoints/ck_lda_params_finales.pkl
  03_Checkpoints/ck_k_estabilidad.pkl  (ahora incluye 'cv_por_semilla' por K)

SIGUIENTE PASO
  Script01c_ModeloFinal_v3.py
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import CFG, rutas, crear_rutas, montar_drive, fijar_semillas, n_cores
import Script01_LDA_v3 as S1

warnings.filterwarnings("ignore")

SEED = CFG['seed']

# ── Configuración local ───────────────────────────────────────────────────
# Deja OVERRIDE_K = <entero> para forzar un K manualmente (p.ej. tras leer
# el Excel y decidir con criterio experto) y saltar TODA la fase de
# decisión (plateau/estabilidad/parsimonia/interpretabilidad). No genera
# gráficos de estabilidad comparativos porque no corre esa fase.
OVERRIDE_K = None

# Candidatos que pasan a la fase de estabilidad (semillas + ARI + boxplot
# + C_v por semilla). Esto es distinto de OVERRIDE_K: aquí SÍ se corre la
# fase de estabilidad, pero solo sobre los K que se listen aquí.
#   None            -> automático: top N_TOP_ESTABILIDAD por score_parcial
#                       (plateau + parsimonia + interpretabilidad).
#   [12, 13, 16]    -> manual: se usan exactamente esos K, se ignora
#                       N_TOP_ESTABILIDAD. Deben existir en la rejilla ya
#                       evaluada por Script01 (ver K_min/K_max/K_step en
#                       config.py); si no, el script se detiene con un
#                       mensaje claro en vez de fallar más adelante con un
#                       error críptico.
K_CANDIDATOS_MANUAL =  [9, 10, 12, 13, 15, 17, 21]

PESOS_K = {'plateau': 0.30, 'stability': 0.30,
          'parsimony': 0.20, 'interpretability': 0.20}
N_TOP_ESTABILIDAD = 5        # candidatos que pasan a la etapa de estabilidad
                              # (solo aplica si K_CANDIDATOS_MANUAL es None)

# Semillas de la fase de estabilidad. Se leen de CFG['lda']['n_corridas_
# estabilidad'] (config.py) para que ambos archivos no se desincronicen
# -- antes config.py declaraba 10 corridas y este script usaba 5 valores
# hardcodeados sin que nada lo advirtiera. Semillas fijas y reproducibles
# (SEED, SEED+1, ..., SEED+n-1), no aleatorias.
N_SEMILLAS_ESTABILIDAD = CFG['lda']['n_corridas_estabilidad']
SEMILLAS_ESTABILIDAD = [SEED + i for i in range(N_SEMILLAS_ESTABILIDAD)]

PASSES_ESTABILIDAD = 10      # passes reducidos: solo para comparar corridas


def _fmt(s):
    m, s = divmod(s, 60)
    return f"{int(m):3d}m {s:05.2f}s"


def _ck(nombre):
    return os.path.join(rutas['check'], f'ck_{nombre}.pkl')


# =============================================================================
# CRITERIOS DE DECISIÓN DE K
# =============================================================================
def _minmax(s):
    r = s.max() - s.min()
    return (s - s.min()) / r if r > 0 else pd.Series(0.5, index=s.index)


def criterio_plateau(df_k):
    """
    Ganancia marginal de C_v al añadir un tópico más. Un K está en la meseta
    cuando la ganancia siguiente es pequeña: seguir subiendo K deja de
    aportar coherencia. Score alto = está en o después de la meseta.
    """
    cv = df_k.sort_values('K')['cv'].values
    ganancia = np.diff(cv, prepend=cv[0])
    # Ganancia relativa suavizada; K con ganancia marginal baja puntúa alto.
    score = 1 - _minmax(pd.Series(np.abs(ganancia), index=df_k.sort_values('K').index))
    return score.reindex(df_k.index)


def criterio_parsimonia(df_k, k_min):
    """
    Penaliza K grandes de forma DIRECTA (no vía entropía, que en v2
    premiaba lo contrario de lo que decía medir). Un modelo con menos
    tópicos es preferible en igualdad de condiciones (navaja de Occam).
    """
    dist = df_k['K'] - k_min
    return 1 - _minmax(dist)


def criterio_interpretabilidad(df_k):
    """
    Proxy de interpretabilidad: diversidad de vocabulario (TD) combinada
    con baja dominancia (ningún tópico absorbe el corpus) y NPMI, que
    correlaciona mejor que C_v con juicios humanos de calidad de tópico.
    """
    return (0.5 * _minmax(df_k['npmi'])
            + 0.3 * _minmax(df_k['diversity'])
            + 0.2 * (1 - _minmax(df_k['dominance'])))


def detectar_meseta_kneedle(df_k, ventanas=(1, 5, 7, 9)):
    """
    Localiza el punto de codo/rodilla de la curva C_v vs K con el algoritmo
    Kneedle (Satopaa et al., 2011). NO se reporta como un único punto
    "objetivo": sobre series de C_v reales (ruidosas, no puramente
    cóncavas-monótonas) el resultado de Kneedle es sensible al
    preprocesamiento. Se corre sobre la curva cruda (ventana=1) y sobre
    varias ventanas de suavizado (media móvil) y se reporta el CONJUNTO de
    resultados como chequeo de sensibilidad -- si convergen, es evidencia
    fuerte de un codo genuino; si no convergen (como puede ocurrir aquí),
    eso mismo es la conclusión honesta a reportar: la curva no tiene un
    único codo bien definido y la decisión de K debe apoyarse en el score
    multicriterio (plateau + estabilidad + parsimonia + interpretabilidad),
    no en un solo punto de Kneedle sin corroborar.

    Requiere `pip install kneed`. Si no está disponible, se omite sin
    detener el pipeline. Devuelve un dict {ventana: knee} (puede estar
    vacío o tener valores None si Kneedle no detecta codo en esa ventana).
    """
    try:
        from kneed import KneeLocator
    except ImportError:
        print("  AVISO: paquete 'kneed' no instalado (pip install kneed). "
              "Se omite la detección formal del plateau.")
        return {}

    d = df_k.sort_values('K')
    K = d['K'].values
    cv = d['cv'].values

    resultados = {}
    for w in ventanas:
        cv_w = (cv if w <= 1 else
               np.convolve(cv, np.ones(w) / w, mode='same'))
        try:
            kl = KneeLocator(K, cv_w, curve='concave',
                             direction='increasing', S=1.0)
            resultados[w] = int(kl.knee) if kl.knee is not None else None
        except Exception:
            resultados[w] = None

    validos = sorted({v for v in resultados.values() if v is not None})
    print(f"  Kneedle por ventana de suavizado: {resultados}")
    if len(validos) == 0:
        print("  AVISO: Kneedle no detectó codo en ninguna ventana.")
    elif len(set(validos)) == 1:
        print(f"  Kneedle CONVERGE en K={validos[0]} bajo todas las "
              f"ventanas probadas -- evidencia sólida de codo genuino.")
    else:
        print(f"  AVISO: Kneedle NO converge entre ventanas (rango "
              f"{min(validos)}-{max(validos)}). La curva de C_v no tiene "
              f"un único codo bien definido; repórtalo así en el texto y "
              f"apóyate en el score multicriterio, no en un punto único "
              f"de Kneedle, para la decisión final de K.")
    return resultados


# =============================================================================
# ESTABILIDAD ENTRE SEMILLAS
# =============================================================================
def _alinear_topicos(lda_ref, lda_otro, k):
    """Empareja tópicos de dos corridas por similitud coseno (húngaro)."""
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics.pairwise import cosine_similarity
    A = lda_ref.get_topics()
    B = lda_otro.get_topics()
    S = cosine_similarity(A, B)
    filas, cols = linear_sum_assignment(-S)
    return dict(zip(cols, filas))   # tópico_otro -> tópico_ref


def estabilidad_k(k, corpus, id2word, escala, n_cores_, semillas, passes,
                  docs_tok=None, dict_eval=None):
    """
    Entrena varias corridas con semillas distintas, alinea tópicos y mide
    ARI de la asignación documento->tópico dominante entre pares de
    corridas. ARI alto = la solución no depende de la semilla.

    Si se pasan docs_tok y dict_eval, también calcula C_v por cada corrida
    (una semilla = un modelo independiente), que es lo que alimenta los
    gráficos de estabilidad (boxplot y líneas por semilla).
    """
    from gensim.models.ldamodel import LdaModel
    from sklearn.metrics import adjusted_rand_score

    modelos, dominantes, cv_por_semilla = [], [], []
    for s in semillas:
        lda = LdaModel(corpus=corpus, num_topics=k, id2word=id2word,
                       alpha='symmetric', eta=None, passes=passes,
                       random_state=s)
        modelos.append(lda)
        dom = [int(np.argmax([p for _, p in
                              lda.get_document_topics(d, minimum_probability=0)]))
              for d in corpus]
        dominantes.append(dom)

        if docs_tok is not None and dict_eval is not None:
            m = S1.metricas_lda(lda, corpus, docs_tok, dict_eval, k)
            cv_por_semilla.append(m['cv'])

    ref = modelos[0]
    aris = []
    for i in range(1, len(modelos)):
        mapa = _alinear_topicos(ref, modelos[i], k)
        alineado = [mapa.get(t, t) for t in dominantes[i]]
        aris.append(adjusted_rand_score(dominantes[0], alineado))

    ari_medio = float(np.mean(aris)) if aris else np.nan
    ari_min = float(np.min(aris)) if aris else np.nan
    return ari_medio, ari_min, cv_por_semilla


# =============================================================================
# GRÁFICOS DE ESTABILIDAD (leyendas y etiquetas en inglés)
# =============================================================================
def _paleta(n):
    cmap = plt.get_cmap('tab10' if n <= 10 else 'tab20')
    return [cmap(i) for i in range(n)]


def graficar_boxplot_estabilidad(cv_por_k, ruta_viz, formatos, dpi):
    """
    Boxplot de C_v por K a través de las semillas de estabilidad.
    Equivalente en inglés al gráfico 'Stability_BoxPlot' original.
    """
    ks = sorted(cv_por_k.keys())
    if not ks:
        return
    datos = [cv_por_k[k] for k in ks]
    colores = _paleta(len(ks))

    fig, ax = plt.subplots(figsize=(9, 6))
    bp = ax.boxplot(datos, labels=[f'K={k}' for k in ks], patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], colores):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    medianas = [np.median(v) for v in datos]
    delta_plateau = round(max(medianas) - min(medianas), 4) if len(medianas) > 1 else 0.0
    ax.text(0.98, 0.03, f'Plateau range: Δ = {delta_plateau:.4f}',
           transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    ax.set_xlabel('Number of topics')
    ax.set_ylabel('Coherence CV')
    ax.set_title('Coherence stability across seeds by K')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    for fmt in formatos:
        fig.savefig(os.path.join(ruta_viz, f'LDA_Stability_BoxPlot.{fmt}'),
                    dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  LDA_Stability_BoxPlot.{{{','.join(formatos)}}}")


def graficar_lineas_cv_semillas(cv_por_k, semillas, ruta_viz, formatos, dpi):
    """
    Coherencia C_v por semilla para cada K, con banda sombreada mean±sigma.
    Equivalente en inglés a 'Stability_CV_Corridas'.
    """
    ks = sorted(cv_por_k.keys())
    if not ks:
        return
    colores = _paleta(len(ks))
    x = list(range(len(semillas)))

    fig, ax = plt.subplots(figsize=(10, 7))
    for k, color in zip(ks, colores):
        vals = cv_por_k[k]
        mu, sigma = float(np.mean(vals)), float(np.std(vals))
        ax.plot(x, vals, 'o-', color=color,
               label=f'K={k}  μ={mu:.4f}  σ={sigma:.4f}')
        ax.axhspan(mu - sigma, mu + sigma, color=color, alpha=0.12)

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in semillas])
    ax.set_xlabel('Seed')
    ax.set_ylabel('Coherence CV')
    ax.set_title('Coherence CV per seed, by K')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for fmt in formatos:
        fig.savefig(os.path.join(ruta_viz, f'LDA_Stability_CV_Seeds.{fmt}'),
                    dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  LDA_Stability_CV_Seeds.{{{','.join(formatos)}}}")


def graficar_radar_estabilidad(df_k, ks_estab, ruta_viz, formatos, dpi):
    """
    Radar de 3 ejes (Parsimony, Interpretability, Stability) para los K
    evaluados en la fase de estabilidad. Equivalente en inglés a
    'Stability_Radar'. Usa los scores ya calculados en df_k (score_parsimony,
    score_interpretability) y, para Stability, 1 menos la desviación estándar
    normalizada de C_v entre semillas (columna 'cv_std', si existe).
    """
    from math import pi

    sub = df_k[df_k['K'].isin(ks_estab)].copy()
    if sub.empty:
        return

    ejes = ['Stability\n(1-σ norm.)', 'Parsimony\n(norm.)', 'Interpretability\n(norm. NPMI)']
    n_ejes = len(ejes)
    angulos = [i / n_ejes * 2 * pi for i in range(n_ejes)]
    angulos += angulos[:1]

    if 'score_stability_cv' not in sub:
        sub['score_stability_cv'] = 0.5
    valores_k = {}
    for _, fila in sub.iterrows():
        vals = [fila['score_stability_cv'], fila['score_parsimony'],
               fila['score_interpretability']]
        valores_k[int(fila['K'])] = vals

    orden = sorted(valores_k, key=lambda k: -np.mean(valores_k[k]))
    colores = _paleta(len(orden))

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angulos[:-1])
    ax.set_xticklabels(ejes)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.00])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'])

    for k, color in zip(orden, colores):
        vals = valores_k[k]
        vals_cerrado = vals + vals[:1]
        score_medio = float(np.mean(vals))
        ax.plot(angulos, vals_cerrado, 'o-', color=color, linewidth=2,
               label=f'K={k}  ({score_medio:.3f})')
        ax.fill(angulos, vals_cerrado, color=color, alpha=0.15)

    ax.set_title('Stability, parsimony and interpretability by K', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    fig.tight_layout()
    for fmt in formatos:
        fig.savefig(os.path.join(ruta_viz, f'LDA_Stability_Radar.{fmt}'),
                    dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  LDA_Stability_Radar.{{{','.join(formatos)}}}")


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
    print("SCRIPT 01b — DECISIÓN DE K Y BÚSQUEDA DE ALPHA/ETA")
    print("=" * 66)

    # ── CARGA ────────────────────────────────────────────────────────────────
    ck_search = os.path.join(rutas['sens'], 'ck_k_search.pkl')
    if not os.path.exists(ck_search):
        raise FileNotFoundError(
            "ck_k_search.pkl no encontrado.\nEjecuta Script01_LDA_v3.py primero.")
    df_k = pd.DataFrame(pickle.load(open(ck_search, 'rb'))).sort_values('K').reset_index(drop=True)

    ck_lda = _ck('corpus_lda')
    d = pickle.load(open(ck_lda, 'rb'))
    docs_tok, id2word, corpus, dict_eval = (
        d['docs_tok'], d['id2word'], d['corpus'], d['dict_eval'])

    base = pickle.load(open(_ck('corpus_base'), 'rb'))
    P = base['params']
    print(f"  Candidatos evaluados: {df_k['K'].tolist()}")

    # estab se carga siempre (no solo bajo selección automática) para que
    # los gráficos de estabilidad puedan generarse aunque se use OVERRIDE_K,
    # siempre que exista un checkpoint previo de una corrida automática.
    ck_estab = os.path.join(rutas['sens'], 'ck_k_estabilidad.pkl')
    estab = pickle.load(open(ck_estab, 'rb')) if os.path.exists(ck_estab) else {}

    if OVERRIDE_K is not None:
        print(f"\n  OVERRIDE_K={OVERRIDE_K}: se omite la selección automática.")
        k_final = OVERRIDE_K
        df_k['score_total'] = np.nan
    else:
        # ── FASE 1: PLATEAU + PARSIMONIA + INTERPRETABILIDAD (todos los K) ──
        print("\n── CRITERIOS (todos los candidatos) ──")
        df_k['score_plateau'] = criterio_plateau(df_k)
        print("\n── CHEQUEO DE SENSIBILIDAD: KNEEDLE (informativo, no "
              "participa en el score) ──")
        kneedle_por_ventana = detectar_meseta_kneedle(df_k)
        df_k['score_parsimony'] = criterio_parsimonia(df_k, df_k['K'].min())
        df_k['score_interpretability'] = criterio_interpretabilidad(df_k)
        df_k['score_parcial'] = (
            PESOS_K['plateau'] * df_k['score_plateau']
            + PESOS_K['parsimony'] * df_k['score_parsimony']
            + PESOS_K['interpretability'] * df_k['score_interpretability'])

        # Candidatos que pasan a la fase de estabilidad: automático (top N
        # por score_parcial) o manual (K_CANDIDATOS_MANUAL). La paleta de
        # colores de los gráficos (_paleta(n)) ya es dinámica según
        # len(ks), así que elegir 3, 4 o más candidatos no rompe nada más
        # adelante en el script.
        if K_CANDIDATOS_MANUAL is not None:
            disponibles = set(df_k['K'])
            pedidos = set(K_CANDIDATOS_MANUAL)
            faltantes = pedidos - disponibles
            if faltantes:
                raise ValueError(
                    f"K_CANDIDATOS_MANUAL incluye K no evaluados en "
                    f"Script01: {sorted(faltantes)}. Rejilla disponible: "
                    f"{sorted(disponibles)}. Ajusta K_min/K_max/K_step en "
                    f"config.py y vuelve a correr Script01, o quita esos "
                    f"valores de K_CANDIDATOS_MANUAL.")
            top_estab = df_k[df_k['K'].isin(K_CANDIDATOS_MANUAL)]
            print(f"  Candidatos fijados manualmente "
                  f"({len(K_CANDIDATOS_MANUAL)}): {sorted(K_CANDIDATOS_MANUAL)}")
        else:
            top_estab = (df_k.sort_values('score_parcial', ascending=False)
                        .head(N_TOP_ESTABILIDAD))
            print(f"  Candidatos a estabilidad (top {N_TOP_ESTABILIDAD} por "
                  f"score parcial): {sorted(top_estab['K'].tolist())}")

        # ── FASE 2: ESTABILIDAD (solo los candidatos top) ────────────────────
        print(f"\n── ESTABILIDAD ({len(SEMILLAS_ESTABILIDAD)} semillas, "
              f"passes={PASSES_ESTABILIDAD}) ──")

        for k in sorted(top_estab['K'].tolist()):
            k = int(k)
            # Invalida el cache si el K ya fue evaluado pero con OTRO número
            # de semillas (p.ej. checkpoint de una corrida previa con 5 y
            # ahora se pide 10) -- si no, se reutilizaría en silencio un
            # resultado calculado con menos corridas de las que el resto
            # del script (y el texto metodológico) asume.
            entrada_previa = estab.get(k)
            if entrada_previa is not None:
                n_prev = len(entrada_previa.get('semillas', []))
                if n_prev == len(SEMILLAS_ESTABILIDAD):
                    continue
                print(f"  K={k}: checkpoint previo con {n_prev} semillas, "
                      f"se pidieron {len(SEMILLAS_ESTABILIDAD)} -> "
                      f"recalculando.")
            t_k = perf_counter()
            ari_medio, ari_min, cv_por_semilla = estabilidad_k(
                k, corpus, id2word, P['escala'], NC,
                SEMILLAS_ESTABILIDAD, PASSES_ESTABILIDAD,
                docs_tok=docs_tok, dict_eval=dict_eval)
            estab[k] = {'ARI_medio': ari_medio, 'ARI_min': ari_min,
                       'cv_por_semilla': cv_por_semilla,
                       'semillas': list(SEMILLAS_ESTABILIDAD),
                       'tiempo_s': round(perf_counter() - t_k, 1)}
            pickle.dump(estab, open(ck_estab, 'wb'))
            print(f"  K={k:>3} -> ARI medio={ari_medio:.4f} min={ari_min:.4f} "
                  f"({_fmt(estab[k]['tiempo_s'])})")

        df_k['ARI_medio'] = df_k['K'].map(lambda k: estab.get(int(k), {}).get('ARI_medio', np.nan))
        df_k['score_stability'] = df_k['ARI_medio'].fillna(0)
        # Renormaliza solo entre los evaluados en estabilidad para no
        # castigar a K fuera del top con un cero que no representa nada.
        evaluados = df_k['ARI_medio'].notna()
        if evaluados.sum() > 1:
            df_k.loc[evaluados, 'score_stability'] = _minmax(
                df_k.loc[evaluados, 'ARI_medio'])

        # Score de estabilidad basado en la dispersión de C_v entre semillas
        # (1 - sigma normalizado), usado exclusivamente en el radar. Menor
        # sigma = más estable frente a la semilla = score más alto.
        def _cv_std(k):
            vals = estab.get(int(k), {}).get('cv_por_semilla', [])
            return float(np.std(vals)) if len(vals) > 1 else np.nan
        df_k['cv_std'] = df_k['K'].map(_cv_std)
        df_k['score_stability_cv'] = np.nan
        con_std = df_k['cv_std'].notna()
        if con_std.sum() > 1:
            df_k.loc[con_std, 'score_stability_cv'] = 1 - _minmax(df_k.loc[con_std, 'cv_std'])
        elif con_std.sum() == 1:
            df_k.loc[con_std, 'score_stability_cv'] = 1.0

        df_k['score_total'] = np.where(
            evaluados,
            (PESOS_K['plateau'] * df_k['score_plateau']
             + PESOS_K['stability'] * df_k['score_stability']
             + PESOS_K['parsimony'] * df_k['score_parsimony']
             + PESOS_K['interpretability'] * df_k['score_interpretability']),
            np.nan)

        ganador = df_k.loc[df_k['score_total'].idxmax()]
        k_final = int(ganador['K'])
        print(f"\n  K ELEGIDO: {k_final}  (score_total={ganador['score_total']:.4f})")
        print(f"    Cv={ganador['cv']:.4f}  NPMI={ganador['npmi']:+.4f}  "
              f"ARI_medio={ganador['ARI_medio']:.4f}")
        print(f"    (Solo por C_v habría sido K="
              f"{int(df_k.loc[df_k['cv'].idxmax(),'K'])} — la diferencia es "
              f"la estabilidad y la parsimonia.)")

        # ── SENSIBILIDAD DE PESOS ────────────────────────────────────────────
        # Verifica si k_final depende de la ponderación elegida (0.30/0.30/
        # 0.20/0.20) o si se sostiene bajo esquemas alternativos. Solo tiene
        # sentido sobre los K que pasaron por estabilidad (evaluados) — los
        # demás quedan en NaN igual que en score_total.
        print("\n── SENSIBILIDAD DE PESOS ──")
        esquemas = {
            'declarado':  {'plateau': 0.30, 'stability': 0.30,
                          'parsimony': 0.20, 'interpretability': 0.20},
            'iguales':    {'plateau': 0.25, 'stability': 0.25,
                          'parsimony': 0.25, 'interpretability': 0.25},
            'estab_dom':  {'plateau': 0.20, 'stability': 0.40,
                          'parsimony': 0.20, 'interpretability': 0.20},
            'parsim_dom': {'plateau': 0.20, 'stability': 0.20,
                          'parsimony': 0.40, 'interpretability': 0.20},
        }
        filas_sens = []
        for nombre, w in esquemas.items():
            score_w = np.where(
                evaluados,
                (w['plateau'] * df_k['score_plateau']
                 + w['stability'] * df_k['score_stability']
                 + w['parsimony'] * df_k['score_parsimony']
                 + w['interpretability'] * df_k['score_interpretability']),
                np.nan)
            k_bajo_esquema = int(df_k.loc[np.nanargmax(score_w), 'K'])
            filas_sens.append({'esquema': nombre, **w, 'K_ganador': k_bajo_esquema})
            print(f"  {nombre:<12} -> K={k_bajo_esquema}")

        df_sens_pesos = pd.DataFrame(filas_sens)
        ks_alternativos = set(df_sens_pesos['K_ganador']) - {k_final}
        if ks_alternativos:
            print(f"  AVISO: el K ganador cambia bajo otros esquemas de "
                  f"pesos: {sorted(ks_alternativos)}. Repórtalo como "
                  f"limitación o evalúa fijar el esquema más conservador.")
        else:
            print(f"  K={k_final} se sostiene bajo los {len(esquemas)} "
                  f"esquemas de pesos evaluados.")
        ruta_sens_pesos = os.path.join(rutas['sens'], 'LDA_Sensibilidad_Pesos.xlsx')
        with pd.ExcelWriter(ruta_sens_pesos, engine='xlsxwriter') as w_sens:
            df_sens_pesos.to_excel(w_sens, sheet_name='Sensibilidad_Pesos', index=False)
            pd.DataFrame(
                [{'ventana_suavizado': k, 'K_kneedle': v}
                 for k, v in kneedle_por_ventana.items()]
            ).to_excel(w_sens, sheet_name='Kneedle_Sensibilidad', index=False)
        print(f"  Excel -> {ruta_sens_pesos}")

    # ── GRÁFICO ──────────────────────────────────────────────────────────────
    print("\n── GRÁFICO K vs CRITERIOS ──")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap('viridis', len(df_k))    # paleta dinámica, sin dict fijo

    ax = axes[0]
    ax.plot(df_k['K'], df_k['cv'], 'o-', label='C_v', color='#2166AC')
    ax.plot(df_k['K'], df_k['npmi'], 's--', label='NPMI', color='#B2182B')
    if k_final in df_k['K'].values:
        ax.axvline(k_final, color='green', ls=':', lw=2,
                  label=f'K elegido = {k_final}')
    ax.set_xlabel('K'); ax.set_ylabel('Coherencia'); ax.legend(fontsize=8)
    ax.set_title('Coherencia vs K')
    ax.grid(alpha=0.3)

    ax = axes[1]
    if 'score_total' in df_k and df_k['score_total'].notna().any():
        ax.plot(df_k['K'], df_k['score_total'], 'D-', color='#1B7837')
        ax.axvline(k_final, color='green', ls=':', lw=2)
    ax.set_xlabel('K'); ax.set_ylabel('Score total')
    ax.set_title('Score de decisión vs K')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for fmt in CFG['viz']['formatos']:
        fig.savefig(os.path.join(rutas['viz'], f'LDA_K_vs_Criterios.{fmt}'),
                    dpi=CFG['viz']['dpi'], bbox_inches='tight')
    plt.close(fig)
    print(f"  LDA_K_vs_Criterios.{{{','.join(CFG['viz']['formatos'])}}}")

    # ── GRÁFICOS DE ESTABILIDAD (BoxPlot, C_v por semilla, Radar) ───────────
    # El boxplot tolera cajas de distinto tamaño, pero graficar_lineas_cv_
    # semillas usa un único eje x = SEMILLAS_ESTABILIDAD para todas las
    # series: si el checkpoint tiene algún K "viejo" calculado con otro
    # número de semillas (p.ej. antes de subir de 5 a 10), se excluye de
    # este gráfico -- incluirlo produciría un ValueError por longitudes
    # distintas entre x y esa serie, o peor, un desalineamiento silencioso
    # semilla<->punto si las longitudes coincidieran por casualidad.
    cv_por_k_todos = {k: v.get('cv_por_semilla', []) for k, v in estab.items()
                      if v.get('cv_por_semilla')}
    n_actual = len(SEMILLAS_ESTABILIDAD)
    cv_por_k = {k: v for k, v in cv_por_k_todos.items() if len(v) == n_actual}
    descartados = sorted(set(cv_por_k_todos) - set(cv_por_k))
    if descartados:
        print(f"\n  AVISO: K={descartados} en el checkpoint de estabilidad "
              f"tienen un número de semillas distinto al actual "
              f"({n_actual}) -- se excluyen de los gráficos comparativos. "
              f"Vuelve a correr la fase de estabilidad para esos K si los "
              f"necesitas con {n_actual} semillas.")
    if cv_por_k:
        print("\n── STABILITY GRAPHS ──")
        graficar_boxplot_estabilidad(cv_por_k, rutas['viz'],
                                     CFG['viz']['formatos'], CFG['viz']['dpi'])
        graficar_lineas_cv_semillas(cv_por_k, SEMILLAS_ESTABILIDAD, rutas['viz'],
                                    CFG['viz']['formatos'], CFG['viz']['dpi'])
        graficar_radar_estabilidad(df_k, sorted(cv_por_k.keys()), rutas['viz'],
                                   CFG['viz']['formatos'], CFG['viz']['dpi'])
    else:
        print("\n  (Sin datos de C_v por semilla — no se generan los "
              "gráficos de estabilidad. Ejecuta la fase de estabilidad "
              "sin OVERRIDE_K al menos una vez.)")

    xl = os.path.join(rutas['sens'], 'LDA_K_Decision.xlsx')
    df_k.to_excel(xl, index=False)
    print(f"  Excel -> {xl}")

    # ── BÚSQUEDA DE ALPHA/ETA (con K ya fijo) ────────────────────────────────
    t0 = perf_counter()
    print(f"\n── BÚSQUEDA ALPHA/ETA (K={k_final}) ──")
    from gensim.models.ldamodel import LdaModel
    from gensim.models.ldamulticore import LdaMulticore
    Cls = LdaModel if (P['escala'] == 'MICRO' or NC == 1) else LdaMulticore
    ex = {} if Cls is LdaModel else {'workers': NC}

    ck_ae = pd.DataFrame()
    filas_ae = []
    for alpha in P['alpha_grid']:
        for eta in P['eta_grid']:
            try:
                lda = Cls(corpus=corpus, num_topics=k_final, id2word=id2word,
                         alpha=alpha, eta=eta, passes=P['passes_fijo'],
                         random_state=SEED, **ex)
                m = S1.metricas_lda(lda, corpus, docs_tok, dict_eval, k_final)
                m.update({'alpha': alpha, 'eta': eta})
                filas_ae.append(m)
                print(f"  alpha={str(alpha):<12} eta={str(eta):<10} "
                      f"-> Cv={m['cv']:.4f} NPMI={m['npmi']:+.4f} "
                      f"dom={m['dominance']:.3f}")
            except Exception as e:
                print(f"  alpha={alpha} eta={eta} ERROR: {e}")

    df_ae = pd.DataFrame(filas_ae)
    df_ae.to_excel(os.path.join(rutas['sens'], 'LDA_Alpha_Eta_Search.xlsx'),
                   index=False)
    mejor_ae = df_ae.loc[df_ae['cv'].idxmax()]
    alpha_final, eta_final = mejor_ae['alpha'], mejor_ae['eta']
    print(f"\n  ALPHA/ETA ELEGIDOS: alpha={alpha_final}  eta={eta_final}  "
          f"(Cv={mejor_ae['cv']:.4f})")
    print("  Criterio: coherencia sobre el K ya decidido — es un ajuste "
          "fino secundario a la elección de K, no otro criterio de K.")
    tiempos_ae = perf_counter() - t0

    # ── CHECKPOINTS PARA Script01c ────────────────────────────────────────────
    pickle.dump(k_final, open(_ck('k_optimo'), 'wb'))
    pickle.dump({'alpha': alpha_final, 'eta': eta_final, 'k': k_final},
               open(_ck('lda_params_finales'), 'wb'))
    print(f"\n  ck_k_optimo.pkl -> K={k_final}")
    print(f"  ck_lda_params_finales.pkl -> alpha={alpha_final}, eta={eta_final}")

    print("\n" + "=" * 66)
    print("SCRIPT 01b COMPLETADO")
    print(f"  K final: {k_final} | alpha={alpha_final} | eta={eta_final}")
    print(f"  Tiempo total: {_fmt(perf_counter()-t_total)}")
    print("\n── SIGUIENTE PASO ──")
    print("  Script01c_ModeloFinal_v3.py")
    print("=" * 66)
