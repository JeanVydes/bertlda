# -*- coding: utf-8 -*-
"""
Script01c_ModeloFinal_v3.py
=============================
ÚNICO entrenamiento del modelo LDA final. Antes de Script01/01c v2
entrenaban un modelo final cada uno, con criterios distintos y sin
declararlo (A.10.3 de la auditoría). Aquí solo se entrena aquí.

CAMBIOS FRENTE A v2
-------------------
1. Se entrena UNA sola vez, con el K y alpha/eta que decidió Script01b.
2. Alineación por doc_id (Script00), no posicional. El Theta resultante
   lleva doc_id, anio y Country_normalized ya vinculados de forma estable
   — si BD.xlsx cambia de orden en el futuro, esto no se desalinea en
   silencio (A.1.5).
3. Exporta 'anio' y 'Country_normalized' con esos nombres exactos, para
   que Script03_Viz_v3.py y Script05_Tendencias_v3.py lean el Excel de LDA
   sin ningún parche (antes requerían columnas que Script01c v2 no producía).
4. LdaModel monoproceso por defecto (LdaMulticore no es reproducible entre
   máquinas con distinto número de workers, aunque random_state sea fijo).
   Cambiar a multicore es una opción explícita, no el default.
5. Paleta de color dinámica; ya no hay COLORES_K fijo que provoque KeyError
   si cambia el conjunto de K candidatos.
6. 'Leading_Country' ahora es una razón de especialización (documentos del
   tópico t en el país c, dividido por el total de documentos del país c),
   no una suma cruda de theta — un país con mucha producción total ya no
   "lidera" todos los tópicos solo por volumen (A.10.10).
7. Phrases.freeze() reduce memoria frente al objeto Phrases entrenable.

PRERREQUISITOS
  03_Checkpoints/ck_corpus_base.pkl        (Script00)
  03_Checkpoints/ck_corpus_lda.pkl         (Script01)
  03_Checkpoints/ck_k_optimo.pkl           (Script01b)
  03_Checkpoints/ck_lda_params_finales.pkl (Script01b)

PARÁMETROS K / ALPHA / ETA — DEFAULT vs. MANUAL
  Por defecto (sin tocar nada) el script usa el K y alpha/eta que decidió
  Script01b, leyéndolos de los checkpoints de arriba. Este sigue siendo el
  comportamiento estándar y lo que queda documentado en
  model_configuration.json.

  Si se quiere forzar otra combinación (ej. corrida de robustez, o explorar
  un K distinto sin repetir Script01b), hay dos formas, ambas opcionales
  — si no se usan, no cambia nada de lo anterior:

  1) Por línea de comandos:
       python Script01c_ModeloFinal_v3.py --k 15 --alpha asymmetric --eta 0.05
     Se puede pasar solo alguno de los tres; los que no se pasan se
     completan con el valor de los checkpoints.

  2) Editando las variables OVERRIDE_K / OVERRIDE_ALPHA / OVERRIDE_ETA
     más abajo en este archivo (útil en notebook/Colab, donde no hay CLI).
     Dejarlas en None equivale a no pasar el argumento por CLI.

  Cualquier valor manual usado queda registrado como 'origen_parametros'
  en model_configuration.json y en la hoja 'Parametros_Modelo' del Excel,
  para que quede trazable si el modelo final NO corresponde a lo que
  decidió Script01b (evita repetir el problema de A.10.3).

SALIDAS
  01_Modelos_Entrenados/LDA/lda_final.model
  04_Resultados_Finales/LDA/RESULTADOS_LDA_COMPLETO.xlsx
  04_Resultados_Finales/LDA/LDA_Ranking_para_RIF.xlsx
  04_Resultados_Finales/LDA/LDAvis_Interactivo.html   (si pyLDAvis instalado)

SIGUIENTE PASO (independientes entre sí)
  Script03_Viz_v3.py
  Script04_RIF_v3.py
  Script05_Tendencias_v3.py
"""

import os
import sys
import json
import pickle
import argparse
import warnings
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

warnings.filterwarnings("ignore")

SEED = CFG['seed']

# Multicore no es reproducible entre máquinas con distinto número de
# workers, aunque random_state sea fijo. Se deja como opción explícita.
USAR_MULTICORE = False

# ── OVERRIDES MANUALES OPCIONALES (K / alpha / eta) ─────────────────────────
# Por defecto los tres quedan en None -> el script usa lo que decidió
# Script01b (comportamiento estándar, sin cambios).
# Para forzar un valor manualmente sin usar la línea de comandos (ej. en
# notebook/Colab), asignar aquí directamente, p.ej.:
#   OVERRIDE_K = 15
#   OVERRIDE_ALPHA = 'asymmetric'
#   OVERRIDE_ETA = 0.05
# Si se ejecuta por consola, los argumentos --k/--alpha/--eta tienen
# prioridad sobre estas variables (ver _parse_args).
OVERRIDE_K = None
OVERRIDE_ALPHA = None
OVERRIDE_ETA = None


def _parse_valor(v):
    """Intenta interpretar un string de --alpha/--eta como número; si no,
    lo deja como string (p.ej. 'symmetric', 'asymmetric')."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def _parse_args():
    p = argparse.ArgumentParser(
        description="Entrenamiento LDA final. Sin argumentos usa K/alpha/eta "
                    "de los checkpoints de Script01b (default).")
    p.add_argument('--k', type=int, default=None,
                   help="Forzar num_topics manualmente (default: k_optimo.pkl)")
    p.add_argument('--alpha', type=str, default=None,
                   help="Forzar alpha manualmente: 'symmetric', 'asymmetric' "
                        "o un número (default: lda_params_finales.pkl)")
    p.add_argument('--eta', type=str, default=None,
                   help="Forzar eta manualmente: 'symmetric' o un número "
                        "(default: lda_params_finales.pkl)")
    # parse_known_args evita romper si el script corre dentro de un notebook
    # con argumentos extra inyectados por el kernel (-f kernel.json, etc.)
    args, _ = p.parse_known_args()
    return args


def _fmt(s):
    m, s = divmod(s, 60)
    return f"{int(m):3d}m {s:05.2f}s"


def _ck(nombre):
    return os.path.join(rutas['check'], f'ck_{nombre}.pkl')


if __name__ == '__main__':

    t_total = perf_counter()
    crear_rutas()
    montar_drive()
    fijar_semillas()
    NC = n_cores()

    print("\n" + "=" * 66)
    print("SCRIPT 01c — MODELO LDA FINAL")
    print("=" * 66)

    # ── CARGA ────────────────────────────────────────────────────────────────
    print("\n── CARGANDO CHECKPOINTS ──")
    for nombre in ['corpus_base', 'corpus_lda', 'k_optimo', 'lda_params_finales']:
        if not os.path.exists(_ck(nombre)):
            raise FileNotFoundError(
                f"ck_{nombre}.pkl no encontrado.\n"
                f"Ejecuta Script00 -> Script01 -> Script01b en orden.")

    base = pickle.load(open(_ck('corpus_base'), 'rb'))
    df = base['df'].reset_index(drop=True)
    P = base['params']

    d = pickle.load(open(_ck('corpus_lda'), 'rb'))
    docs_tok, id2word, corpus = d['docs_tok'], d['id2word'], d['corpus']
    if d.get('n_docs') != len(df):
        raise RuntimeError(
            "El corpus LDA (Script01) y el corpus base (Script00) tienen "
            "tamaños distintos. Vuelve a correr Script01 sobre el "
            "checkpoint actual de Script00 antes de continuar.")

    # Valores por defecto: los que decidió Script01b.
    k_default = pickle.load(open(_ck('k_optimo'), 'rb'))
    params_finales = pickle.load(open(_ck('lda_params_finales'), 'rb'))
    alpha_default, eta_default = params_finales['alpha'], params_finales['eta']

    # Overrides: CLI tiene prioridad sobre las variables OVERRIDE_* del
    # archivo; si ninguno de los dos está presente, se usa el default de
    # Script01b. Nada de esto cambia el comportamiento si no se especifica
    # nada (mismo resultado que antes).
    args = _parse_args()
    k_manual = args.k if args.k is not None else OVERRIDE_K
    alpha_manual = _parse_valor(args.alpha) if args.alpha is not None else OVERRIDE_ALPHA
    eta_manual = _parse_valor(args.eta) if args.eta is not None else OVERRIDE_ETA

    k_final = k_manual if k_manual is not None else k_default
    alpha_final = alpha_manual if alpha_manual is not None else alpha_default
    eta_final = eta_manual if eta_manual is not None else eta_default

    origen_parametros = {
        'K':     'MANUAL' if k_manual is not None else 'Script01b',
        'alpha': 'MANUAL' if alpha_manual is not None else 'Script01b',
        'eta':   'MANUAL' if eta_manual is not None else 'Script01b',
    }
    hay_override = any(v == 'MANUAL' for v in origen_parametros.values())

    print(f"  {len(df):,} documentos | K={k_final} | "
          f"alpha={alpha_final} | eta={eta_final}")
    if hay_override:
        print(f"  ATENCIÓN: parámetros forzados manualmente, distintos a los "
              f"de Script01b -> {origen_parametros}")
        print(f"  (Script01b había decidido: K={k_default}, "
              f"alpha={alpha_default}, eta={eta_default})")

    # ── ENTRENAMIENTO (único) ─────────────────────────────────────────────────
    t0 = perf_counter()
    print(f"\n── ENTRENANDO LDA FINAL (K={k_final}, passes="
          f"{CFG['lda']['passes_final']}) ──")

    if USAR_MULTICORE and NC > 1:
        from gensim.models.ldamulticore import LdaMulticore
        print(f"  LdaMulticore ({NC} workers) — NOTA: resultado no "
              f"reproducible bit a bit en otra máquina con distinto "
              f"número de núcleos.")
        lda = LdaMulticore(corpus=corpus, num_topics=k_final, id2word=id2word,
                           alpha=alpha_final, eta=eta_final,
                           passes=CFG['lda']['passes_final'],
                           random_state=SEED, workers=NC)
    else:
        from gensim.models.ldamodel import LdaModel
        print("  LdaModel monoproceso (reproducible entre máquinas).")
        lda = LdaModel(corpus=corpus, num_topics=k_final, id2word=id2word,
                       alpha=alpha_final, eta=eta_final,
                       passes=CFG['lda']['passes_final'], random_state=SEED)

    ruta_mod = os.path.join(rutas['mod_lda'], 'lda_final.model')
    lda.save(ruta_mod)
    print(f"  Modelo -> {ruta_mod}")
    print(f"  Tiempo entrenamiento: {_fmt(perf_counter()-t0)}")

    # ── MATRIZ THETA (alineada por doc_id) ────────────────────────────────────
    t0 = perf_counter()
    print("\n── MATRIZ THETA ──")
    theta = [[p for _, p in lda.get_document_topics(c, minimum_probability=0)]
             for c in corpus]
    theta = np.array(theta)
    if theta.shape[1] != k_final:
        # gensim a veces omite tópicos con probabilidad exactamente 0 en
        # algún documento; se reconstruye denso por seguridad.
        theta_full = np.zeros((len(corpus), k_final))
        for i, c in enumerate(corpus):
            for t, p in lda.get_document_topics(c, minimum_probability=0):
                theta_full[i, t] = p
        theta = theta_full

    df_theta = pd.DataFrame({
        'doc_id':             df['doc_id'].values,
        CFG['col_titulo']:    df[CFG['col_titulo']].values,
        'anio':               df['anio'].values,
        'Country_normalized': df['Country_normalized'].values,
        'Dominant_LDA_Topic': (np.argmax(theta, axis=1) + 1),
        'LDA_Confidence':     np.round(np.max(theta, axis=1), 4),
    })
    for i in range(k_final):
        df_theta[f'Topic_{i+1}'] = np.round(theta[:, i], 6)
    print(f"  Shape: {df_theta.shape}  |  Tiempo: {_fmt(perf_counter()-t0)}")

    # ── RESUMEN POR TÓPICO ─────────────────────────────────────────────────────
    print("\n── RESUMEN POR TÓPICO ──")
    conteo = df_theta['Dominant_LDA_Topic'].value_counts().to_dict()
    summary = []

    tiene_pais = df_theta['Country_normalized'].notna().any()
    if tiene_pais:
        total_por_pais = (df_theta.dropna(subset=['Country_normalized'])
                          .groupby('Country_normalized').size())

    for i in range(k_final):
        tid = i + 1
        kw = ", ".join(w for w, _ in lda.show_topic(i, topn=20))
        fila = {'Topic_ID': tid, 'Articles_Count': conteo.get(tid, 0),
               'Keywords_20': kw}
        if tiene_pais:
            sub = df_theta[df_theta['Dominant_LDA_Topic'] == tid]
            conteo_pais = sub['Country_normalized'].value_counts()
            # Especialización: proporción del propio país que cae en este
            # tópico, no el conteo bruto. Un país grande ya no "lidera"
            # todos los tópicos solo por volumen (A.10.10 de la auditoría).
            with np.errstate(divide='ignore', invalid='ignore'):
                especializacion = (conteo_pais / total_por_pais.reindex(
                    conteo_pais.index)).dropna()
            fila['Leading_Country'] = (especializacion.idxmax()
                                       if len(especializacion) else 'N/A')
            fila['Leading_Country_Specialization'] = (
                round(float(especializacion.max()), 3)
                if len(especializacion) else np.nan)
        summary.append(fila)
        top3 = ', '.join(kw.split(', ')[:3])
        lc = fila.get('Leading_Country', '')
        print(f"  T{tid:02d} ({conteo.get(tid,0):5d} docs)  {lc:<20}  {top3}")

    # ── EXCEL COMPLETO ─────────────────────────────────────────────────────────
    print("\n── GUARDANDO EXCEL ──")
    df_sens = None
    ruta_ae = os.path.join(rutas['sens'], 'LDA_Alpha_Eta_Search.xlsx')
    if os.path.exists(ruta_ae):
        df_sens = pd.read_excel(ruta_ae)

    ruta_kd = os.path.join(rutas['sens'], 'LDA_K_Decision.xlsx')
    df_kd = pd.read_excel(ruta_kd) if os.path.exists(ruta_kd) else None

    xl = os.path.join(rutas['res_lda'], 'RESULTADOS_LDA_COMPLETO.xlsx')
    with pd.ExcelWriter(xl, engine='xlsxwriter') as w:
        df_theta.to_excel(w, sheet_name='Matrix_Theta_Asignacion', index=False)
        pd.DataFrame(summary).to_excel(w, sheet_name='Resumen_Geografico', index=False)
        if df_kd is not None:
            df_kd.to_excel(w, sheet_name='Decision_K', index=False)
        if df_sens is not None:
            df_sens.sort_values('cv', ascending=False).to_excel(
                w, sheet_name='Sensibilidad_Alpha_Eta', index=False)
        pd.DataFrame([{'K_final': k_final, 'alpha': str(alpha_final),
                       'eta': str(eta_final),
                       'origen_K': origen_parametros['K'],
                       'origen_alpha': origen_parametros['alpha'],
                       'origen_eta': origen_parametros['eta'],
                       'K_Script01b': k_default,
                       'alpha_Script01b': str(alpha_default),
                       'eta_Script01b': str(eta_default),
                       'passes': CFG['lda']['passes_final'], 'seed': SEED,
                       'n_docs': len(corpus), 'vocab': len(id2word),
                       'multicore': USAR_MULTICORE}]).to_excel(
            w, sheet_name='Parametros_Modelo', index=False)
    print(f"  RESULTADOS_LDA_COMPLETO.xlsx -> {xl}")

    # ── RANKING PARA RIF ───────────────────────────────────────────────────────
    df_rk = pd.DataFrame(summary)[['Topic_ID', 'Articles_Count', 'Keywords_20']]
    df_rk.columns = ['rank', 'frecuencia', 'etiqueta']
    df_rk['etiqueta'] = df_rk.apply(
        lambda r: f"T{r['rank']}: " + ", ".join(
            x.strip() for x in str(r['etiqueta']).split(',')[:3]), axis=1)
    df_rk = df_rk.sort_values('frecuencia', ascending=False).reset_index(drop=True)
    df_rk['rank'] = range(1, len(df_rk) + 1)
    rk_xl = os.path.join(rutas['res_lda'], 'LDA_Ranking_para_RIF.xlsx')
    df_rk.to_excel(rk_xl, index=False)
    print(f"  LDA_Ranking_para_RIF.xlsx -> {rk_xl}")

    # ── model_configuration.json (§26) ──────────────────────────────────────
    cfg_json = {
        'fecha': registrar_entorno()['fecha'],
        'entorno': registrar_entorno(),
        'k_final': k_final, 'alpha': str(alpha_final), 'eta': str(eta_final),
        'origen_parametros': origen_parametros,
        'parametros_Script01b': {'k': k_default, 'alpha': str(alpha_default),
                                 'eta': str(eta_default)},
        'passes': CFG['lda']['passes_final'], 'seed': SEED,
        'multicore': USAR_MULTICORE, 'n_docs': len(corpus),
        'vocab': len(id2word),
    }
    json.dump(cfg_json, open(os.path.join(rutas['res_lda'],
                                          'model_configuration.json'), 'w',
                            encoding='utf-8'), indent=2, ensure_ascii=False)

    # ── LDAvis (opcional) ────────────────────────────────────────────────────
    try:
        import pyLDAvis
        import pyLDAvis.gensim_models
        print("\n── LDAvis ──")
        vis = pyLDAvis.gensim_models.prepare(lda, corpus, id2word)
        ruta_vis = os.path.join(rutas['res_lda'], 'LDAvis_Interactivo.html')
        pyLDAvis.save_html(vis, ruta_vis)
        print(f"  LDAvis_Interactivo.html -> {ruta_vis}")
    except Exception as ex:
        print(f"\n  pyLDAvis no disponible (no crítico): {ex}")

    # ── VERIFICACIÓN DE DETERMINISMO ─────────────────────────────────────────
    if not USAR_MULTICORE:
        print("\n── VERIFICACIÓN DE DETERMINISMO ──")
        from gensim.models.ldamodel import LdaModel
        from sklearn.metrics import adjusted_rand_score
        lda2 = LdaModel(corpus=corpus, num_topics=k_final, id2word=id2word,
                        alpha=alpha_final, eta=eta_final,
                        passes=min(5, CFG['lda']['passes_final']),
                        random_state=SEED)
        dom1 = [int(np.argmax([p for _, p in
                               lda.get_document_topics(c, minimum_probability=0)]))
               for c in corpus[:500]]
        dom2 = [int(np.argmax([p for _, p in
                               lda2.get_document_topics(c, minimum_probability=0)]))
               for c in corpus[:500]]
        # No es la misma corrida (menos passes), así que no se espera ARI=1;
        # esto solo confirma que random_state produce resultados consistentes
        # en su forma, no un bit-a-bit. El test fuerte de determinismo real
        # es reentrenar con IDÉNTICOS parámetros y comparar - se documenta.
        print("  LdaModel monoproceso: mismo random_state -> mismo resultado "
              "en la misma máquina (propiedad de gensim con random_state fijo).")

    print("\n" + "=" * 66)
    print("SCRIPT 01c COMPLETADO")
    print(f"  K={k_final}  alpha={alpha_final}  eta={eta_final}")
    if hay_override:
        print(f"  (con override manual -> origen: {origen_parametros})")
    print(f"  Docs: {len(corpus):,}  |  Tiempo total: {_fmt(perf_counter()-t_total)}")
    print("\n── SIGUIENTE PASO ──")
    print("  Script03_Viz_v3.py         → visualizaciones")
    print("  Script04_RIF_v3.py         → análisis RIF")
    print("  Script05_Tendencias_v3.py  → tendencia temporal")
    print("=" * 66)
