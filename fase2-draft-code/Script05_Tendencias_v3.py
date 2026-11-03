# -*- coding: utf-8 -*-
"""
Script05_Tendencias_v3.py
=========================
Análisis de tendencia temporal de la prevalencia de tópicos (§18).
Funciona con LDA, con BERTopic o con ambos.

MÉTODO
  Para cada tópico se construye la serie de prevalencia anual y se aplican
  tres contrastes complementarios:

  1. OLS sobre la proporción anual        -> pendiente beta, R2, p
  2. Mann-Kendall (no paramétrico)        -> tau, p
  3. Regresión BETA sobre logit(p)        -> pendiente en escala logit, p

  Por qué los tres. OLS asume errores normales y homocedásticos, supuesto
  que una proporción acotada en [0,1] incumple: la varianza es maxima cerca
  de 0.5 y minima en los extremos. Mann-Kendall no asume forma funcional y
  resiste valores atipicos, pero solo detecta monotonia. La regresion beta
  modela explicitamente una respuesta en (0,1) con heterocedasticidad, que
  es la distribucion correcta para una prevalencia. Usarlos juntos convierte
  la clasificacion en una afirmacion sobre CONVERGENCIA de evidencia y no
  sobre el resultado de un unico test.

  Proporciones en 0 o 1 se transforman con Smithson-Verkuilen (2006):
      y' = (y*(n-1) + 0.5) / n
  porque la distribucion beta no admite los extremos.

CLASIFICACION (alpha = 0.05)
  Increasing / Declining   : OLS y MK significativos y con el mismo signo
  Increasing* / Declining* : solo uno de los dos significativo, mismo signo
  Stable                   : signos discordantes o ninguno significativo

  La regresion beta se reporta como comprobacion de robustez.

PRERREQUISITOS (al menos uno)
  04_Resultados_Finales/BERTopic/RESULTADOS_BERTOPIC_COMPLETO.xlsx
  04_Resultados_Finales/LDA/RESULTADOS_LDA_COMPLETO.xlsx

SALIDAS
  04_Resultados_Finales/Tendencias/Trend_Analysis_<MODELO>.xlsx
  06_Visualizaciones/Trend_Grid_<MODELO>.(png|svg|pdf)
"""

import os
import sys
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

from config import CFG, rutas, crear_rutas, montar_drive

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURACIÓN LOCAL
# =============================================================================
# Años con muy pocos documentos producen proporciones de altísima varianza
# (con 3 documentos, un solo artículo mueve la prevalencia 33 puntos).
# Se excluyen del ajuste y se deja constancia en el Excel.
MIN_DOCS_POR_ANIO = 20

# El último año suele estar incompleto en la exportación (indexación parcial).
EXCLUIR_ULTIMO_ANIO = True

ALPHA = 0.05

# Corrección por comparaciones múltiples sobre los p-valores de cada test.
# Con ~20 tópicos y alpha=0.05 se esperan ~1 falso positivo por test.
APLICAR_FDR = True

# 'dominante' = proporción de documentos cuyo tópico dominante es k
# 'theta'     = media de theta por año (solo LDA; usa toda la distribución)
DEFINICION_PREVALENCIA = 'dominante'

COLOR_UP, COLOR_DOWN, COLOR_FLAT = '#D62828', '#1F77B4', '#6C757D'


# =============================================================================
# CONTRASTES
# =============================================================================
def mann_kendall(x):
    """
    Mann-Kendall con corrección por empates y aproximación normal.
    Implementado a mano para no depender de pymannkendall.
    Devuelve (tau, p_valor, S).
    """
    from scipy.stats import norm
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4:
        return np.nan, np.nan, np.nan

    S = 0
    for k in range(n - 1):
        S += np.sign(x[k + 1:] - x[k]).sum()

    _, cuentas = np.unique(x, return_counts=True)
    empates = cuentas[cuentas > 1]
    var = (n * (n - 1) * (2 * n + 5)
           - sum(t * (t - 1) * (2 * t + 5) for t in empates)) / 18.0
    if var <= 0:
        return 0.0, 1.0, S

    if S > 0:
        z = (S - 1) / np.sqrt(var)
    elif S < 0:
        z = (S + 1) / np.sqrt(var)
    else:
        z = 0.0

    p = 2 * (1 - norm.cdf(abs(z)))
    denom = n * (n - 1) / 2.0
    return float(S / denom), float(p), float(S)


def ols_tendencia(anios, prop):
    """OLS de la proporción contra el año. Devuelve (beta, R2, p)."""
    import statsmodels.api as sm
    X = sm.add_constant(np.asarray(anios, dtype=float))
    mod = sm.OLS(np.asarray(prop, dtype=float), X).fit()
    return float(mod.params[1]), float(mod.rsquared), float(mod.pvalues[1])


def smithson_verkuilen(y):
    """Comprime (0,1) para admitir 0 y 1 exactos en la regresión beta."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    return (y * (n - 1) + 0.5) / n


def beta_regresion(anios, prop):
    """
    Regresión beta con enlace logit. El año se centra en su media para que
    el intercepto sea interpretable y mejore la convergencia.
    Devuelve (pendiente_logit, p_valor).
    """
    from statsmodels.othermod.betareg import BetaModel
    import statsmodels.api as sm

    y = smithson_verkuilen(np.asarray(prop, dtype=float))
    x = np.asarray(anios, dtype=float)
    X = sm.add_constant(x - x.mean())
    try:
        # link_precision=None hacía que statsmodels intentara llamar None()
        # como función de enlace (TypeError) y el except silenciaba el fallo
        # para TODOS los tópicos -> columnas Beta_slope_logit/p_Beta vacías
        # en el Excel y "Beta reg. p=nan" en la figura. Se usa el enlace por
        # defecto (Log) para la precisión phi, que es el estándar en
        # regresión beta y converge sin problema.
        mod = BetaModel(y, X).fit(disp=0, maxiter=500)
        return float(mod.params[1]), float(mod.pvalues[1])
    except Exception as e:
        print(f"    AVISO: regresión beta no convergió para este tópico "
              f"({type(e).__name__}: {e})")
        return np.nan, np.nan


def benjamini_hochberg(pvals, alpha=0.05):
    """Devuelve (rechazado, p_ajustado) con control de FDR."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    ajust = np.full_like(p, np.nan)
    if ok.sum() == 0:
        return np.zeros_like(p, dtype=bool), ajust
    idx = np.where(ok)[0]
    orden = idx[np.argsort(p[idx])]
    m = len(orden)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        val = min(prev, p[orden[i]] * m / (i + 1))
        ajust[orden[i]] = val
        prev = val
    return (ajust <= alpha) & ok, ajust


def clasificar(b_ols, sig_ols, tau, sig_mk):
    """Clasificación por convergencia de evidencia."""
    if np.isnan(b_ols) or np.isnan(tau):
        return 'Stable'
    s_ols, s_mk = np.sign(b_ols), np.sign(tau)
    if s_ols != s_mk or s_ols == 0:
        return 'Stable'
    if sig_ols and sig_mk:
        return 'Increasing' if s_ols > 0 else 'Declining'
    if sig_ols or sig_mk:
        return 'Increasing*' if s_ols > 0 else 'Declining*'
    return 'Stable'


# =============================================================================
# SERIES DE PREVALENCIA
# =============================================================================
def series_desde_asignacion(df, col_topic, col_anio, etiquetas=None):
    """Prevalencia = proporción de documentos con tópico dominante k."""
    d = df[[col_anio, col_topic]].dropna()
    d = d[d[col_topic] != -1]
    d[col_anio] = d[col_anio].astype(int)

    total = d.groupby(col_anio).size().rename('n_docs')
    tabla = (d.groupby([col_anio, col_topic]).size()
             .unstack(fill_value=0).reindex(total.index, fill_value=0))
    prop = tabla.div(total, axis=0)
    nombres = {t: (etiquetas.get(t, f"Topic {t}") if etiquetas else f"Topic {t}")
               for t in prop.columns}
    return prop, total, nombres


def series_desde_theta(df, cols_topic, col_anio, etiquetas=None):
    """Prevalencia = media de theta por año (aprovecha toda la distribución)."""
    d = df[[col_anio] + list(cols_topic)].dropna(subset=[col_anio])
    d[col_anio] = d[col_anio].astype(int)
    total = d.groupby(col_anio).size().rename('n_docs')
    prop = d.groupby(col_anio)[list(cols_topic)].mean()
    prop = prop.div(prop.sum(axis=1), axis=0)     # renormaliza a 1 por año
    nombres = {c: (etiquetas.get(c, c) if etiquetas else c) for c in prop.columns}
    return prop, total, nombres


# =============================================================================
# ANÁLISIS
# =============================================================================
def analizar(prop, total, nombres, modelo):
    anios_validos = total[total >= MIN_DOCS_POR_ANIO].index.tolist()
    if EXCLUIR_ULTIMO_ANIO and anios_validos:
        ultimo = max(anios_validos)
        anios_validos = [a for a in anios_validos if a != ultimo]
        print(f"  Último año ({ultimo}) excluido por indexación parcial.")

    excluidos = sorted(set(total.index) - set(anios_validos))
    print(f"  Años analizados: {min(anios_validos)}–{max(anios_validos)} "
          f"({len(anios_validos)} puntos)")
    if excluidos:
        print(f"  Años excluidos ({MIN_DOCS_POR_ANIO} docs mínimo): "
              f"{', '.join(map(str, excluidos))}")
    if len(anios_validos) < 8:
        print(f"  AVISO: solo {len(anios_validos)} puntos temporales. "
              f"La potencia de Mann-Kendall y de la regresión beta es baja; "
              f"interpreta las clasificaciones con cautela.")

    P = prop.loc[anios_validos]
    x = np.array(anios_validos, dtype=float)

    filas = []
    for col in P.columns:
        y = P[col].values.astype(float)
        b, r2, p_ols = ols_tendencia(x, y)
        tau, p_mk, _ = mann_kendall(y)
        b_beta, p_beta = beta_regresion(x, y)
        filas.append({
            'Topic_ID': col,
            'Topic': nombres.get(col, str(col)),
            'Mean_prevalence_pct': round(float(y.mean() * 100), 2),
            'OLS_beta_x1e3': round(b * 1000, 2),
            'OLS_R2': round(r2, 3),
            'p_OLS': p_ols,
            'MK_tau': round(tau, 3) if not np.isnan(tau) else np.nan,
            'p_MK': p_mk,
            'Beta_slope_logit': round(b_beta, 4) if not np.isnan(b_beta) else np.nan,
            'p_Beta': p_beta,
        })

    df = pd.DataFrame(filas)

    # Significancia (con o sin control de FDR)
    if APLICAR_FDR:
        for test in ['OLS', 'MK', 'Beta']:
            rech, aj = benjamini_hochberg(df[f'p_{test}'].values, ALPHA)
            df[f'p_{test}_FDR'] = np.round(aj, 4)
            df[f'sig_{test}'] = rech
        print(f"  Corrección FDR (Benjamini-Hochberg) aplicada sobre "
              f"{len(df)} tópicos.")
    else:
        for test in ['OLS', 'MK', 'Beta']:
            df[f'sig_{test}'] = df[f'p_{test}'] < ALPHA

    df['Beta_agrees_OLS'] = [
        'Yes' if (s and not np.isnan(bb) and np.sign(bb) == np.sign(bo))
        else 'No'
        for s, bb, bo in zip(df['sig_Beta'], df['Beta_slope_logit'],
                             df['OLS_beta_x1e3'])]

    df['Trend'] = [clasificar(bo, so, t, sm) for bo, so, t, sm
                   in zip(df['OLS_beta_x1e3'], df['sig_OLS'],
                          df['MK_tau'], df['sig_MK'])]

    df = df.sort_values('Mean_prevalence_pct', ascending=False).reset_index(drop=True)
    for c in ['p_OLS', 'p_MK', 'p_Beta']:
        df[c] = df[c].astype(float).round(4)
    return df, P, x, anios_validos, excluidos


# =============================================================================
# FIGURA
# =============================================================================
def figura_grid(df, P, x, modelo, ruta_sin_ext):
    n = len(df)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for i, (_, r) in enumerate(df.iterrows()):
        ax = axes[i]
        y = P[r['Topic_ID']].values * 100
        trend = r['Trend']
        color = (COLOR_UP if 'Increasing' in trend else
                 COLOR_DOWN if 'Declining' in trend else COLOR_FLAT)
        estilo = ('-' if 'Increasing' in trend else
                  '--' if 'Declining' in trend else ':')

        ax.plot(x, y, color=color, lw=2.0)
        ax.fill_between(x, y, color=color, alpha=0.15)
        b, a = np.polyfit(x, y, 1)
        ax.plot(x, b * x + a, color='black', ls=estilo, lw=1.1)

        ax.set_title(str(r['Topic'])[:46], fontsize=10, fontweight='bold')
        ax.set_xlabel('Year', fontsize=8)
        ax.set_ylabel('Annual topic prevalence (%)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.5)

        etq = {'Increasing': '↑ Increasing', 'Increasing*': '↑ Increasing*',
               'Declining': '↓ Declining', 'Declining*': '↓ Declining*',
               'Stable': '→ Stable'}[trend]
        ax.text(0.98, 0.96, etq, transform=ax.transAxes, ha='right', va='top',
                fontsize=8, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc=color, ec='none'))

        pb = r['p_Beta']
        if np.isnan(pb):
            # La regresión beta no convergió para este tópico: no reportar
            # "n.s./differs", que se leería como un desacuerdo real con OLS.
            txt_beta = "Beta reg.: not available"
        else:
            txt_beta = (f"Beta reg. p={pb:.3f} "
                        f"({'agrees' if r['Beta_agrees_OLS']=='Yes' else 'n.s./differs'})")
        ax.text(0.02, 0.03,
                f"OLS: β={r['OLS_beta_x1e3']:.2f}×10⁻³  R²={r['OLS_R2']:.3f}  "
                f"p={r['p_OLS']:.3f}\n"
                f"MK: τ={r['MK_tau']:.3f}  p={r['p_MK']:.3f}\n"
                f"{txt_beta}",
                transform=ax.transAxes, fontsize=6.5, va='bottom')

    for j in range(n, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f'Temporal trends in annual topic prevalence — {modelo} '
                 f'({int(x.min())}–{int(x.max())})',
                 fontsize=15, fontweight='bold', y=0.999)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend(handles=[
        Patch(fc=COLOR_UP, label='Increasing (OLS + MK, p < 0.05)'),
        Patch(fc=COLOR_DOWN, label='Declining (OLS + MK, p < 0.05)'),
        Patch(fc=COLOR_FLAT, label='Stable (no convergent evidence)'),
        Line2D([], [], color='k', ls='-', label='OLS trend (solid = increasing)'),
        Line2D([], [], color='k', ls='--', label='OLS trend (dashed = declining)'),
        Line2D([], [], color='k', ls=':', label='OLS trend (dotted = stable)'),
    ], loc='lower center', ncol=3, fontsize=9, frameon=True)
    fig.tight_layout(rect=[0, 0.045, 1, 0.985])

    for fmt in CFG['viz']['formatos']:
        fig.savefig(f"{ruta_sin_ext}.{fmt}", dpi=CFG['viz']['dpi'],
                    bbox_inches='tight')
    plt.close(fig)
    print(f"  Figura -> {ruta_sin_ext}.{{{','.join(CFG['viz']['formatos'])}}}")


# =============================================================================
# CARGA DE MODELOS
# =============================================================================
def cargar_bertopic():
    ruta = os.path.join(rutas['res_bert'], 'RESULTADOS_BERTOPIC_COMPLETO.xlsx')
    if not os.path.exists(ruta):
        return None
    docs = pd.read_excel(ruta, sheet_name='Doc_Topic_Assignment')
    info = pd.read_excel(ruta, sheet_name='Topic_Info')
    col_t = 'Topic_reduced' if 'Topic_reduced' in docs.columns else 'Topic'
    etq = {}
    if 'Topic' in info.columns:
        cn = 'Name' if 'Name' in info.columns else info.columns[-1]
        etq = {int(t): str(n) for t, n in zip(info['Topic'], info[cn])}
    prop, total, nombres = series_desde_asignacion(docs, col_t, 'anio', etq)
    print(f"  BERTopic: asignación '{col_t}' | {prop.shape[1]} tópicos")
    return prop, total, nombres


def cargar_lda():
    ruta = os.path.join(rutas['res_lda'], 'RESULTADOS_LDA_COMPLETO.xlsx')
    if not os.path.exists(ruta):
        return None
    theta = pd.read_excel(ruta, sheet_name='Matrix_Theta_Asignacion')
    col_anio = 'anio' if 'anio' in theta.columns else CFG['col_anio']
    if col_anio not in theta.columns:
        print("  LDA: sin columna de año; se omite.")
        return None
    theta = theta.rename(columns={col_anio: 'anio'})

    etq = {}
    try:
        res = pd.read_excel(ruta, sheet_name='Resumen_Geografico')
        for _, r in res.iterrows():
            kw = str(r.get('Keywords_20', '')).split(',')[:3]
            etq[f"Topic_{int(r['Topic_ID'])}"] = \
                f"T{int(r['Topic_ID'])}: " + ", ".join(k.strip() for k in kw)
    except Exception:
        pass

    if DEFINICION_PREVALENCIA == 'theta':
        cols = [c for c in theta.columns if c.startswith('Topic_')]
        prop, total, nombres = series_desde_theta(theta, cols, 'anio', etq)
        print(f"  LDA: prevalencia = media de θ | {len(cols)} tópicos")
    else:
        etq2 = {int(k.split('_')[1]): v for k, v in etq.items()}
        prop, total, nombres = series_desde_asignacion(
            theta, 'Dominant_LDA_Topic', 'anio', etq2)
        print(f"  LDA: prevalencia = tópico dominante | {prop.shape[1]} tópicos")
    return prop, total, nombres


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':

    t0 = perf_counter()
    crear_rutas()
    montar_drive()
    ruta_out = os.path.join(rutas['base'], '04_Resultados_Finales', 'Tendencias')
    os.makedirs(ruta_out, exist_ok=True)

    print("\n" + "=" * 66)
    print("SCRIPT 05 — ANÁLISIS DE TENDENCIA TEMPORAL")
    print(f"  Mínimo de documentos por año : {MIN_DOCS_POR_ANIO}")
    print(f"  Excluir último año           : {EXCLUIR_ULTIMO_ANIO}")
    print(f"  Corrección FDR               : {APLICAR_FDR}")
    print("=" * 66)

    hecho = False
    for modelo, cargar in [('BERTopic', cargar_bertopic), ('LDA', cargar_lda)]:
        print(f"\n── {modelo} ──")
        try:
            datos = cargar()
        except Exception as e:
            print(f"  Error al cargar: {e}")
            continue
        if datos is None:
            print(f"  Resultados de {modelo} no encontrados; se omite.")
            continue

        prop, total, nombres = datos
        df, P, x, anios, excluidos = analizar(prop, total, nombres, modelo)

        xl = os.path.join(ruta_out, f'Trend_Analysis_{modelo}.xlsx')
        with pd.ExcelWriter(xl, engine='xlsxwriter') as w:
            df.to_excel(w, sheet_name='Trend_Analysis', index=False)
            (P * 100).round(3).to_excel(w, sheet_name='Prevalencia_Anual')
            total.reset_index().to_excel(w, sheet_name='Docs_por_Anio', index=False)
            pd.DataFrame([{
                'modelo': modelo,
                'definicion_prevalencia': DEFINICION_PREVALENCIA,
                'min_docs_por_anio': MIN_DOCS_POR_ANIO,
                'excluir_ultimo_anio': EXCLUIR_ULTIMO_ANIO,
                'anios_analizados': f"{int(min(anios))}-{int(max(anios))}",
                'n_puntos': len(anios),
                'anios_excluidos': ', '.join(map(str, excluidos)),
                'alpha': ALPHA, 'correccion_FDR': APLICAR_FDR,
            }]).to_excel(w, sheet_name='Configuracion', index=False)
        print(f"  Excel -> {xl}")

        figura_grid(df, P, x, modelo,
                    os.path.join(rutas['viz'], f'Trend_Grid_{modelo}'))

        print(f"\n  {'Tópico':<40} {'Media%':>7} {'β×10⁻³':>9} "
              f"{'τ':>7}  Tendencia")
        for _, r in df.iterrows():
            print(f"  {str(r['Topic'])[:40]:<40} "
                  f"{r['Mean_prevalence_pct']:>7.2f} "
                  f"{r['OLS_beta_x1e3']:>9.2f} "
                  f"{r['MK_tau']:>7.3f}  {r['Trend']}")
        print("\n  Resumen: " + " | ".join(
            f"{k}={v}" for k, v in df['Trend'].value_counts().items()))
        hecho = True

    if not hecho:
        print("\n  No se encontraron resultados de ningún modelo.")
        print("  Ejecuta Script02_Final_BERTopic_v3.py o Script01c_ModeloFinal_v3.py.")

    print(f"\n{'='*66}\nSCRIPT 05 COMPLETADO  ({perf_counter()-t0:.1f}s)\n{'='*66}")
