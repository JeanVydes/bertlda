# -*- coding: utf-8 -*-
"""
phase4_trends.py
================
Phase 4 - temporal trend analysis of topic prevalence.

For each topic, three complementary tests are run:
  1. OLS on annual prevalence (slope, R2, p)
  2. Mann-Kendall (non-parametric, tau, p)
  3. Beta regression on logit(p) (slope, p)

Classification (alpha=0.05, with Benjamini-Hochberg FDR correction):
  Increasing / Declining : OLS and MK significant, same sign
  Increasing* / Declining*: only one significant, same sign
  Stable                 : discordant signs or neither significant

Reads:  LDA result CSVs
Writes: outputs/results/trends/trend_analysis_<MODEL>.csv
        outputs/results/trends/prevalencia_anual_<MODEL>.csv
        outputs/visualizations/trend_grid_<MODEL>.png/.svg
"""
from __future__ import annotations

import os
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    _DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _DIR = os.getcwd()
sys.path.insert(0, _DIR)

from config import CFG, RUTAS
from pipeline_base import (
    asegurar_carpetas, _fmt, print_separador,
)


def _guard_statsmodels():
    try:
        import statsmodels  # noqa
        import scipy  # noqa
    except ImportError:
        raise ImportError(
            "Phase 4 requires statsmodels + scipy. Install: uv add statsmodels scipy"
        )


# ---------------- contrasts ----------------
def mann_kendall(x):
    from scipy.stats import norm
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4:
        return float("nan"), float("nan"), float("nan")
    S = 0
    for k in range(n - 1):
        S += int(np.sign(x[k + 1:] - x[k]).sum())
    _, cuentas = np.unique(x, return_counts=True)
    empates = cuentas[cuentas > 1]
    var = (n * (n - 1) * (2 * n + 5)
           - sum(t * (t - 1) * (2 * t + 5) for t in empates)) / 18.0
    if var <= 0:
        return 0.0, 1.0, float(S)
    z = (S - 1) / np.sqrt(var) if S > 0 else (
        (S + 1) / np.sqrt(var) if S < 0 else 0.0)
    p = 2 * (1 - norm.cdf(abs(z)))
    denom = n * (n - 1) / 2.0
    return float(S / denom), float(p), float(S)


def ols_tendencia(anios, prop):
    import statsmodels.api as sm
    X = sm.add_constant(np.asarray(anios, dtype=float))
    mod = sm.OLS(np.asarray(prop, dtype=float), X).fit()
    return float(mod.params[1]), float(mod.rsquared), float(mod.pvalues[1])


def smithson_verkuilen(y):
    y = np.asarray(y, dtype=float)
    n = len(y)
    return (y * (n - 1) + 0.5) / n


def beta_regresion(anios, prop):
    try:
        from statsmodels.othermod.betareg import BetaModel
        import statsmodels.api as sm
        y = smithson_verkuilen(np.asarray(prop, dtype=float))
        x = np.asarray(anios, dtype=float)
        X = sm.add_constant(x - x.mean())
        mod = BetaModel(y, X).fit(disp=0, maxiter=500)
        return float(mod.params[1]), float(mod.pvalues[1])
    except Exception as e:
        print(f"    WARNING: beta regression did not converge: {type(e).__name__}")
        return float("nan"), float("nan")


def benjamini_hochberg(pvals, alpha=0.05):
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
    if np.isnan(b_ols) or np.isnan(tau):
        return "Stable"
    s_ols, s_mk = np.sign(b_ols), np.sign(tau)
    if s_ols != s_mk or s_ols == 0:
        return "Stable"
    if sig_ols and sig_mk:
        return "Increasing" if s_ols > 0 else "Declining"
    if sig_ols or sig_mk:
        return "Increasing*" if s_ols > 0 else "Declining*"
    return "Stable"


# ---------------- series ----------------
def series_desde_asignacion(df, col_topic, col_anio, etiquetas=None):
    d = df[[col_anio, col_topic]].dropna()
    d = d[d[col_topic] != -1]
    d[col_anio] = d[col_anio].astype(int)
    total = d.groupby(col_anio).size().rename("n_docs")
    tabla = (d.groupby([col_anio, col_topic]).size()
             .unstack(fill_value=0).reindex(total.index, fill_value=0))
    prop = tabla.div(total, axis=0)
    nombres = {t: (etiquetas.get(t, f"Topic {t}") if etiquetas else f"Topic {t}")
               for t in prop.columns}
    return prop, total, nombres


def analizar(prop, total, nombres, modelo):
    ALPHA = 0.05
    MIN_DOCS = 20
    EXCLUIR_ULTIMO = True
    anios_validos = [a for a in total[total >= MIN_DOCS].index.tolist()]
    if EXCLUIR_ULTIMO and anios_validos:
        anios_validos = [a for a in anios_validos if a != max(anios_validos)]
    excluidos = sorted(set(total.index) - set(anios_validos))
    print(f"  Years analyzed: {min(anios_validos)}-{max(anios_validos)} "
          f"({len(anios_validos)} points)")
    if excluidos:
        print(f"  Excluded years (<{MIN_DOCS} docs): {', '.join(map(str, excluidos))}")
    if len(anios_validos) < 8:
        print(f"  WARNING: only {len(anios_validos)} time points - power is low.")

    P = prop.loc[anios_validos]
    x = np.array(anios_validos, dtype=float)
    rows = []
    for col in P.columns:
        y = P[col].values.astype(float)
        b, r2, p_ols = ols_tendencia(x, y)
        tau, p_mk, _ = mann_kendall(y)
        b_beta, p_beta = beta_regresion(x, y)
        rows.append({
            "topic_id": col,
            "topic": nombres.get(col, str(col)),
            "mean_prevalence_pct": round(float(y.mean() * 100), 2),
            "ols_beta_x1e3": round(b * 1000, 2),
            "ols_r2": round(r2, 3),
            "p_ols": p_ols,
            "mk_tau": (round(tau, 3) if not np.isnan(tau) else None),
            "p_mk": p_mk,
            "beta_slope_logit": (round(b_beta, 4) if not np.isnan(b_beta) else None),
            "p_beta": p_beta,
        })
    df = pd.DataFrame(rows)
    # FDR correction
    for test in ("ols", "mk", "beta"):
        col_p = "p_" + ("OLS" if test == "ols" else ("MK" if test == "mk" else "Beta"))
        rech, aj = benjamini_hochberg(df[col_p].values, 0.05)
        df["p_" + test.upper() + "_fdr"] = np.round(aj, 4)
        df["sig_" + test.upper()] = rech
    df["beta_agrees_ols"] = [
        "Yes" if (s and (bb is not None) and np.sign(bb) == np.sign(bo)) else "No"
        for s, bb, bo in zip(df["sig_BETA"], df["beta_slope_logit"], df["ols_beta_x1e3"])
    ]
    df["trend"] = [clasificar(bo, so, t, sm) for bo, so, t, sm
                   in zip(df["ols_beta_x1e3"], df["sig_OLS"], df["mk_tau"], df["sig_MK"])]
    df = df.sort_values("mean_prevalence_pct", ascending=False).reset_index(drop=True)
    return df, P, x, anios_validos, excluidos


def figura_grid(df, P, x, modelo, ruta_sin_ext):
    n = len(df)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    COLOR_UP, COLOR_DOWN, COLOR_FLAT = "#D62828", "#1F77B4", "#6C757D"
    for i, (_, r) in enumerate(df.iterrows()):
        ax = axes[i]
        y = P[r["topic_id"]].values * 100
        trend = r["trend"]
        color = (COLOR_UP if "Increasing" in trend else
                 COLOR_DOWN if "Declining" in trend else COLOR_FLAT)
        estilo = ("-" if "Increasing" in trend else "--" if "Declining" in trend else ":")
        ax.plot(x, y, color=color, lw=2.0)
        ax.fill_between(x, y, color=color, alpha=0.15)
        b, a = np.polyfit(x, y, 1)
        ax.plot(x, b * x + a, color="black", ls=estilo, lw=1.1)
        ax.set_title(str(r["topic"])[:46], fontsize=10, fontweight="bold")
        ax.set_xlabel("Year", fontsize=8)
        ax.set_ylabel("Annual topic prevalence (%)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.5)
        etq = {"Increasing": "↑ Increasing", "Increasing*": "↑ Increasing*",
               "Declining": "↓ Declining", "Declining*": "↓ Declining*",
               "Stable": "→ Stable"}[trend]
        ax.text(0.98, 0.96, etq, transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="none"))
        pb = r["p_beta"]
        txt_beta = ("Beta reg. p=nan (not available)" if (pb is None or (isinstance(pb, float) and np.isnan(pb)))
                    else f"Beta reg. p={pb:.3f} "
                    f"({'agrees' if r['beta_agrees_ols'] == 'Yes' else 'n.s./differs'})")
        ax.text(0.02, 0.03,
                f"OLS: β={r['ols_beta_x1e3']:.2f}×10⁻³  R²={r['ols_r2']:.3f}  p={r['p_ols']:.3f}\n"
                f"MK: τ={r['mk_tau']:.3f}  p={r['p_mk']:.3f}\n{txt_beta}",
                transform=ax.transAxes, fontsize=6.5, va="bottom")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Temporal trends in annual topic prevalence — {modelo} "
                 f"({int(x.min())}–{int(x.max())})", fontsize=15, fontweight="bold", y=0.999)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Patch(fc=COLOR_UP, label="Increasing (OLS + MK, p < 0.05)"),
        Patch(fc=COLOR_DOWN, label="Declining (OLS + MK, p < 0.05)"),
        Patch(fc=COLOR_FLAT, label="Stable (no convergent evidence)"),
    ], loc="lower center", ncol=3, fontsize=9, frameon=True)
    fig.tight_layout(rect=[0, 0.045, 1, 0.985])
    for fmt in CFG["viz"]["formatos"]:
        fig.savefig(f"{ruta_sin_ext}.{fmt}", dpi=CFG["viz"]["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure -> {ruta_sin_ext}.{{{','.join(CFG['viz']['formatos'])}}}")


def cargar_lda():
    theta_path = RUTAS["results"] / "lda" / "lda_theta.csv"
    summary_path = RUTAS["results"] / "lda" / "lda_topic_summary.csv"
    if not theta_path.exists():
        return None
    theta = pd.read_csv(theta_path)
    col_anio = "anio" if "anio" in theta.columns else "Year"
    if col_anio not in theta.columns:
        return None
    theta = theta.rename(columns={col_anio: "anio"})
    etiquetas = {}
    try:
        res = pd.read_csv(summary_path)
        for _, r in res.iterrows():
            kw = str(r.get("keywords_20", "")).split(",")[:3]
            etiquetas[f"Topic_{int(r['topic_id'])}"] = f"T{int(r['topic_id'])}: " + ", ".join(k.strip() for k in kw)
    except Exception:
        pass
    if "Dominant_LDA_Topic" in theta.columns:
        etq2 = {int(k.split("_")[1]): v for k, v in etiquetas.items()}
        prop, total, nombres = series_desde_asignacion(theta, "Dominant_LDA_Topic", "anio", etq2)
    else:
        cols = [c for c in theta.columns if c.startswith("topic_")]
        # fallback: dominant topic column named 'dominant_topic'
        if "dominant_topic" in theta.columns:
            etq2 = {int(k.split("_")[1]): v for k, v in etiquetas.items()}
            prop, total, nombres = series_desde_asignacion(theta, "dominant_topic", "anio", etq2)
        else:
            prop, total, nombres = None, None, None
    return prop, total, nombres


def main() -> dict:
    _guard_statsmodels()
    asegurar_carpetas()
    print_separador("PHASE 4 — TEMPORAL TREND ANALYSIS")
    t0 = perf_counter()
    for modelo, cargar in [("LDA", cargar_lda)]:
        print(f"\n-- {modelo} --")
        try:
            datos = cargar()
        except Exception as e:
            print(f"  Error loading: {e}")
            continue
        if datos is None:
            print(f"  Results for {modelo} not found; skipping.")
            continue
        prop, total, nombres = datos
        df, P, x, anios, excluidos = analizar(prop, total, nombres, modelo)
        trend_dir = RUTAS["results"] / "trends"
        trend_dir.mkdir(parents=True, exist_ok=True)
        xl = trend_dir / f"trend_analysis_{modelo}.csv"
        df.to_csv(xl, index=False, encoding="utf-8-sig")
        (P * 100).round(3).to_csv(trend_dir / f"prevalencia_anual_{modelo}.csv",
                                  encoding="utf-8-sig")
        total.reset_index().to_csv(trend_dir / f"docs_por_anio_{modelo}.csv",
                                  index=False, encoding="utf-8-sig")
        print(f"  CSV -> {xl}")
        figura_grid(df, P, x, modelo,
                    str(RUTAS["visualizations"] / f"trend_grid_{modelo}"))
        print(f"\n  {'Topic':<40} {'Mean%':>7} {'β×1e3':>9} {'τ':>7}  Trend")
        for _, r in df.iterrows():
            print(f"  {str(r['topic'])[:40]:<40} {r['mean_prevalence_pct']:>7.2f} "
                  f"{r['ols_beta_x1e3']:>9.2f} {r['mk_tau']:>7.3f}  {r['trend']}")
        print("\n  Summary: " + " | ".join(f"{k}={v}" for k, v in df["trend"].value_counts().items()))
        hecho = True
    if not hecho:
        print("\n  No results found for any model.")
        print("  Run phase1c and/or phase2c first.")
    print_separador("PHASE 4 COMPLETED")
    print(f"  Time: {_fmt(perf_counter() - t0)}")
    return {"ok": hecho}


if __name__ == "__main__":
    main()
