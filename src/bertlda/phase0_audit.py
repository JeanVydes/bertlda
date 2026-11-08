# -*- coding: utf-8 -*-
"""
phase0_audit.py
===============
Phase 0 - corpus audit and validation, with descriptive statistics and maps.

Everything produced here describes the RAW corpus (before any modeling):
    * distribution by year
    * country distribution (world choropleth + ranked bars + CSV)
    * journal / source ranking (CSV + bar chart)
    * country x year heatmap
    * document exclusions, stopword candidates

Inputs:  raw_corpus.csv (Scopus export)
Outputs: outputs/results/corpus/*.csv
         outputs/visualizations/*.png (and .svg)
         outputs/checkpoints/ck_corpus_base.pkl
"""
from __future__ import annotations

import os
import re
import sys
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    _DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _DIR = os.getcwd()
sys.path.insert(0, _DIR)

from config import (
    CFG, RUTAS, COL_TITLE, COL_ABSTRACT, COL_YEAR, COL_AUTHORS,
    COL_AFFILIATIONS, COL_DOI, COL_DOC_ID, COL_TEXT, COL_ANIO,
    COL_PAIS_ORIGINAL, COL_PAIS_NORMALIZADO, OUTPUT_DIR,
    COL_SOURCE_TITLE,
)
from pipeline_base import (
    asegurar_carpetas, leer_csv, guardar_csv, guardar_checkpoint,
    enrich_with_countries, _fmt, print_separador,
)
from text_processing import (
    clean_title_translated, normalizar_casing, detectar_mayusculas,
    derive_domain_stopwords, get_params,
)

MIN_PALABRAS = CFG["min_palabras_doc"]
UMBRAL_MAYUS = CFG["umbral_mayusculas"]
DPI = CFG["viz"]["dpi"]
FORMATOS = CFG["viz"]["formatos"]


def _norm_title(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _guardar_fig(fig, nombre):
    out = RUTAS["visualizations"]
    out.mkdir(parents=True, exist_ok=True)
    for fmt in FORMATOS:
        fig.savefig(out / f"{nombre}.{fmt}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {nombre}.{{{','.join(FORMATOS)}}}")


def _world_choropleth_country(df_country: pd.DataFrame):
    """Draw a choropleth of publication counts by country using Natural Earth.
    Falls back to a horizontal bar chart if geopandas is unavailable."""
    try:
        import geopandas as gpd
        import pycountry
    except Exception as e:
        print(f"  geopandas/pycountry unavailable ({e}); skipping choropleth.")
        return False

    try:
        world = gpd.read_file(
            "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
            "master/geojson/ne_110m_admin_0_countries.geojson")
    except Exception as e:
        print(f"  Could not load Natural Earth data: {e}; skipping choropleth.")
        return False

    # Map our canonical country names to ISO3
    def to_iso3(name):
        if not isinstance(name, str):
            return None
        name = name.strip()
        try:
            c = pycountry.countries.get(name=name)
            if c is not None:
                return c.alpha_3
        except Exception:
            pass
        # fuzzy search
        try:
            res = pycountry.countries.search_fuzzy(name)
            if res:
                return res[0].alpha_3
        except Exception:
            pass
        # manual aliases
        aliases = {"United States": "USA", "United Kingdom": "GBR",
                   "South Korea": "KOR", "Russia": "RUS", "Vietnam": "VNM",
                   "Turkey": "TUR", "Czech Republic": "CZE", "Czechia": "CZE",
                   "Iran": "IRN", "Taiwan": "TWN", "Hong Kong": "HKG",
                   "Macau": "MAC"}
        return aliases.get(name)

    m = df_country.copy()
    m["iso3"] = m["country"].map(to_iso3)
    m = m.dropna(subset=["iso3"])
    merged = world.merge(m, how="left", left_on="ISO_A3", right_on="iso3")
    merged["documents"] = merged["documents"].fillna(0)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_axis_off()
    merged.plot(column="documents", cmap="YlOrRd", linewidth=0.3,
                edgecolor="#666666", legend=True, ax=ax,
                legend_kwds={"label": "Number of publications", "shrink": 0.6},
                missing_kwds={"color": "#EFEFEF", "label": "No data"})
    ax.set_title("World map of publications by country", fontsize=13, pad=12)
    # top countries text box (legend)
    top_n = min(8, len(m))
    lines = ["Top countries:"]
    for _, row in m.sort_values("documents", ascending=False).head(top_n).iterrows():
        lines.append(f"{row['iso3']}  {int(row['documents'])}")
    txt = "\n".join(lines)
    ax.annotate(txt, xy=(0.02, 0.02), xycoords="axes fraction",
                fontsize=8, family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999", alpha=0.85),
                va="bottom", ha="left")
    fig.tight_layout()
    _guardar_fig(fig, "World_Map_Publications")
    return True


def main() -> dict:
    asegurar_carpetas()
    print_separador("PHASE 0 — CORPUS AUDIT AND VALIDATION")
    t0 = perf_counter()

    # 1. LOAD
    print("\n-- LOAD --")
    df_raw = leer_csv(CFG["input_file"])
    print(f"  {len(df_raw):,} rows x {df_raw.shape[1]} columns")

    # 2. VALIDATE
    print("\n-- VALIDATE COLUMNS --")
    required = [COL_TITLE, COL_ABSTRACT, COL_YEAR]
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    for c in required:
        print(f"  OK  {c:<16} nulls={df_raw[c].isna().sum():>5}")

    # 3. STABLE doc_id
    df = df_raw.reset_index(drop=True).copy()
    df.insert(0, COL_DOC_ID, np.arange(len(df), dtype=int))

    # 4. UPPERCASE DETECTION
    print("\n-- UPPERCASE DETECTION --")
    frac_ti, mayus_ti = detectar_mayusculas(df[COL_TITLE], UMBRAL_MAYUS)
    frac_ab, mayus_ab = detectar_mayusculas(df[COL_ABSTRACT], UMBRAL_MAYUS)
    print(f"  {COL_TITLE}: {frac_ti * 100:5.1f}% ALL CAPS")
    print(f"  {COL_ABSTRACT}: {frac_ab * 100:5.1f}% ALL CAPS")
    normalizar_mayus = bool(mayus_ti or mayus_ab)
    print("  -> normalization ON" if normalizar_mayus else "  -> natural casing, no normalization")

    # 5. DUPLICATES
    print("\n-- DUPLICATES --")
    dup_doi = 0
    if "DOI" in df.columns:
        doi = df["DOI"].dropna().astype(str).str.strip().str.lower()
        doi = doi[(doi != "") & (doi != "nan")]
        dup_doi = int(len(doi) - doi.nunique())
        print(f"  DOI present: {len(doi):,} | duplicates: {dup_doi}")
    else:
        print("  DOI column not found, skipping.")
    ti_norm = df[COL_TITLE].map(_norm_title)
    dup_ti = int(ti_norm.duplicated().sum())
    print(f"  Title duplicates: {dup_ti}")
    print("  Duplicates " + ("found (not removed)." if (dup_doi or dup_ti) else "none."))

    # 6. TEXT BUILD + EXCLUSIONS
    print("\n-- TEXT BUILD --")
    ti_src = df[COL_TITLE].fillna("").astype(str).map(clean_title_translated)
    ab_src = df[COL_ABSTRACT].fillna("").astype(str)
    n_trad = int(df[COL_TITLE].astype(str).str.contains(r";\s*\[.*\]\s*$", regex=True).sum())
    if n_trad:
        print(f"  Titles with appended translation in brackets: {n_trad} (removed)")
    df[COL_TEXT] = (ti_src.str.strip() + ". " + ab_src.str.strip()).str.strip()
    tiene_ti = ti_src.str.strip().str.len() > 0
    tiene_ab = ab_src.str.strip().str.len() > 0
    n_pal = df[COL_TEXT].str.split().str.len()
    suficiente = n_pal >= MIN_PALABRAS
    print(f"  With title: {int(tiene_ti.sum()):,} | with abstract: {int(tiene_ab.sum()):,} | both: {int((tiene_ti & tiene_ab).sum()):,}")
    print(f"  Without useful text: {int((~tiene_ti & ~tiene_ab).sum()):,} | <{MIN_PALABRAS} words: {int((~suficiente).sum()):,}")
    print(f"  Words: min={int(n_pal.min())} median={int(n_pal.median())} max={int(n_pal.max())}")
    n_largos = int((n_pal > 350).sum())
    print(f"  Docs >350 words (truncation risk at 512 tokens): {n_largos:,}")
    df_excl = df.loc[~suficiente, [COL_DOC_ID, COL_TITLE, COL_YEAR]].copy()
    df_excl["reason"] = f"< {MIN_PALABRAS} words"
    df_ok = df.loc[suficiente].copy().reset_index(drop=True)
    print(f"  FINAL DOCUMENTS: {len(df_ok):,} (excluded {len(df_excl):,})")

    # 7. YEARS
    print("\n-- YEARS --")
    py = pd.to_numeric(df_ok[COL_YEAR], errors="coerce")
    validos = py.between(1900, 2100)
    n_bad = int(py.isna().sum()) + int((~validos & py.notna()).sum())
    print(f"  Non-numeric / out of range: {n_bad}")
    print(f"  Range: {int(py.min())}–{int(py.max())}")
    df_ok[COL_ANIO] = py.where(validos)
    tabla_anios = (df_ok[COL_ANIO].dropna().astype(int).value_counts()
                   .sort_index().rename_axis("year").reset_index(name="documents"))
    tabla_anios["percentage"] = (tabla_anios["documents"] / tabla_anios["documents"].sum() * 100).round(2)
    ultimo = int(tabla_anios["year"].max())
    print(f"  LAST YEAR = {ultimo} with {int(tabla_anios.iloc[-1]['documents'])} docs (likely INCOMPLETE)")

    # 8. COUNTRIES (extracted from Affiliations)
    print("\n-- COUNTRIES (extracted from Affiliations) --")
    if "Affiliations" in df_ok.columns:
        df_ok = enrich_with_countries(df_ok)
        n_with = int(df_ok[COL_PAIS_NORMALIZADO].notna().sum())
        multi = int(df_ok["paises_todos"].astype(str).str.contains(";").sum())
        print(f"  Documents with primary country: {n_with:,}")
        print(f"  Documents with multiple countries: {multi:,}")
        tabla_paises = (df_ok[COL_PAIS_NORMALIZADO].dropna().value_counts()
                        .rename_axis("country").reset_index(name="documents"))
        tabla_paises["percentage"] = (tabla_paises["documents"] / tabla_paises["documents"].sum() * 100).round(2)
        print(f"  Distinct countries: {df_ok[COL_PAIS_NORMALIZADO].nunique()}")
        print(f"  Without country: {int(df_ok[COL_PAIS_NORMALIZADO].isna().sum())}")
        print("  Top 5: " + ", ".join(
            f"{r['country']}({r['documents']})" for _, r in tabla_paises.head(5).iterrows()))
    else:
        tabla_paises = pd.DataFrame()

    # 9. JOURNAL / SOURCE RANKING
    print("\n-- JOURNAL / SOURCE RANKING --")
    col_source = COL_SOURCE_TITLE if COL_SOURCE_TITLE in df_ok.columns else ("Source title" if "Source title" in df_ok.columns else None)
    if col_source:
        tabla_fuentes = (df_ok[col_source].dropna().value_counts()
                         .rename_axis("source").reset_index(name="documents"))
        tabla_fuentes["percentage"] = (tabla_fuentes["documents"] / tabla_fuentes["documents"].sum() * 100).round(2)
        print(f"  Distinct sources: {df_ok[col_source].nunique()}")
        print("  Top 5: " + ", ".join(
            f"{r['source']}({r['documents']})" for _, r in tabla_fuentes.head(5).iterrows()))
    else:
        tabla_fuentes = pd.DataFrame()

    # 10. STOPWORD CANDIDATES (empirical)
    print("\n-- DOMAIN STOPWORD CANDIDATES (empirical) --")
    cand = derive_domain_stopwords(df_ok[COL_TEXT].tolist(), threshold=0.50, top_n=60)
    tabla_stops = pd.DataFrame(cand, columns=["term", "document_frequency_pct"])

    # 11. EXPORT CSV
    print("\n-- EXPORT --")
    audit_dir = RUTAS["results"] / "corpus"
    audit_dir.mkdir(parents=True, exist_ok=True)
    RUTAS["visualizations"].mkdir(parents=True, exist_ok=True)
    resumen = pd.DataFrame([
        {"metric": "Total records", "value": len(df_raw)},
        {"metric": "With title", "value": int(tiene_ti.sum())},
        {"metric": "With abstract", "value": int(tiene_ab.sum())},
        {"metric": "With both", "value": int((tiene_ti & tiene_ab).sum())},
        {"metric": "Without useful text", "value": int((~tiene_ti & ~tiene_ab).sum())},
        {"metric": "DOI duplicates", "value": dup_doi},
        {"metric": "Title duplicates", "value": dup_ti},
        {"metric": "Excluded by length", "value": len(df_excl)},
        {"metric": "USABLE DOCUMENTS", "value": len(df_ok)},
        {"metric": "Year range", "value": f"{int(py.min())}-{int(py.max())}"},
        {"metric": "Countries (normalized)", "value": int(df_ok[COL_PAIS_NORMALIZADO].nunique()) if COL_PAIS_NORMALIZADO in df_ok.columns else 0},
        {"metric": "Corpus ALL CAPS", "value": f"{frac_ab * 100:.1f}%"},
        {"metric": "Uppercase normalization", "value": normalizar_mayus},
        {"metric": "Docs >350 words", "value": n_largos},
        {"metric": "Distinct sources", "value": int(df_ok[col_source].nunique()) if col_source else 0},
    ])
    resumen.to_csv(audit_dir / "corpus_quality.csv", index=False, encoding="utf-8-sig")
    tabla_anios.to_csv(audit_dir / "annual_distribution.csv", index=False, encoding="utf-8-sig")
    if len(tabla_paises):
        tabla_paises.to_csv(audit_dir / "country_distribution.csv", index=False, encoding="utf-8-sig")
    if len(tabla_fuentes):
        tabla_fuentes.to_csv(audit_dir / "source_ranking.csv", index=False, encoding="utf-8-sig")
    if len(df_excl):
        df_excl.to_csv(audit_dir / "excluded_documents.csv", index=False, encoding="utf-8-sig")
    tabla_stops.to_csv(audit_dir / "stopword_candidates.csv", index=False, encoding="utf-8-sig")
    print(f"  Outputs in: {audit_dir}")

    # 12. DESCRIPTIVE GRAPHS (before any modeling)
    print("\n-- DESCRIPTIVE GRAPHS (pre-modeling) --")
    # annual (bar)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(tabla_anios["year"].astype(str), tabla_anios["documents"], color="#2166AC")
    ax.set_xlabel("Year"); ax.set_ylabel("Documents")
    ax.set_title("Documents per year")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    _guardar_fig(fig, "annual_distribution")

    # annual trend line (with 3-year moving average)
    if len(tabla_anios) >= 3:
        yrs = tabla_anios["year"].astype(int).to_numpy()
        vals = tabla_anios["documents"].astype(float).to_numpy()
        smooth = pd.Series(vals).rolling(window=3, center=True, min_periods=1).mean().to_numpy()
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(yrs, vals, marker="o", lw=1.5, color="#2166AC", label="Documents")
        ax.plot(yrs, smooth, lw=2.0, color="#B2182B", label="3-yr moving avg")
        ax.fill_between(yrs, vals, color="#2166AC", alpha=0.15)
        ax.set_xlabel("Year"); ax.set_ylabel("Documents")
        ax.set_title("Publication trend by year")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        _guardar_fig(fig, "annual_trend")

    # countries top 20 bar
    if len(tabla_paises):
        top = tabla_paises.head(20)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(top["country"][::-1], top["documents"][::-1], color="#B2182B")
        ax.set_xlabel("Documents"); ax.set_title("Top 20 countries by documents")
        for i, v in enumerate(top["documents"][::-1]):
            ax.text(v, i, f"  {v}", va="center", fontsize=8)
        fig.tight_layout()
        _guardar_fig(fig, "country_distribution_top20")

    # world choropleth
    if len(tabla_paises):
        _world_choropleth_country(tabla_paises)

    # sources top 15 (truncate long labels)
    if len(tabla_fuentes):
        top = tabla_fuentes.head(15).copy()
        top["src_short"] = top["source"].astype(str).str.slice(0, 42)
        top.loc[top["source"].astype(str).str.len() > 42, "src_short"] += "..."
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.barh(top["src_short"][::-1], top["documents"][::-1], color="#1B7837", height=0.7)
        ax.set_xlabel("Documents"); ax.set_title("Top 15 publishing sources")
        ax.tick_params(axis="y", labelsize=8)
        for i, v in enumerate(top["documents"][::-1]):
            ax.text(v, i, f"  {int(v)}", va="center", fontsize=8)
        fig.tight_layout()
        _guardar_fig(fig, "source_ranking_top15")

    # country x year heatmap (top 12 countries)
    if COL_PAIS_NORMALIZADO in df_ok.columns and COL_ANIO in df_ok.columns:
        top_c = df_ok[COL_PAIS_NORMALIZADO].value_counts().head(12).index.tolist()
        dd = df_ok[df_ok[COL_PAIS_NORMALIZADO].isin(top_c)]
        dd = dd[[COL_ANIO, COL_PAIS_NORMALIZADO]].dropna()
        dd[COL_ANIO] = dd[COL_ANIO].astype(int)
        tabla = (dd.groupby([COL_PAIS_NORMALIZADO, COL_ANIO]).size()
                 .unstack(fill_value=0))
        # restrict to years with data
        tabla = tabla.loc[top_c]
        fig, ax = plt.subplots(figsize=(max(8, tabla.shape[1] * 0.45),
                                       max(4, tabla.shape[0] * 0.32)))
        im = ax.imshow(tabla.values, aspect="auto", cmap="Reds")
        ax.set_xticks(range(tabla.shape[1]))
        ax.set_xticklabels(tabla.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(tabla.shape[0]))
        ax.set_yticklabels(tabla.index, fontsize=8)
        ax.set_xlabel("Year"); ax.set_title("Country x Year (document counts)")
        cb = fig.colorbar(im, ax=ax, shrink=0.7)
        cb.set_label("Documents", fontsize=9)
        fig.tight_layout()
        _guardar_fig(fig, "country_year_heatmap")

    # 13. CHECKPOINT
    P = get_params(len(df_ok))
    cols_to_save = [COL_DOC_ID, COL_TEXT, COL_ANIO, COL_PAIS_ORIGINAL,
                    COL_PAIS_NORMALIZADO, COL_TITLE]
    cols_to_save += [c for c in [
        COL_AUTHORS, COL_AFFILIATIONS, "DOI", "Source title",
        "Publisher", "Document Type", "Open Access",
    ] if c in df_ok.columns]
    if "paises_todos" in df_ok.columns and "paises_todos" not in cols_to_save:
        cols_to_save.append("paises_todos")
    checkpoint = {
        "df": df_ok[cols_to_save].copy(),
        "normalizar_mayus": normalizar_mayus,
        "params": P,
        "n_docs": len(df_ok),
    }
    guardar_checkpoint(checkpoint, "ck_corpus_base")

    # 14. SUMMARY
    print_separador("PHASE 0 COMPLETED")
    print(f"  Usable documents : {len(df_ok):,} [{P['escala']}]")
    print(f"  Profile          : K {P['k_min']}-{P['k_max']}")
    print(f"  Time             : {_fmt(perf_counter() - t0)}")
    print("\n-- NEXT STEP --")
    print("  LDA phases: phase1a -> phase1b -> phase1c")
    print("=" * 66)
    return checkpoint


if __name__ == "__main__":
    main()
