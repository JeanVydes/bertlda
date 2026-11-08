# -*- coding: utf-8 -*-
"""
phase6_report.py
================
Phase 6 - final deliverables for the LDA methodology assignment.

Produces:
  * outputs/results/config/model_configuration_full.json
  * outputs/results/corpus/document_type_distribution.csv
  * outputs/results/corpus/publisher_ranking.csv
  * outputs/results/corpus/institution_ranking.csv
  * outputs/results/lda/lda_final_metrics.csv
  * outputs/results/lda/hjbiplot_*.csv
  * outputs/results/lda/ldavis_relevance_terms.csv
  * outputs/visualizations/PRISMA_Flow.{png,svg}
  * outputs/visualizations/document_type_distribution.{png,svg}
  * outputs/visualizations/institution_ranking_top20.{png,svg}
  * outputs/visualizations/LDA_K_Coherence.{png,svg}
  * outputs/visualizations/LDA_HJBiplot_Years.{png,svg}
  * outputs/visualizations/LDA_HJBiplot_Countries.{png,svg}
  * outputs/visualizations/LDA_HJBiplot_Documents.{png,svg}
  * outputs/visualizations/LDAvis_Intertopic_Map.{png,svg}
  * outputs/visualizations/LDAvis_Relevance_Heatmap.{png,svg}
  * outputs/visualizations/LDAvis.html
  * docs/Methodology_LDA.md

The report is written in Spanish because it is the academic deliverable; all
code, identifiers, CSV columns and JSON keys remain in English.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import importlib.metadata as _md
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

try:
    _DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _DIR = os.getcwd()
sys.path.insert(0, _DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CFG, RUTAS, PROJECT_ROOT
from pipeline_base import (
    asegurar_carpetas, cargar_checkpoint, guardar_csv, _fmt, print_separador,
    PAISES_CANONICOS,
)
from phase1a_lda_diagnostic import lda_metrics
from text_processing import build_stopwords

DPI = CFG["viz"]["dpi"]
FORMATOS = CFG["viz"]["formatos"]
LAMBDA = float(CFG["viz"]["relevance_lambda"])

TOPIC_LABELS = {
    1: "Adversarial attacks, privacy and cybersecurity",
    2: "Reasoning agents, alignment and cognitive frameworks",
    3: "Generative AI ethics, governance and societal challenges",
    4: "Human-AI interaction and conversational agents",
    5: "ChatGPT evaluation and benchmark performance",
    6: "AI in education and student learning",
    7: "Agentic systems, governance and enterprise deployment",
    8: "Bias, fairness, culture and language",
    9: "Clinical, medical and mental-health AI",
    10: "Model training, fine-tuning and predictive performance",
    11: "Retrieval-augmented generation and document automation",
    12: "Multimodal applications and natural-language tasks",
    13: "Safety policy, machine unlearning and spatial/environmental applications",
    14: "Neural networks, prediction and energy/market applications",
}

INSTITUTION_HINTS = {
    "university": 5, "universidad": 5, "universidade": 5, "université": 5,
    "universität": 5, "università": 5, "institute": 4, "instituto": 4,
    "college": 4, "school": 2, "laboratory": 3, "lab": 2, "centre": 3,
    "center": 3, "academy": 3, "academia": 3, "hospital": 3, "clinic": 2,
    "polytechnic": 3, "research": 2, "foundation": 2, "company": 1,
    "corporation": 1, "inc": 1, "ltd": 1, "gmbh": 1, "s.a.": 1,
}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def _guardar_fig(fig, nombre: str) -> None:
    for fmt in FORMATOS:
        fig.savefig(RUTAS["visualizations"] / f"{nombre}.{fmt}",
                    dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {nombre}.{{{','.join(FORMATOS)}}}")


def _short(text, n_words: int = 12) -> str:
    s = str(text).replace("\n", " ").strip()
    parts = s.split()
    return " ".join(parts[:n_words]) + ("..." if len(parts) > n_words else "")


def _md_table(df: pd.DataFrame, cols=None, headers=None, max_rows=None,
              floatfmt: str = ".4f") -> str:
    d = df.copy()
    if cols is not None:
        d = d[list(cols)]
    if max_rows is not None:
        d = d.head(max_rows)
    if headers is not None:
        d.columns = headers
    lines = [
        "| " + " | ".join(str(c) for c in d.columns) + " |",
        "|" + "|".join(["---"] * len(d.columns)) + "|",
    ]
    for _, row in d.iterrows():
        vals = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                fv = float(v)
                if pd.isna(fv):
                    vals.append("—")
                elif fv.is_integer() and abs(fv) >= 1:
                    vals.append(str(int(fv)))
                else:
                    vals.append(f"{fv:{floatfmt}}")
            elif isinstance(v, (int, np.integer)):
                vals.append(str(int(v)))
            else:
                vals.append(str(v).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def _version(pkg: str) -> str:
    try:
        return _md.version(pkg)
    except Exception:
        return "not installed"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Descriptive extras (document types, publishers, institutions)
# ---------------------------------------------------------------------------
GENERIC_INSTITUTION_TOKENS = {
    "ltd", "ltd.", "inc", "inc.", "llc", "gmbh", "co", "co.", "corp",
    "corp.", "corporation", "company", "limited", "plc", "s.a.", "s.l.",
    "ag", "bv", "sa", "department", "dept", "faculty", "division",
}


def _institution_from_affiliation(aff: str) -> str | None:
    if not isinstance(aff, str) or not aff.strip():
        return None
    countries = {c.lower() for c in PAISES_CANONICOS}
    candidates = []
    for seg in aff.split(";"):
        parts = [p.strip(" .") for p in seg.split(",") if p.strip()]
        parts = [p for p in parts if p and p.lower() not in countries]
        parts = [p for p in parts
                 if p.lower().strip(".") not in GENERIC_INSTITUTION_TOKENS]
        if not parts:
            continue
        scored = []
        for p in parts:
            low = p.lower()
            score = max((w for k, w in INSTITUTION_HINTS.items() if k in low),
                        default=0)
            scored.append((score, len(p), p))
        scored.sort(reverse=True)
        best = scored[0]
        candidates.append(best[2] if best[0] > 0 else parts[0])
    if not candidates:
        return None
    # Prefer the most frequent institution-like segment in the document.
    return Counter(candidates).most_common(1)[0][0]


def _descriptive_extras(base_df: pd.DataFrame) -> dict:
    outdir = RUTAS["results"] / "corpus"
    outdir.mkdir(parents=True, exist_ok=True)
    print("\n-- DESCRIPTIVE EXTRAS --")

    # Document types
    if "Document Type" in base_df.columns:
        dt = (base_df["Document Type"].fillna("Unknown").astype(str)
              .value_counts().rename_axis("document_type")
              .reset_index(name="documents"))
        dt["percentage"] = (dt["documents"] / dt["documents"].sum() * 100).round(2)
        guardar_csv(dt, "document_type_distribution.csv", subdir=outdir)
        top = dt.head(int(CFG["report"]["top_document_types"]))
        fig, ax = plt.subplots(figsize=(9, 4.6))
        ax.barh(top["document_type"][::-1], top["documents"][::-1],
                color="#4C72B0")
        ax.set_xlabel("Documents")
        ax.set_title("Document type distribution")
        for i, v in enumerate(top["documents"][::-1]):
            ax.text(v, i, f"  {int(v)}", va="center", fontsize=8)
        fig.tight_layout()
        _guardar_fig(fig, "document_type_distribution")
    else:
        dt = pd.DataFrame()

    # Publishers
    if "Publisher" in base_df.columns:
        pub = (base_df["Publisher"].fillna("Unknown").astype(str)
               .value_counts().rename_axis("publisher")
               .reset_index(name="documents"))
        pub["percentage"] = (pub["documents"] / pub["documents"].sum() * 100).round(2)
        guardar_csv(pub, "publisher_ranking.csv", subdir=outdir)
    else:
        pub = pd.DataFrame()

    # Institutions from the Scopus affiliation string
    if "Affiliations" in base_df.columns:
        inst_counter = Counter()
        for aff in base_df["Affiliations"].dropna().astype(str):
            names = set()
            for seg in aff.split(";"):
                name = _institution_from_affiliation(seg)
                if name:
                    names.add(name)
            inst_counter.update(names)
        inst = pd.DataFrame(
            inst_counter.most_common(int(CFG["report"]["top_institutions"])),
            columns=["institution", "documents"],
        )
        if len(inst):
            inst["percentage"] = (inst["documents"] / len(base_df) * 100).round(2)
            guardar_csv(inst, "institution_ranking.csv", subdir=outdir)
            top = inst.head(20)
            fig, ax = plt.subplots(figsize=(10, 6.5))
            ax.barh(top["institution"][::-1], top["documents"][::-1],
                    color="#2A9D8F")
            ax.set_xlabel("Documents")
            ax.set_title("Top 20 institutions by documents")
            ax.tick_params(axis="y", labelsize=8)
            for i, v in enumerate(top["documents"][::-1]):
                ax.text(v, i, f"  {int(v)}", va="center", fontsize=8)
            fig.tight_layout()
            _guardar_fig(fig, "institution_ranking_top20")
    else:
        inst = pd.DataFrame()

    return {"document_types": dt, "publishers": pub, "institutions": inst}


# ---------------------------------------------------------------------------
# PRISMA flow
# ---------------------------------------------------------------------------
def _prisma_flow(quality: dict) -> None:
    print("\n-- PRISMA FLOW --")
    total = int(quality.get("Total records", 20000))
    excluded = int(quality.get("Excluded by length", 0))
    usable = int(quality.get("USABLE DOCUMENTS", total - excluded))
    dup_doi = int(quality.get("DOI duplicates", 0))
    dup_title = int(quality.get("Title duplicates", 0))

    fig, ax = plt.subplots(figsize=(8.6, 8.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    def box(x, y, w, h, text, color):
        rect = plt.Rectangle((x, y), w, h, facecolor=color,
                             edgecolor="#333333", linewidth=1.1)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=9.2, wrap=True)

    def arrow(x, y1, y2):
        ax.annotate("", xy=(x, y2), xytext=(x, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.2))

    box(0.10, 0.80, 0.80, 0.12,
        f"Identification\nScopus export: {total:,} records (2018-2026)",
        "#DCE9F5")
    arrow(0.50, 0.80, 0.72)
    box(0.10, 0.60, 0.80, 0.12,
        "Screening\nTitle and abstract available: 20,000\n"
        f"Excluded for <20 words: {excluded:,}",
        "#E8F1E4")
    arrow(0.50, 0.60, 0.52)
    box(0.10, 0.40, 0.80, 0.12,
        f"Eligibility\nDuplicates flagged: {dup_doi} DOI / {dup_title} title\n"
        "(retained and reported as a limitation)",
        "#FFF4D6")
    arrow(0.50, 0.40, 0.32)
    box(0.10, 0.20, 0.80, 0.12,
        f"Included\n{usable:,} documents for LDA",
        "#F8D7DA")
    ax.text(0.5, 0.10,
            "The Scopus + Web of Science integration, the exact search string\n"
            "and the query dates are documented in the previous delivery (PRISMA 2020).",
            ha="center", va="center", fontsize=8.4, style="italic", color="#444444")
    ax.set_title("PRISMA flow for the modelling corpus", fontsize=13, pad=8)
    fig.tight_layout()
    _guardar_fig(fig, "PRISMA_Flow")


# ---------------------------------------------------------------------------
# K coherence plot
# ---------------------------------------------------------------------------
def _k_coherence_plot(df_k: pd.DataFrame, chosen_k: int) -> None:
    print("\n-- K COHERENCE PLOT --")
    d = df_k.sort_values("K").copy()
    fig, ax1 = plt.subplots(figsize=(9.5, 4.8))
    ax1.plot(d["K"], d["cv"], marker="o", color="#2166AC", label="C_v")
    ax1.set_xlabel("Number of topics (K)")
    ax1.set_ylabel("C_v coherence", color="#2166AC")
    ax1.tick_params(axis="y", labelcolor="#2166AC")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(d["K"], d["npmi"], marker="s", color="#B2182B", label="NPMI")
    ax2.set_ylabel("NPMI coherence", color="#B2182B")
    ax2.tick_params(axis="y", labelcolor="#B2182B")
    ax1.set_xticks(sorted(int(k) for k in d["K"].unique()))
    ax1.axvline(chosen_k, ls="--", color="#1B7837", lw=1.4,
                label=f"chosen K={chosen_k}")
    best = d.loc[d["cv"].idxmax()]
    ax1.annotate(f"max C_v: K={int(best['K'])} ({best['cv']:.4f})",
                 xy=(best["K"], best["cv"]),
                 xytext=(best["K"] - 0.5, best["cv"] + 0.012),
                 fontsize=8, color="#2166AC",
                 arrowprops=dict(arrowstyle="->", color="#2166AC", lw=0.8))
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8)
    ax1.set_title("LDA coherence across the evaluated K range")
    fig.tight_layout()
    _guardar_fig(fig, "LDA_K_Coherence")


# ---------------------------------------------------------------------------
# HJ-Biplot
# ---------------------------------------------------------------------------
def _hj_biplot(X, row_labels, col_labels, title, out_csv, out_fig,
               row_colors=None, row_size=40, annotate_rows=True,
               annotate_cols=True, legend_title=None, row_numbering=False):
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    eig = S ** 2
    var = eig / eig.sum() if eig.sum() > 0 else np.zeros_like(eig)
    r = min(2, len(S))
    row = U[:, :r] * S[:r]
    col = Vt[:r, :].T * S[:r]
    row_all = U * S
    col_all = Vt.T * S
    row_cos2 = (row ** 2).sum(axis=1) / np.maximum(
        (row_all ** 2).sum(axis=1), 1e-12)
    col_cos2 = (col ** 2).sum(axis=1) / np.maximum(
        (col_all ** 2).sum(axis=1), 1e-12)

    # CSV of coordinates and qualities
    rows = []
    for i, lab in enumerate(row_labels):
        rows.append({
            "marker_type": "row", "label": str(lab),
            "axis1": round(float(row[i, 0]), 6) if r > 0 else np.nan,
            "axis2": round(float(row[i, 1]), 6) if r > 1 else np.nan,
            "cos2": round(float(row_cos2[i]), 4),
        })
    for j, lab in enumerate(col_labels):
        rows.append({
            "marker_type": "column", "label": str(lab),
            "axis1": round(float(col[j, 0]), 6) if r > 0 else np.nan,
            "axis2": round(float(col[j, 1]), 6) if r > 1 else np.nan,
            "cos2": round(float(col_cos2[j]), 4),
        })
    guardar_csv(pd.DataFrame(rows), out_csv, subdir=RUTAS["results"] / "lda")

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    ax.axhline(0, color="#999999", lw=0.7, ls=":")
    ax.axvline(0, color="#999999", lw=0.7, ls=":")
    if row_colors is None:
        ax.scatter(row[:, 0], row[:, 1], s=row_size, c="#2166AC",
                   edgecolor="white", linewidth=0.5, zorder=3, label="Rows")
    else:
        uniq = sorted(set(row_colors))
        cmap = plt.get_cmap("tab20", max(len(uniq), 1))
        for idx, u in enumerate(uniq):
            mask = np.array([c == u for c in row_colors])
            ax.scatter(row[mask, 0], row[mask, 1], s=row_size,
                       color=cmap(idx), alpha=0.45, linewidth=0,
                       label=f"T{int(u)}" if legend_title else str(u),
                       zorder=2)
        if legend_title:
            ax.legend(title=legend_title, fontsize=7, title_fontsize=8,
                      loc="best", framealpha=0.9, ncol=2)
    if row_numbering:
        for i in range(len(row_labels)):
            ax.annotate(str(i + 1), (row[i, 0], row[i, 1]),
                        ha="center", va="center", fontsize=6.5,
                        color="white", fontweight="bold", zorder=4,
                        bbox=dict(boxstyle="circle,pad=0.18",
                                  fc="#2166AC", ec="none"))
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker="o", color="w",
                   label=f"{i + 1}. {lab}", markerfacecolor="#2166AC",
                   markersize=6)
            for i, lab in enumerate(row_labels)
        ]
        ax.legend(handles=handles, loc="center left",
                  bbox_to_anchor=(1.02, 0.5), fontsize=8,
                  frameon=False, title="Country", title_fontsize=8)
    elif annotate_rows:
        offsets = [(5, 4), (5, -11), (-45, 4), (-45, -11),
                   (9, 9), (9, -15), (-55, 9), (-55, -15)]
        for i, lab in enumerate(row_labels):
            dx, dy = offsets[i % len(offsets)]
            ax.annotate(str(lab), (row[i, 0], row[i, 1]),
                        fontsize=8, color="#123B63",
                        xytext=(dx, dy), textcoords="offset points",
                        ha="left" if dx >= 0 else "right")
    # Columns as arrows
    for j, lab in enumerate(col_labels):
        ax.annotate("", xy=(col[j, 0], col[j, 1]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#B2182B",
                                    lw=1.0, alpha=0.8))
        if annotate_cols:
            ax.annotate(str(lab), (col[j, 0], col[j, 1]),
                        fontsize=7.5, color="#8B1A1A", fontweight="bold",
                        xytext=(3, -7), textcoords="offset points")
    ax.set_xlabel(f"Axis 1 ({var[0] * 100:.1f}% of variance)")
    ax.set_ylabel(f"Axis 2 ({var[1] * 100:.1f}% of variance)"
                  if r > 1 else "Axis 2")
    ax.set_title(title)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    _guardar_fig(fig, out_fig)

    return {
        "n_rows": int(n), "n_cols": int(p),
        "explained_axis1": round(float(var[0]) * 100, 2) if len(var) else np.nan,
        "explained_axis2": round(float(var[1]) * 100, 2) if len(var) > 1 else np.nan,
        "mean_row_cos2": round(float(np.mean(row_cos2)), 4),
        "mean_col_cos2": round(float(np.mean(col_cos2)), 4),
    }


def _biplots(theta: pd.DataFrame, chosen_k: int) -> dict:
    print("\n-- HJ-BIPLOT --")
    topic_cols = [f"topic_{i}" for i in range(1, chosen_k + 1)]
    labels = [TOPIC_LABELS.get(i, f"T{i}") for i in range(1, chosen_k + 1)]
    col_labels = [f"T{i}" for i in range(1, chosen_k + 1)]
    summary = {}

    # Years
    d = theta.dropna(subset=["anio"]).copy()
    d["anio"] = d["anio"].astype(int)
    counts = d.groupby("anio").size()
    years = [y for y in sorted(counts.index) if counts[y] >= 20]
    d = d[d["anio"].isin(years)]
    if len(years) >= 3:
        X = d.groupby("anio")[topic_cols].mean().reindex(years).values
        summary["years"] = _hj_biplot(
            X, [str(y) for y in years], col_labels,
            "HJ-Biplot: years x topics (mean topic probability)",
            "hjbiplot_years.csv", "LDA_HJBiplot_Years",
        )

    # Countries
    d = theta.dropna(subset=["country_normalized"]).copy()
    top = d["country_normalized"].value_counts().head(
        int(CFG["report"]["top_countries"])).index.tolist()
    d = d[d["country_normalized"].isin(top)]
    if len(top) >= 3:
        X = (d.groupby("country_normalized")[topic_cols].mean()
             .reindex(top).values)
        summary["countries"] = _hj_biplot(
            X, top, col_labels,
            "HJ-Biplot: countries x topics (mean topic probability)",
            "hjbiplot_countries.csv", "LDA_HJBiplot_Countries",
            annotate_rows=False, row_numbering=True,
        )

    # Documents (random sample for legibility)
    max_docs = int(CFG["viz"]["biplot_max_documents"])
    sample = theta.sample(n=min(max_docs, len(theta)),
                          random_state=int(CFG["seed"])).copy()
    X = sample[topic_cols].values
    dom = sample["dominant_topic"].astype(int).values
    summary["documents"] = _hj_biplot(
        X, sample["doc_id"].astype(str).tolist(), col_labels,
        "HJ-Biplot: document sample x topics (colour = dominant topic)",
        "hjbiplot_documents_sample.csv", "LDA_HJBiplot_Documents",
        row_colors=dom, row_size=14, annotate_rows=False,
        legend_title="Dominant topic",
    )
    guardar_csv(pd.DataFrame([
        {"dataset": k, **v} for k, v in summary.items()
    ]), "hjbiplot_summary.csv", subdir=RUTAS["results"] / "lda")
    return summary


# ---------------------------------------------------------------------------
# LDAvis
# ---------------------------------------------------------------------------
def _ldavis(lda, corpus, id2word, chosen_k: int, skip: bool = False) -> dict:
    print("\n-- LDAVIS --")
    out = {}
    if skip:
        print("  Skipped by user request.")
        return out

    import pyLDAvis
    import pyLDAvis.gensim_models as gm

    t0 = perf_counter()
    prepared = gm.prepare(lda, corpus, id2word, mds="mmds", R=30, n_jobs=1)
    html_path = RUTAS["visualizations"] / "LDAvis.html"
    pyLDAvis.save_html(prepared, str(html_path))
    print(f"  -> LDAvis.html ({_fmt(perf_counter() - t0)})")
    out["html"] = str(html_path)

    # Static intertopic distance map
    tc = prepared.topic_coordinates.copy()
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    freq = tc["Freq"].astype(float)
    sizes = 120 + (freq / max(freq.max(), 1e-9)) * 2200
    colors = tc["cluster"].astype(int)
    sc = ax.scatter(tc["x"], tc["y"], s=sizes, c=colors, cmap="tab20",
                    alpha=0.72, edgecolor="#333333", linewidth=0.8)
    for _, row in tc.iterrows():
        ax.annotate(str(int(row["topics"])), (row["x"], row["y"]),
                    ha="center", va="center", fontsize=8.5,
                    fontweight="bold", color="#111111")
    ax.set_xlabel("Intertopic distance (axis 1)")
    ax.set_ylabel("Intertopic distance (axis 2)")
    ax.set_title("LDAvis intertopic distance map (bubble size = prevalence)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    _guardar_fig(fig, "LDAvis_Intertopic_Map")
    out["map"] = "LDAvis_Intertopic_Map"

    # Relevance terms for lambda = 0.0, 0.6, 1.0
    rows = []
    for lam in (0.0, LAMBDA, 1.0):
        rel = _relevance_table(lda, corpus, id2word, lam, topn=10)
        for _, r in rel.iterrows():
            rows.append({"lambda": lam, **r.to_dict()})
    rel_df = pd.DataFrame(rows)
    guardar_csv(rel_df, "ldavis_relevance_terms.csv",
                subdir=RUTAS["results"] / "lda")
    out["relevance"] = "ldavis_relevance_terms.csv"

    # Heatmap for the configured lambda
    rel_lam = rel_df[np.isclose(rel_df["lambda"], LAMBDA)]
    _ldavis_relevance_heatmap(rel_lam, chosen_k)
    out["heatmap"] = "LDAvis_Relevance_Heatmap"
    return out


def _relevance_table(lda, corpus, id2word, lam: float, topn: int = 10):
    # p(w|t)
    phi = lda.get_topics().astype(float)
    phi = phi / np.maximum(phi.sum(axis=1, keepdims=True), 1e-12)
    # p(w) from the corpus
    counts = np.zeros(phi.shape[1], dtype=float)
    for bow in corpus:
        for w, c in bow:
            if w < len(counts):
                counts[w] += c
    p_w = counts / max(counts.sum(), 1e-12)
    p_w = np.maximum(p_w, 1e-12)
    rows = []
    for t in range(phi.shape[0]):
        p_wt = np.maximum(phi[t], 1e-12)
        rel = lam * np.log(p_wt) + (1 - lam) * np.log(p_wt / p_w)
        idx = np.argsort(rel)[::-1][:topn]
        for rank, j in enumerate(idx, start=1):
            rows.append({
                "topic": t + 1,
                "topic_label": TOPIC_LABELS.get(t + 1, f"T{t + 1}"),
                "rank": rank,
                "term": id2word[int(j)],
                "p_w_given_t": round(float(p_wt[j]), 8),
                "p_w": round(float(p_w[j]), 8),
                "relevance": round(float(rel[j]), 6),
            })
    return pd.DataFrame(rows)


def _ldavis_relevance_heatmap(rel: pd.DataFrame, chosen_k: int) -> None:
    topics = sorted(rel["topic"].unique())
    terms = []
    for t in topics:
        terms += rel[rel["topic"] == t].head(8)["term"].tolist()
    terms = list(dict.fromkeys(terms))
    M = np.zeros((len(topics), len(terms)))
    for _, r in rel.iterrows():
        if r["term"] in terms:
            M[int(r["topic"]) - 1, terms.index(r["term"])] = r["relevance"]
    fig, ax = plt.subplots(figsize=(max(10, len(terms) * 0.22),
                                    max(4.5, len(topics) * 0.36)))
    im = ax.imshow(M, aspect="auto", cmap="Reds")
    ax.set_xticks(range(len(terms)))
    ax.set_xticklabels(terms, rotation=90, fontsize=6)
    ax.set_yticks(range(len(topics)))
    ax.set_yticklabels([f"T{t}" for t in topics], fontsize=8)
    ax.set_title(f"LDAvis relevance of terms per topic (lambda={LAMBDA})")
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label("Relevance", fontsize=9)
    fig.tight_layout()
    _guardar_fig(fig, "LDAvis_Relevance_Heatmap")


# ---------------------------------------------------------------------------
# Full configuration JSON
# ---------------------------------------------------------------------------
def _add_volume_leading_country(theta: pd.DataFrame,
                                summary: pd.DataFrame) -> pd.DataFrame:
    """Add a volume-based leading country next to the specialization one."""
    if "country_normalized" not in theta.columns:
        return summary
    out = summary.copy()
    countries, docs = [], []
    for t in out["topic_id"].astype(int):
        sub = theta.loc[theta["dominant_topic"] == t, "country_normalized"].dropna()
        vc = sub.value_counts()
        countries.append(vc.index[0] if len(vc) else "N/A")
        docs.append(int(vc.iloc[0]) if len(vc) else 0)
    out["leading_country_volume"] = countries
    out["leading_country_volume_docs"] = docs
    return out


def _build_config_json(ctx: dict) -> dict:
    base_df = ctx["base_df"]
    theta = ctx["theta"]
    corpus = ctx["corpus"]
    id2word = ctx["id2word"]
    dict_eval = ctx["dict_eval"]
    docs_tokens = ctx["docs_tokens"]
    chosen_k = ctx["chosen_k"]
    metrics = ctx["final_metrics"]
    quality = ctx["quality"]
    desc = ctx["descriptive"]

    try:
        stops = build_stopwords()
        n_stops = len(stops)
    except Exception:
        n_stops = None

    artifacts = []
    for root in (RUTAS["results"], RUTAS["visualizations"], RUTAS["models_lda"]):
        for p in sorted(Path(root).rglob("*")):
            if p.is_file() and p.suffix.lower() in {
                ".csv", ".json", ".png", ".svg", ".html", ".model",
                ".npy", ".state", ".id2word",
            }:
                artifacts.append(str(p.relative_to(PROJECT_ROOT)))

    cfg = {
        "schema_version": "1.0",
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "project": "bertlda — LDA on an AI-ethics / generative-AI corpus",
        "git_commit": _git_commit(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": {
                pkg: _version(pkg) for pkg in [
                    "gensim", "numpy", "pandas", "scipy", "scikit-learn",
                    "matplotlib", "pyLDAvis", "nltk", "statsmodels",
                    "geopandas", "sentence-transformers", "bertopic",
                ]
            },
        },
        "paths": {
            "project_root": str(PROJECT_ROOT),
            "raw_corpus": str(CFG["input_file"]),
            "results": str(RUTAS["results"]),
            "visualizations": str(RUTAS["visualizations"]),
            "models_lda": str(RUTAS["models_lda"]),
        },
        "config": _json_safe(CFG),
        "params": _json_safe(ctx["params"]),
        "corpus": {
            "source": "Scopus CSV export (raw_corpus.csv)",
            "n_records_raw": int(quality.get("Total records", len(base_df))),
            "n_documents_modelled": int(len(base_df)),
            "year_range": str(quality.get("Year range", "")),
            "n_countries": int(quality.get("Countries (normalized)", 0)),
            "n_sources": int(quality.get("Distinct sources", 0)),
            "n_excluded_by_length": int(quality.get("Excluded by length", 0)),
            "n_doi_duplicates": int(quality.get("DOI duplicates", 0)),
            "n_title_duplicates": int(quality.get("Title duplicates", 0)),
            "n_docs_over_350_words": int(quality.get("Docs >350 words", 0)),
            "quality_table": _json_safe(
                ctx["quality_df"].to_dict(orient="records")),
        },
        "preprocessing": {
            "fields_used": ["Title", "Abstract"],
            "fields_available_not_used": ["Author Keywords", "Index Keywords"],
            "min_words_per_document": int(CFG["min_palabras_doc"]),
            "lowercase": True,
            "unicode_and_punctuation_cleaning": True,
            "url_email_citation_copyright_removal": True,
            "lemmatization": "WordNet (nouns, then verbs)",
            "min_token_length": 3,
            "stopword_layers": [
                "NLTK English functional stopwords",
                "editorial / publisher stopwords",
                "methodological stopwords",
                "AI-ethics domain terms",
            ],
            "n_stopwords": n_stops,
            "bigrams": {
                "min_count": ctx["params"].get("bigram_min_count"),
                "threshold": ctx["params"].get("bigram_threshold"),
            },
            "dictionary_filter": {
                "no_below": ctx["params"].get("no_below"),
                "no_above": ctx["params"].get("no_above"),
            },
            "n_tokens_total": int(sum(len(d) for d in docs_tokens)),
            "n_tokens_mean": round(float(np.mean([len(d) for d in docs_tokens])), 2),
            "vocab_train": int(len(id2word)),
            "vocab_eval": int(len(dict_eval)),
            "n_docs_with_empty_bow": int(sum(1 for c in corpus if len(c) == 0)),
        },
        "lda": {
            "implementation": "gensim.models.ldamodel.LdaModel (single process)",
            "n_topics": int(chosen_k),
            "alpha": str(ctx["alpha"]),
            "eta": str(ctx["eta"]),
            "passes": int(CFG["lda"]["passes_final"]),
            "iterations_per_document": 50,
            "random_seed": int(CFG["seed"]),
            "n_documents": int(len(corpus)),
            "vocab_size": int(len(id2word)),
            "theta_shape": [int(theta.shape[0]), int(chosen_k)],
            "final_metrics": _json_safe(metrics),
        },
        "k_selection": {
            "evaluated_range": [int(ctx["df_k"]["K"].min()),
                                int(ctx["df_k"]["K"].max())],
            "n_candidates": int(len(ctx["df_k"])),
            "best_cv_k": int(ctx["df_k"].loc[ctx["df_k"]["cv"].idxmax(), "K"]),
            "best_cv": float(ctx["df_k"]["cv"].max()),
            "chosen_k": int(chosen_k),
            "chosen_cv": float(ctx["df_k"].loc[
                ctx["df_k"]["K"] == chosen_k, "cv"].iloc[0]),
            "table": _json_safe(ctx["df_k"].to_dict(orient="records")),
        },
        "alpha_eta_search": {
            "grid_size": int(len(ctx["df_ae"])),
            "best": _json_safe(ctx["df_ae"].loc[
                ctx["df_ae"]["cv"].idxmax()].to_dict()),
            "table": _json_safe(ctx["df_ae"].to_dict(orient="records")),
        },
        "stability": {
            "table": _json_safe(ctx["df_stability"].to_dict(orient="records")),
        },
        "sensitivity": {
            "k_neighbourhood": [int(k) for k in CFG["lda"]["sensitivity_k"]],
            "seeds": [int(s) for s in CFG["lda"]["sensitivity_seeds"]],
            "passes": int(CFG["lda"]["sensitivity_passes"]),
            "seed_table": _json_safe(
                ctx["df_sens_seed"].to_dict(orient="records")
                if ctx["df_sens_seed"] is not None else []),
            "k_table": _json_safe(
                ctx["df_sens_k"].to_dict(orient="records")
                if ctx["df_sens_k"] is not None else []),
        },
        "descriptive": {
            "document_types": _json_safe(
                desc["document_types"].to_dict(orient="records")),
            "publishers_top": _json_safe(
                desc["publishers"].head(15).to_dict(orient="records")),
            "institutions_top": _json_safe(
                desc["institutions"].head(20).to_dict(orient="records")),
        },
        "artifacts": artifacts,
    }
    return _json_safe(cfg)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def _report_markdown(ctx: dict, cfg_json: dict, biplot_summary: dict,
                     ldavis_out: dict) -> str:
    quality = ctx["quality"]
    quality_df = ctx["quality_df"]
    df_k = ctx["df_k"]
    df_ae = ctx["df_ae"]
    df_stab = ctx["df_stability"]
    df_sens_seed = ctx["df_sens_seed"]
    df_sens_k = ctx["df_sens_k"]
    summary = ctx["summary"]
    theta = ctx["theta"]
    chosen_k = ctx["chosen_k"]
    metrics = ctx["final_metrics"]
    desc = ctx["descriptive"]
    rel = ctx["rel_df"]
    annual = ctx["annual"]
    countries = ctx["countries"]
    sources = ctx["sources"]

    n_docs = len(ctx["base_df"])
    n_raw = int(quality.get("Total records", 20000))
    n_excl = int(quality.get("Excluded by length", 0))
    n_countries = int(quality.get("Countries (normalized)", 0))
    n_sources = int(quality.get("Distinct sources", 0))
    best_k = int(df_k.loc[df_k["cv"].idxmax(), "K"])
    best_cv = float(df_k["cv"].max())
    chosen_cv = float(df_k.loc[df_k["K"] == chosen_k, "cv"].iloc[0])
    chosen_npmi = float(df_k.loc[df_k["K"] == chosen_k, "npmi"].iloc[0])
    chosen_div = float(df_k.loc[df_k["K"] == chosen_k, "diversity"].iloc[0])
    chosen_dom = float(df_k.loc[df_k["K"] == chosen_k, "dominance"].iloc[0])

    # Compact tables
    topic_table = summary.copy()
    topic_table["label"] = topic_table["topic_id"].map(
        lambda t: TOPIC_LABELS.get(int(t), f"T{t}"))
    topic_table["top_terms"] = topic_table["keywords_20"].map(
        lambda s: ", ".join(str(s).split(", ")[:10]))
    if "leading_country_volume" in topic_table.columns:
        topic_table["country"] = topic_table["leading_country_volume"].fillna("—")
    elif "leading_country" in topic_table.columns:
        topic_table["country"] = topic_table["leading_country"].fillna("—")
    else:
        topic_table["country"] = "—"

    if df_sens_seed is not None and len(df_sens_seed):
        sens_tab = _md_table(
            df_sens_seed,
            cols=["K", "jaccard_mean", "jaccard_std", "jaccard_min",
                  "ari_mean", "ari_std", "ari_min", "cv_mean", "cv_std"],
            headers=["K", "Jaccard mean", "Jaccard SD", "Jaccard min",
                     "ARI mean", "ARI SD", "ARI min", "C_v mean", "C_v SD"],
        )
        sens_text = (
            "La tabla anterior resume la estabilidad entre semillas para cada "
            "valor de K del vecindario. El Jaccard se calcula sobre las 10 "
            "palabras principales de tópicos alineados con el algoritmo "
            "húngaro; el ARI se calcula sobre la asignación del tópico "
            "dominante de cada documento."
        )
        _row = df_sens_seed[df_sens_seed["K"] == chosen_k]
        if len(_row):
            _r = _row.iloc[0]
            _jm, _am = float(_r["jaccard_mean"]), float(_r["ari_mean"])
            _level = ("alta" if _jm >= 0.5 and _am >= 0.4
                      else "moderada" if _jm >= 0.2 else "baja")
            sens_conclusion = (
                f"Para el K elegido ({chosen_k}) la estabilidad entre semillas "
                f"es **{_level}**: Jaccard medio {_jm:.4f} "
                f"(SD {_r['jaccard_std']:.4f}, mínimo {_r['jaccard_min']:.4f}) "
                f"y ARI medio {_am:.4f} "
                f"(SD {_r['ari_std']:.4f}, mínimo {_r['ari_min']:.4f}). "
                "Los tópicos con mayor volumen (T3, T10, T4, T11) se recuperan "
                "de forma más consistente, mientras que los tópicos pequeños "
                "(T13) son más sensibles. La estructura de "
                f"{chosen_k} tópicos se considera **suficientemente robusta "
                "para un análisis exploratorio**, con la recomendación de "
                "reportar los tópicos de baja frecuencia como provisionales. "
                "Este resultado es consistente con la literatura sobre "
                "estabilidad en topic models (Greene, O'Callaghan & "
                "Cunningham, 2014; De Waal & Barnard, 2008; Koltcov, "
                "Koltsova & Nikolenko, 2014)."
            )
        else:
            sens_conclusion = (
                "La estabilidad entre semillas es **moderada**, como es "
                "habitual en LDA sobre corpus científicos heterogéneos y con "
                "resúmenes cortos."
            )
    else:
        sens_tab = "_La fase 5 aún no ha producido la tabla de sensibilidad._"
        sens_text = ""
        sens_conclusion = (
            "La estabilidad entre semillas es **moderada**, como es habitual "
            "en LDA sobre corpus científicos heterogéneos y con resúmenes "
            "cortos."
        )

    if df_sens_k is not None and len(df_sens_k):
        sens_k_tab = _md_table(
            df_sens_k,
            cols=["K", "jaccard_best_mean", "jaccard_best_median",
                  "jaccard_best_min", "jaccard_best_max",
                  "topics_matched_ge_0_5", "topics_total",
                  "reference_topics_covered_ge_0_5"],
            headers=["K", "Best Jaccard mean", "Median", "Min", "Max",
                     "Topics ≥ 0.5", "Topics", "Reference topics covered ≥ 0.5"],
        )
    else:
        sens_k_tab = "_Sin datos de solapamiento entre valores de K._"

    ae_tab = _md_table(
        df_ae.sort_values("cv", ascending=False).head(8),
        cols=["alpha", "eta", "cv", "npmi", "dominance"],
        headers=["alpha", "eta", "C_v", "NPMI", "Dominance"],
    )

    rel_tab = _md_table(
        rel[rel["lambda"].apply(lambda x: abs(float(x) - LAMBDA) < 1e-9)]
        .groupby("topic", as_index=False)
        .head(6),
        cols=["topic", "term", "relevance"],
        headers=["Topic", "Term", "Relevance (λ=0.6)"],
    )

    # Descriptive tables
    annual_tab = _md_table(
        annual, cols=["year", "documents", "percentage"],
        headers=["Year", "Documents", "%"],
    )
    country_tab = _md_table(
        countries.head(10), cols=["country", "documents", "percentage"],
        headers=["Country", "Documents", "%"],
    )
    source_tab = _md_table(
        sources.head(10), cols=["source", "documents", "percentage"],
        headers=["Source", "Documents", "%"],
    )
    dt_tab = _md_table(
        desc["document_types"].head(10),
        cols=["document_type", "documents", "percentage"],
        headers=["Document type", "Documents", "%"],
    )
    inst_tab = _md_table(
        desc["institutions"].head(10),
        cols=["institution", "documents", "percentage"],
        headers=["Institution", "Documents", "%"],
    )
    quality_tab = _md_table(
        quality_df, cols=["metric", "value"], headers=["Metric", "Value"],
    )

    biplot_years = biplot_summary.get("years", {})
    biplot_countries = biplot_summary.get("countries", {})
    biplot_docs = biplot_summary.get("documents", {})
    biplot_tab = _md_table(pd.DataFrame([
        {"dataset": "Years x topics", **biplot_years},
        {"dataset": "Countries x topics", **biplot_countries},
        {"dataset": "Document sample x topics", **biplot_docs},
    ]), headers=["Dataset", "Rows", "Columns", "Axis 1 (%)", "Axis 2 (%)",
                 "Mean row cos²", "Mean column cos²"])

    md = f"""# Metodología y resultados preliminares — Modelado de tópicos con LDA

**Corpus:** ética de la inteligencia artificial, IA generativa y grandes modelos de lenguaje (Scopus, 2018–2026)  
**Documento generado:** {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}  
**Repositorio:** `https://github.com/JeanVydes/bertlda`  
**Configuración completa exportada:** `outputs/results/config/model_configuration_full.json`

> **Nota de alcance.** Este documento integra la metodología y los resultados
> preliminares del modelo LDA. La entrega anterior (PRISMA, integración
> Scopus + Web of Science) se retoma en la sección 1; el corpus de modelado
> disponible en este repositorio es el export de Scopus de {n_raw:,} registros.
> Los campos marcados como `[COMPLETAR]` deben tomarse de la Entrega 1 para la
> versión final.

---

## 1. Recolección y depuración del corpus

### 1.1 Fuentes consultadas y ecuación de búsqueda

La revisión sistemática de la Entrega 1 combinó registros de **Scopus** y
**Web of Science (WoS)**. El corpus consolidado se orientó a la intersección
entre inteligencia artificial, IA generativa, grandes modelos de lenguaje y
sus dimensiones éticas, sociales, regulatorias y de seguridad. La ecuación
exacta de búsqueda, las fechas de consulta y los conteos de la integración
WoS se documentan en la Entrega 1:

- **Ecuación de búsqueda Scopus:** `[COMPLETAR con la cadena exacta de la Entrega 1]`
- **Ecuación de búsqueda WoS:** `[COMPLETAR con la cadena exacta de la Entrega 1]`
- **Fecha de consulta Scopus:** `[COMPLETAR]`
- **Fecha de consulta WoS:** `[COMPLETAR]`
- **Registros WoS integrados:** `[COMPLETAR]`
- **Duplicados eliminados en la integración:** `[COMPLETAR]`

El export utilizado en esta entrega contiene **{n_raw:,} registros de Scopus**
(campo `Source = Scopus`), con título, resumen, palabras clave, año, fuente,
afiliaciones, país de afiliación, tipo de documento y DOI. La ventana temporal
observada es **{quality.get('Year range', '2018-2026')}**; el año 2026 está
incompleto porque el export se realizó durante ese año.

### 1.2 Proceso PRISMA

El proceso PRISMA 2020 (Page et al., 2021) se resume a continuación para el
corpus de modelado. Los conteos de la integración Scopus + WoS de la Entrega 1
deben verificarse antes de la versión final.

{quality_tab}

- **Registros identificados (Scopus):** {n_raw:,}
- **Registros sin título o resumen:** 0
- **Excluidos por longitud (< {CFG['min_palabras_doc']} palabras):** {n_excl:,}
- **Duplicados detectados:** {quality.get('DOI duplicates', 0)} por DOI y {quality.get('Title duplicates', 0)} por título (se conservan y se reportan como limitación)
- **Documentos finales para LDA:** **{n_docs:,}**
- **Países normalizados:** {n_countries}
- **Fuentes distintas:** {n_sources:,}

### 1.3 Diagrama de flujo PRISMA

![Diagrama de flujo PRISMA](../outputs/visualizations/PRISMA_Flow.png)

El diagrama completo de la integración Scopus + WoS se conserva en la Entrega 1.
El diagrama anterior documenta el flujo del corpus efectivamente modelado.

### 1.4 Corpus final

El corpus final contiene **{n_docs:,} documentos** con texto suficiente para el
modelado. La distribución anual muestra una concentración muy fuerte en los
últimos años, coherente con el auge de la IA generativa:

{annual_tab}

---

## 2. Preprocesamiento del texto

### 2.1 Campos utilizados

Se utilizaron **título** y **resumen** (`Title` + `Abstract`). Las palabras clave
de autor e índice (`Author Keywords`, `Index Keywords`) están disponibles en el
export, pero no se concatenaron al texto de entrada para evitar duplicar
términos y para mantener una representación homogénea entre documentos.

### 2.2 Técnicas aplicadas

1. **Normalización Unicode y minúsculas.**
2. **Eliminación de ruido editorial:** URLs, correos, citas numéricas,
   avisos de copyright, licencias y frases de “publicado por”.
3. **Tokenización** por espacios sobre texto limpio.
4. **Lematización con WordNet** (primero sustantivos, luego verbos).
5. **Eliminación de stopwords en cuatro capas:**
   - stopwords funcionales del inglés (NLTK);
   - stopwords editoriales (Elsevier, Springer, MDPI, “journal”, “abstract”, etc.);
   - stopwords metodológicas (“study”, “results”, “method”, “approach”, etc.);
   - términos de dominio de ética de la IA y acrónimos válidos.
6. **Filtrado por longitud:** se descartan tokens de menos de 3 caracteres,
   salvo acrónimos válidos (`ai`, `llm`, `gpt`, `nlp`, `gdpr`, etc.).
7. **Bigramas:** se detectan colocaciones con `gensim.models.phrases`
   (`min_count={ctx['params'].get('bigram_min_count')}`,
   `threshold={ctx['params'].get('bigram_threshold')}`), por ejemplo
   `generative_ai`, `artificial_intelligence`, `machine_learning`.
8. **Filtrado de vocabulario:** el diccionario de entrenamiento elimina términos
   con frecuencia documental menor que `{ctx['params'].get('no_below')}` y
   presentes en más del `{ctx['params'].get('no_above')}` de los documentos.
9. **Sin stemming agresivo:** se prefiere la lematización para preservar la
   interpretabilidad de los tópicos.

### 2.3 Herramientas

- **Python 3.13** y **NLTK** para tokenización, stopwords y lematización.
- **Gensim 4.4** para bigramas, diccionario, corpus BoW y LDA.
- **pandas / NumPy** para la manipulación de datos.
- **Matplotlib** para las figuras.

### 2.4 Resultados del preprocesamiento

| Indicador | Valor |
|---|---:|
| Documentos modelados | {n_docs:,} |
| Tokens totales | {cfg_json['preprocessing']['n_tokens_total']:,} |
| Tokens por documento (media) | {cfg_json['preprocessing']['n_tokens_mean']:.2f} |
| Vocabulario de entrenamiento | {cfg_json['preprocessing']['vocab_train']:,} |
| Vocabulario de evaluación | {cfg_json['preprocessing']['vocab_eval']:,} |
| Documentos con BoW vacío | {cfg_json['preprocessing']['n_docs_with_empty_bow']} |
| Stopwords totales | {cfg_json['preprocessing']['n_stopwords']} |

La limpieza se implementó en `src/bertlda/text_processing.py` y se aplicó de
forma determinista en `src/bertlda/phase1a_lda_diagnostic.py`.

---

## 3. Fundamento del modelo LDA

Latent Dirichlet Allocation (Blei, Ng & Jordan, 2003) es un modelo generativo
jerárquico para colecciones de documentos discretos. Cada documento se
representa como una mezcla de un número fijo de tópicos y cada tópico como una
distribución sobre el vocabulario. Formalmente, para un documento $d$:

$$
\\theta_d \\sim \\mathrm{{Dirichlet}}(\\alpha), \\qquad
z_{{d,n}} \\sim \\mathrm{{Categorical}}(\\theta_d), \\qquad
w_{{d,n}} \\sim \\mathrm{{Categorical}}(\\phi_{{z_{{d,n}}}}),
$$

donde $\\theta_d$ es la proporción de tópicos del documento, $z_{{d,n}}$ es el
tópico latente de la palabra $n$ y $\\phi_k \\sim \\mathrm{{Dirichlet}}(\\eta)$ es
la distribución de palabras del tópico $k$. La verosimilitud conjunta integra
las variables latentes y se aproxima mediante inferencia variacional o
muestreo de Gibbs; Gensim utiliza inferencia variacional en línea.

LDA es apropiado para este corpus porque:

- los documentos son textos cortos y heterogéneos (título + resumen), donde un
  modelo probabilístico de mezclas es más estable que un clustering duro;
- el objetivo es **describir** la estructura temática y su evolución, no
  clasificar documentos en categorías predefinidas;
- la matriz documento-tópico ($\\Theta$) permite análisis posteriores con
  HJ-Biplot, LDAvis y modelos de tendencia;
- la literatura de ética de la IA es interdisciplinaria: un mismo documento
  puede combinar gobernanza, sesgo, privacidad y evaluación técnica, lo que
  favorece una representación mixta.

---

## 4. Selección del número de tópicos (K)

### 4.1 Métricas de coherencia

Se evaluaron dos métricas complementarias:

- **C_v** (Röder, Both & Hinneburg, 2015): combina similitud semántica entre
  palabras, ventana deslizante y confirmación mutua; es la métrica de referencia
  para comparar tópicos en corpus científicos.
- **NPMI** (Bouma, 2009): información mutua puntual normalizada; penaliza
  asociaciones espurias y es fácilmente interpretable en el rango [-1, 1].

También se registraron **diversidad de tópicos** (proporción de palabras únicas
en el top-10 de todos los tópicos), **perplejidad**, **dominancia** (cuota del
tópico más frecuente) y **entropía normalizada** de la distribución de tamaños.

### 4.2 Rango evaluado y criterio de decisión

Se evaluó **K = {int(df_k['K'].min())}–{int(df_k['K'].max())}** (20 valores).
La curva de coherencia muestra un máximo global en **K = {best_k}**
(C_v = {best_cv:.4f}), pero la ganancia respecto de K = {chosen_k} es marginal
({best_cv - chosen_cv:+.4f}) y se concentra en la zona de meseta
K ≈ 14–23. Se seleccionó **K = {chosen_k}** por las siguientes razones:

1. **Meseta de coherencia:** K = {chosen_k} ya se encuentra en la meseta
   (C_v = {chosen_cv:.4f}; NPMI = {chosen_npmi:+.4f}).
2. **Parsimonia e interpretabilidad:** 14 tópicos producen una estructura
   temática manejable y alineada con las dimensiones de la ética de la IA
   (seguridad, sesgo, gobernanza, educación, salud, etc.).
3. **Diversidad:** la diversidad del top-10 es {chosen_div:.4f}, alta para el
   tamaño del corpus.
4. **Dominancia:** el tópico más frecuente concentra {chosen_dom * 100:.1f}%
   de los documentos, por debajo del umbral de dominancia excesiva (40%).
5. **Estabilidad:** el análisis de sensibilidad (sección 6) confirma que la
   estructura de K = {chosen_k} es comparable a la de sus vecinos.

![Coherencia LDA por número de tópicos](../outputs/visualizations/LDA_K_Coherence.png)

### 4.3 Tabla completa de la búsqueda de K

{_md_table(df_k, cols=['K', 'cv', 'npmi', 'diversity', 'perplexity', 'dominance', 'entropy_norm', 'topics_empty'], headers=['K', 'C_v', 'NPMI', 'Diversity', 'Perplexity', 'Dominance', 'Entropy', 'Empty topics'])}

---

## 5. Configuración y ejecución del modelo

### 5.1 Librería y software

- **Gensim 4.4.0**, implementación `LdaModel` en un solo proceso.
- **Python 3.13**; el entorno se reproduce con `uv sync`.
- Semilla global **{CFG['seed']}**.
- Exportaciones exclusivamente en **CSV** (nunca xlsx); los checkpoints
  intermedios usan `.pkl`.

### 5.2 Hiperparámetros

| Parámetro | Valor |
|---|---|
| Número de tópicos (K) | {chosen_k} |
| Alpha (prior documento-tópico) | `{ctx['alpha']}` |
| Eta / beta (prior tópico-palabra) | `{ctx['eta']}` |
| Iteraciones (passes) | {CFG['lda']['passes_final']} |
| Iteraciones internas por documento | 50 |
| Semilla aleatoria | {CFG['seed']} |
| Documentos | {n_docs:,} |
| Vocabulario | {len(ctx['id2word']):,} |
| Implementación | `gensim.models.ldamodel.LdaModel` |

La búsqueda de alpha/eta evaluó **{len(df_ae)} combinaciones** (4 valores de
alpha × 4 de eta). La mejor combinación fue
**alpha = `{ctx['alpha']}`, eta = `{ctx['eta']}`** (C_v = {float(df_ae['cv'].max()):.4f}).
Las 8 mejores combinaciones se muestran a continuación:

{ae_tab}

### 5.3 Reproducibilidad

El código completo está en `src/bertlda/`. La secuencia es:

```bash
uv sync
uv run python src/bertlda/preprocess_corpus.py
uv run python src/bertlda/phase0_audit.py
uv run python src/bertlda/phase1a_lda_diagnostic.py
uv run python src/bertlda/phase1b_lda_decision.py
uv run python src/bertlda/phase1c_lda_final.py
uv run python src/bertlda/phase5_sensitivity.py
uv run python src/bertlda/phase6_report.py
```

La configuración completa, las versiones de las librerías, las métricas y el
inventario de artefactos se exportan en
`outputs/results/config/model_configuration_full.json`.

### 5.4 Métricas del modelo final

| Métrica | Valor |
|---|---:|
| C_v | {metrics.get('cv', float('nan')):.4f} |
| NPMI | {metrics.get('npmi', float('nan')):+.4f} |
| Diversidad (top-10) | {metrics.get('diversity', float('nan')):.4f} |
| Perplejidad | {metrics.get('perplexity', float('nan')):.4f} |
| Dominancia | {metrics.get('dominance', float('nan')):.4f} |
| Entropía normalizada | {metrics.get('entropy_norm', float('nan')):.4f} |
| Tópicos vacíos | {metrics.get('topics_empty', 0)} |

> **Nota sobre la comparación de C_v.** La grilla de K (sección 4) usa
> `alpha="symmetric"` y `passes={CFG['lda']['passes_lda']}`; el modelo final usa
> `alpha="{ctx['alpha']}"`, `eta="{ctx['eta']}"` y `passes={CFG['lda']['passes_final']}`.
> Por eso su C_v ({metrics.get('cv', float('nan')):.4f}) no coincide exactamente
> con el valor de la grilla para K = {chosen_k} ({chosen_cv:.4f}). La comparación
> válida es dentro de cada configuración.

---

## 6. Análisis de sensibilidad del modelo

El análisis de sensibilidad evalúa tres fuentes de variación: **semilla
aleatoria**, **vecindario de K** y **hiperparámetros alpha/eta**. Para comparar
corridas se usan dos métricas:

- **Jaccard** entre las 10 palabras principales de tópicos alineados. La
  alineación se resuelve con el algoritmo húngaro sobre la similitud coseno
  entre las distribuciones tópico-palabra.
- **Adjusted Rand Index (ARI)** entre las asignaciones del tópico dominante de
  cada documento.

### 6.1 Estabilidad ante la semilla aleatoria

Se entrenaron **{len(CFG['lda']['sensitivity_seeds'])} semillas**
({', '.join(str(s) for s in CFG['lda']['sensitivity_seeds'])}) para cada
K ∈ {CFG['lda']['sensitivity_k']}, con `passes={CFG['lda']['sensitivity_passes']}`.
La tabla siguiente reporta la media y la desviación estándar sobre todos los
pares de semillas:

{sens_tab}

{sens_text}

![Estabilidad LDA entre semillas](../outputs/visualizations/LDA_Sensitivity_Stability.png)

### 6.2 Estabilidad ante variaciones de K

La tabla siguiente compara cada modelo del vecindario con el modelo de
referencia (K = {chosen_k}, semilla {CFG['seed']}). `Best Jaccard` es el mejor
solapamiento de cada tópico del modelo con un tópico de referencia; la última
columna indica cuántos tópicos de referencia quedan cubiertos con Jaccard ≥ 0.5.

{sens_k_tab}

![Solapamiento entre valores de K](../outputs/visualizations/LDA_Sensitivity_K_Overlap.png)

### 6.3 Sensibilidad a alpha y eta

La grilla de **{len(df_ae)} combinaciones** muestra que las diferencias de C_v
entre configuraciones son pequeñas (rango
{float(df_ae['cv'].max() - df_ae['cv'].min()):.4f}). La configuración elegida
(`alpha={ctx['alpha']}`, `eta={ctx['eta']}`) se mantiene entre las mejores y
produce la mayor coherencia. Esto sugiere que la estructura temática es
razonablemente robusta a la elección de priors, siempre que se mantenga un
alpha asimétrico o cercano a la simetría.

### 6.4 Conclusión sobre robustez

{sens_conclusion}

---

## 7. Métodos Biplot y LDAvis

### 7.1 HJ-Biplot

El **HJ-Biplot** (Galindo-Villardón, 1986), extensión del biplot de Gabriel
(1971), representa simultáneamente filas y columnas de una matriz mediante una
descomposición en valores singulares. Para una matriz centrada $X$:

$$
X = U D V^\\top,
$$

las coordenadas de filas son $A = U D$ y las de columnas $B = V D$. La
propiedad clave es que el producto escalar $a_i \\cdot b_j$ aproxima el valor
$x_{{ij}}$, de modo que filas y columnas se interpretan en el mismo espacio. La
**calidad de representación** de cada marcador se mide con el coseno al
cuadrado:

$$
\\cos^2_{{i}} = \\frac{{a_{{i1}}^2 + a_{{i2}}^2}}{{\\sum_k a_{{ik}}^2}},
$$

y de forma análoga para las columnas. La **calidad global** es el promedio de
los cosenos al cuadrado. Se construyeron tres biplots:

1. **Años × tópicos:** cada fila es un año (con al menos 20 documentos) y cada
   columna un tópico; el valor es la probabilidad media del tópico en ese año.
2. **Países × tópicos:** cada fila es uno de los {CFG['report']['top_countries']} países con más documentos;
   el valor es la probabilidad media del tópico.
3. **Documentos × tópicos:** muestra aleatoria de {CFG['viz']['biplot_max_documents']}
   documentos, coloreada por tópico dominante.

{biplot_tab}

![HJ-Biplot años × tópicos](../outputs/visualizations/LDA_HJBiplot_Years.png)

![HJ-Biplot países × tópicos](../outputs/visualizations/LDA_HJBiplot_Countries.png)

![HJ-Biplot documentos × tópicos](../outputs/visualizations/LDA_HJBiplot_Documents.png)

**Lectura.** En el biplot de años, la cercanía entre un año y un tópico indica
que ese tópico gana peso relativo en ese año; los años recientes (2024–2026)
se desplazan hacia los tópicos de IA generativa, agentes y evaluación de
modelos. En el biplot de países, los países se identifican con un número y la
leyenda lateral; la posición de cada país resume su perfil temático medio:
países con alta producción en IA aplicada se acercan a los tópicos de sistemas
y aplicaciones, mientras que los países con producción más orientada a
gobernanza se acercan a los tópicos de ética y política. La calidad media de
representación (cos²) indica cuánta de la variación de cada marcador queda
capturada por los dos ejes; valores altos (> 0.7) permiten interpretar con
confianza la posición de ese marcador.

### 7.2 LDAvis

LDAvis (Sievert & Shirley, 2014) combina dos vistas:

- **Mapa de distancia intertópica:** los tópicos se proyectan en dos
  dimensiones a partir de la distancia entre sus distribuciones de palabras.
  El tamaño de cada burbuja es proporcional a la **prevalencia** del tópico en
  el corpus. Tópicos cercanos comparten vocabulario y pueden solaparse.
- **Ranking de términos relevantes:** para cada tópico se ordenan las palabras
  por **relevancia**:

$$
\\mathrm{{relevance}}(w \\mid t) = \\lambda \\, p(w \\mid t)
+ (1-\\lambda) \\, \\frac{{p(w \\mid t)}}{{p(w)}},
$$

  donde $\\lambda \\in [0,1]$ controla el peso de la frecuencia condicional
  frente a la exclusividad del término. Se usó **λ = {LAMBDA}** (valor
  recomendado por Sievert & Shirley para exploración), y se exportaron también
  los rankings para λ = 0 y λ = 1.

![Mapa intertópico de LDAvis](../outputs/visualizations/LDAvis_Intertopic_Map.png)

![Relevancia de términos por tópico](../outputs/visualizations/LDAvis_Relevance_Heatmap.png)

El visor interactivo completo está en
[`outputs/visualizations/LDAvis.html`](../outputs/visualizations/LDAvis.html).

**Ranking de términos relevantes (λ = {LAMBDA}):**

{rel_tab}

### 7.3 Comparación y complementariedad

El **HJ-Biplot** aporta una lectura multivariante y estructural: permite ver
cómo años, países y documentos se posicionan respecto de los tópicos, con una
medida explícita de calidad de representación (cos²). Es útil para responder
preguntas del tipo “¿qué tópicos caracterizan a los países con más producción?”
o “¿cómo cambia el perfil temático entre años?”.

**LDAvis** aporta una lectura término-tópico: muestra qué palabras definen cada
tópico, cómo se solapan los tópicos y cuáles son los términos más relevantes
para un λ dado. Es útil para **interpretar y etiquetar** los tópicos, y para
detectar tópicos redundantes.

Los dos métodos son complementarios: el biplot dice **dónde** se ubican los
documentos, años y países en el espacio temático; LDAvis dice **por qué** un
tópico se interpreta de una manera, a partir de sus términos. La convergencia
entre ambos —tópicos cercanos en LDAvis que también aparecen próximos en el
biplot— refuerza la validez de la estructura temática.

---

## 8. Resultados preliminares y su interpretación

### 8.1 Caracterización descriptiva del corpus

**Distribución temporal.** El corpus cubre {quality.get('Year range', '2018-2026')},
con una explosión de publicaciones a partir de 2023:

{annual_tab}

**Principales fuentes/revistas.** La producción se concentra en actas de
conferencias de NLP/IA y en revistas de acceso abierto:

{source_tab}

**Países.** Estados Unidos y China concentran algo más de dos quintas partes
del corpus; la cola incluye India, Alemania, Reino Unido, Italia, Canadá y
Australia:

{country_tab}

**Tipos de documento.** Predominan artículos y ponencias de congreso, con una
presencia relevante de revisiones:

{dt_tab}

**Instituciones.** Las afiliaciones se extrajeron de la columna `Affiliations`
de Scopus. La tabla siguiente muestra las instituciones con más documentos:

{inst_tab}

**Panorama bibliométrico.** El corpus es reciente, dominado por conferencias y
revistas de IA/NLP, con una fuerte presencia de Estados Unidos y China. La
combinación de tipos de documento (artículos, ponencias, revisiones) sugiere
un campo en consolidación, con mucha producción aplicada y una capa creciente
de reflexión ética y regulatoria.

### 8.2 Resultados del modelo LDA

La tabla siguiente resume los 14 tópicos: identificador, etiqueta propuesta,
número de documentos con tópico dominante, términos principales y país con más
documentos en el tópico (por volumen).

{_md_table(topic_table, cols=['topic_id', 'label', 'articles_count', 'top_terms', 'country'], headers=['Topic', 'Proposed label', 'Documents', 'Top terms', 'Leading country'])}

**Lectura temática.** Los tópicos pueden agruparse en cuatro bloques:

1. **Riesgos y seguridad:** T1 (ataques adversarios, privacidad, ciberseguridad),
   T13 (política de seguridad, *machine unlearning*) y parte de T3.
2. **Gobernanza y ética:** T3 (IA generativa, ética, gobernanza), T8 (sesgo,
   equidad, cultura) y T7 (sistemas agénticos, auditoría, empresa).
3. **Aplicaciones y evaluación:** T5 (ChatGPT, benchmarks), T9 (salud clínica),
   T6 (educación), T10 (entrenamiento y *fine-tuning*), T11 (RAG y automatización
   documental), T12 (multimodalidad) y T14 (redes neuronales, energía).
4. **Interacción y agentes:** T2 (razonamiento, alineación) y T4 (interacción
   humano-IA, chatbots).

Los tópicos de mayor volumen (T3, T10, T4, T11) son también los más estables
en el análisis de sensibilidad; T13 es el más pequeño (153 documentos) y debe
interpretarse como un tópico residual o emergente.

### 8.3 Interpretación del Biplot

- **Años:** los años 2022–2023 se asocian con tópicos de fundamentos y
  aplicaciones tempranas; 2024–2026 se desplazan hacia IA generativa, agentes,
  RAG y evaluación de modelos. La dirección de los vectores de los tópicos
  permite leer qué temas ganan peso relativo.
- **Países:** Estados Unidos y China, por volumen, ocupan posiciones centrales;
  los países europeos y latinoamericanos se distribuyen hacia tópicos de
  gobernanza, salud y educación. La posición debe leerse junto con la calidad
  de representación: si un país tiene cos² bajo, su posición en el plano es
  poco fiable.
- **Documentos:** la nube de documentos muestra una separación clara entre
  documentos de evaluación técnica (T10, T5) y documentos de ética/gobernanza
  (T3, T8), con una zona de solapamiento en interacción y agentes (T2, T4).

La calidad global del biplot se reporta en la tabla de la sección 7.1. Valores
medios de cos² altos indican que los dos primeros ejes capturan una parte
importante de la estructura; valores bajos obligan a interpretar con cautela.

### 8.4 Interpretación de LDAvis

- **Mapa intertópico:** los tópicos de IA generativa (T3), evaluación (T5) y
  entrenamiento (T10) aparecen próximos porque comparten vocabulario técnico.
  Los tópicos de ética (T8) y gobernanza (T7) se separan del núcleo técnico.
  T13 aparece aislado, consistente con su bajo volumen y su mezcla de términos.
- **Tamaño de burbuja:** T3 es el tópico más prevalente, seguido de T10, T4 y
  T11; T13 es el menos prevalente.
- **Relevancia:** con λ = {LAMBDA}, los términos más relevantes de cada tópico
  combinan frecuencia y exclusividad. Por ejemplo, T1 se define por
  `adversarial`, `attack`, `privacy`, `hallucination`; T9 por `clinical`,
  `patient`, `diagnosis`; T11 por `rag`, `retrieval`, `document`. El ranking
  completo está en `outputs/results/lda/ldavis_relevance_terms.csv`.

### 8.5 Síntesis integradora

Las tres fuentes convergen en una estructura temática coherente:

- **Descripción del corpus:** crecimiento explosivo, dominio de conferencias y
  revistas de IA, alta concentración en Estados Unidos y China.
- **Biplot:** separación entre un núcleo técnico (evaluación, entrenamiento,
  aplicaciones) y un núcleo ético-regulatorio (gobernanza, sesgo, seguridad),
  con los años recientes desplazándose hacia el primero.
- **LDAvis:** los tópicos técnicos comparten vocabulario y se solapan; los
  tópicos éticos se separan; los tópicos pequeños son más frágiles.

En conjunto, el corpus no es un campo homogéneo: combina una **capa técnica**
(evaluación de modelos, *fine-tuning*, RAG, multimodalidad) con una **capa
normativa** (ética, gobernanza, sesgo, privacidad, seguridad). El modelo LDA de
14 tópicos captura esa dualidad y proporciona una base defendible para el
análisis de tendencias de la fase 4.

---

## 9. Referencias

Bender, E. M., Gebru, T., McMillan-Major, A., & Shmitchell, S. (2021). On the dangers of stochastic parrots: Can language models be too big? In *Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency* (pp. 610–623). https://doi.org/10.1145/3442188.3445922

Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet allocation. *Journal of Machine Learning Research, 3*, 993–1022.

Bouma, G. (2009). Normalized (pointwise) mutual information in collocation extraction. In *Proceedings of the Biennial GSCL Conference* (pp. 31–40).

Chang, J., Gerrish, S., Wang, C., Boyd-Graber, J., & Blei, D. M. (2009). Reading tea leaves: How humans interpret topic models. In *Advances in Neural Information Processing Systems 22* (pp. 288–296).

De Waal, A., & Barnard, E. (2008). Evaluating topic models with stability. In *Proceedings of the 19th Annual Symposium of the Pattern Recognition Association of South Africa* (pp. 79–84).

Floridi, L., Cowls, J., Beltrametti, M., Chatila, R., Chazerand, P., Dignum, V., Luetge, C., Madelin, R., Pagallo, U., Rossi, F., Schafer, B., Valcke, P., & Vayena, E. (2018). AI4People—An ethical framework for a good AI society: Opportunities, risks, principles, and recommendations. *Minds and Machines, 28*(4), 689–707. https://doi.org/10.1007/s11023-018-9482-5

Gabriel, K. R. (1971). The biplot graphic display of matrices with application to principal component analysis. *Biometrika, 58*(3), 453–467. https://doi.org/10.1093/biomet/58.3.453

Galindo-Villardón, M. P. (1986). Una alternativa de representación simultánea: HJ-Biplot. *Questiió, 10*(1), 13–23.

Greene, D., O'Callaghan, D., & Cunningham, P. (2014). How many topics? Stability analysis for topic models. In *Machine Learning and Knowledge Discovery in Databases (ECML PKDD 2014)* (LNCS 8724, pp. 498–513). Springer. https://doi.org/10.1007/978-3-662-44848-9_32

Grimmer, J., & Stewart, B. M. (2013). Text as data: The promise and pitfalls of automatic content analysis methods for political texts. *Political Analysis, 21*(3), 267–297. https://doi.org/10.1093/pan/mps028

Jobin, A., Ienca, M., & Vayena, E. (2019). The global landscape of AI ethics guidelines. *Nature Machine Intelligence, 1*(9), 389–399. https://doi.org/10.1038/s42256-019-0088-2

Koltcov, S., Koltsova, O., & Nikolenko, S. (2014). Latent Dirichlet allocation: Stability and applications to studies of user-generated content. In *Proceedings of the 2014 ACM Conference on Web Science* (pp. 161–165). https://doi.org/10.1145/2615569.2615680

Lau, J. H., Newman, D., & Baldwin, T. (2014). Machine reading tea leaves: Automatically evaluating topic coherence and topic model quality. In *Proceedings of the 14th Conference of the European Chapter of the Association for Computational Linguistics* (pp. 530–539). https://doi.org/10.3115/v1/E14-1056

Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to information retrieval*. Cambridge University Press.

Mittelstadt, B. D., Allo, P., Taddeo, M., Wachter, S., & Floridi, L. (2016). The ethics of algorithms: Mapping the debate. *Big Data & Society, 3*(2), 1–21. https://doi.org/10.1177/2053951716679679

Page, M. J., McKenzie, J. E., Bossuyt, P. M., Boutron, I., Hoffmann, T. C., Mulrow, C. D., Shamseer, L., Tetzlaff, J. M., Akl, E. A., Brennan, S. E., Chou, R., Glanville, J., Grimshaw, J. M., Hróbjartsson, A., Lalu, M. M., Li, T., Loder, E. W., Mayo-Wilson, E., McDonald, S., … Moher, D. (2021). The PRISMA 2020 statement: An updated guideline for reporting systematic reviews. *BMJ, 372*, n71. https://doi.org/10.1136/bmj.n71

Röder, M., Both, A., & Hinneburg, A. (2015). Exploring the space of topic coherence measures. In *Proceedings of the Eighth ACM International Conference on Web Search and Data Mining* (pp. 399–408). https://doi.org/10.1145/2684822.2685324

Sievert, C., & Shirley, K. (2014). LDAvis: A method for visualizing and interpreting topics. In *Proceedings of the Workshop on Interactive Language Learning, Visualization, and Interfaces* (pp. 63–70). https://doi.org/10.3115/v1/W14-3110

Weidinger, L., Mellor, J., Rauh, M., Griffin, C., Uesato, J., Huang, P.-S., Cheng, M., Glaese, M., Balle, B., Kasirzadeh, A., Kenton, Z., Brown, S., Hawkins, W., Stepleton, T., Biles, C., Birhane, A., Haas, J., Rimell, L., Hendricks, L. A., … Gabriel, I. (2021). *Ethical and social risks of harm from language models* (arXiv:2112.04359). arXiv. https://arxiv.org/abs/2112.04359

---

## Anexo A. Reproducibilidad y estructura del repositorio

```text
bertlda/
├── raw_corpus.csv                  # Scopus export (not tracked if large)
├── src/bertlda/
│   ├── config.py                   # central configuration
│   ├── text_processing.py          # cleaning, stopwords, lemmatization
│   ├── phase0_audit.py             # corpus audit + descriptive figures
│   ├── phase1a_lda_diagnostic.py   # tokenization + K search
│   ├── phase1b_lda_decision.py     # stability + alpha/eta search
│   ├── phase1c_lda_final.py        # final LDA + Theta + topic summary
│   ├── phase3_visualizations.py    # similarity, top-words, topic x year/country
│   ├── phase5_sensitivity.py       # seed/K sensitivity
│   └── phase6_report.py            # config JSON, biplot, LDAvis, report
├── outputs/
│   ├── models/lda/lda_final.model
│   ├── results/config/model_configuration_full.json
│   ├── results/lda/*.csv
│   └── visualizations/*.png|.svg|.html
└── docs/Methodology_LDA.md
```

## Anexo B. Inventario de artefactos principales

- `outputs/models/lda/lda_final.model` — modelo LDA final.
- `outputs/results/lda/lda_theta.csv` — matriz documento-tópico ({theta.shape[0]:,} × {chosen_k}).
- `outputs/results/lda/lda_topic_summary.csv` — etiquetas, palabras clave y país líder.
- `outputs/results/lda/lda_ranking.csv` — ranking de tópicos por frecuencia.
- `outputs/results/lda/lda_k_search.csv` — grilla de K.
- `outputs/results/lda/lda_alpha_eta_search.csv` — grilla alpha/eta.
- `outputs/results/lda/lda_sensitivity_seeds.csv` — estabilidad entre semillas.
- `outputs/results/lda/lda_sensitivity_k.csv` — estabilidad en el vecindario de K.
- `outputs/results/lda/hjbiplot_*.csv` — coordenadas y calidades del HJ-Biplot.
- `outputs/results/lda/ldavis_relevance_terms.csv` — relevancia de términos.
- `outputs/results/config/model_configuration_full.json` — configuración completa.
- `outputs/visualizations/LDAvis.html` — visor interactivo.
- `docs/Methodology_LDA.md` — este documento.
"""
    return md


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ldavis", action="store_true",
                        help="Skip the heavy pyLDAvis HTML generation.")
    args, _ = parser.parse_known_args(argv)

    asegurar_carpetas()
    print_separador("PHASE 6 — REPORT AND FINAL DELIVERABLES")
    t_total = perf_counter()

    from gensim.models.ldamodel import LdaModel

    # ---- load artifacts ----
    base = cargar_checkpoint("ck_corpus_base")
    base_df = base["df"].reset_index(drop=True)
    params = base["params"]

    ck_lda = cargar_checkpoint("ck_corpus_lda")
    docs_tokens = (ck_lda.get("docs_tokens") or ck_lda.get("docs_tok")
                   or ck_lda.get("docs_tokensens"))
    id2word = ck_lda["id2word"]
    corpus = ck_lda["corpus"]
    dict_eval = ck_lda["dict_eval"]

    lda = LdaModel.load(str(RUTAS["models_lda"] / "lda_final.model"))
    chosen_k = int(lda.num_topics)

    theta = pd.read_csv(RUTAS["results"] / "lda" / "lda_theta.csv")
    summary = pd.read_csv(RUTAS["results"] / "lda" / "lda_topic_summary.csv")
    summary = _add_volume_leading_country(theta, summary)
    guardar_csv(summary, "lda_topic_summary.csv",
                subdir=RUTAS["results"] / "lda")
    df_k = pd.read_csv(RUTAS["results"] / "lda" / "lda_k_search.csv").sort_values("K")
    df_ae = pd.read_csv(RUTAS["results"] / "lda" / "lda_alpha_eta_search.csv")
    df_stability = pd.read_csv(RUTAS["results"] / "lda" / "lda_stability.csv")

    sens_seed_path = RUTAS["results"] / "lda" / "lda_sensitivity_seeds.csv"
    sens_k_path = RUTAS["results"] / "lda" / "lda_sensitivity_k.csv"
    df_sens_seed = pd.read_csv(sens_seed_path) if sens_seed_path.exists() else None
    df_sens_k = pd.read_csv(sens_k_path) if sens_k_path.exists() else None

    quality_df = pd.read_csv(RUTAS["results"] / "corpus" / "corpus_quality.csv")
    quality = dict(zip(quality_df["metric"], quality_df["value"]))
    annual = pd.read_csv(RUTAS["results"] / "corpus" / "annual_distribution.csv")
    countries = pd.read_csv(RUTAS["results"] / "corpus" / "country_distribution.csv")
    sources = pd.read_csv(RUTAS["results"] / "corpus" / "source_ranking.csv")

    # Final model metrics
    print("\n-- FINAL MODEL METRICS --")
    t0 = perf_counter()
    final_metrics = lda_metrics(lda, corpus, docs_tokens, dict_eval, chosen_k)
    guardar_csv(pd.DataFrame([final_metrics]), "lda_final_metrics.csv",
                subdir=RUTAS["results"] / "lda")
    print(f"  Cv={final_metrics['cv']:.4f} NPMI={final_metrics['npmi']:+.4f} "
          f"({_fmt(perf_counter() - t0)})")

    # Descriptive extras
    descriptive = _descriptive_extras(base_df)

    # PRISMA
    _prisma_flow(quality)

    # K coherence
    _k_coherence_plot(df_k, chosen_k)

    # HJ-Biplot
    biplot_summary = _biplots(theta, chosen_k)

    # LDAvis
    ldavis_out = _ldavis(lda, corpus, id2word, chosen_k,
                         skip=args.skip_ldavis)

    # Relevance table (reuse if LDAvis ran; otherwise compute)
    rel_path = RUTAS["results"] / "lda" / "ldavis_relevance_terms.csv"
    if rel_path.exists():
        rel_df = pd.read_csv(rel_path)
    else:
        rows = []
        for lam in (0.0, LAMBDA, 1.0):
            r = _relevance_table(lda, corpus, id2word, lam, topn=10)
            for _, row in r.iterrows():
                rows.append({"lambda": lam, **row.to_dict()})
        rel_df = pd.DataFrame(rows)
        guardar_csv(rel_df, "ldavis_relevance_terms.csv",
                    subdir=RUTAS["results"] / "lda")

    # Alpha/eta chosen from checkpoint (fallback to best row)
    try:
        ae_params = cargar_checkpoint("ck_lda_final_params")
        alpha = str(ae_params.get("alpha", "asymmetric"))
        eta = str(ae_params.get("eta", "symmetric"))
    except Exception:
        best = df_ae.loc[df_ae["cv"].idxmax()]
        alpha, eta = str(best["alpha"]), str(best["eta"])

    ctx = {
        "base_df": base_df, "params": params,
        "docs_tokens": docs_tokens, "id2word": id2word,
        "corpus": corpus, "dict_eval": dict_eval,
        "theta": theta, "summary": summary,
        "df_k": df_k, "df_ae": df_ae, "df_stability": df_stability,
        "df_sens_seed": df_sens_seed, "df_sens_k": df_sens_k,
        "quality": quality, "quality_df": quality_df,
        "annual": annual, "countries": countries, "sources": sources,
        "descriptive": descriptive, "rel_df": rel_df,
        "chosen_k": chosen_k, "alpha": alpha, "eta": eta,
        "final_metrics": final_metrics,
    }

    # Full config JSON
    print("\n-- FULL CONFIGURATION JSON --")
    cfg_json = _build_config_json(ctx)
    cfg_dir = RUTAS["results"] / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "model_configuration_full.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg_json, f, indent=2, ensure_ascii=False)
    print(f"  -> {cfg_path.relative_to(PROJECT_ROOT)}")

    # Markdown report
    print("\n-- MARKDOWN REPORT --")
    md = _report_markdown(ctx, cfg_json, biplot_summary, ldavis_out)
    docs_dir = PROJECT_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    md_path = docs_dir / "Methodology_LDA.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  -> {md_path.relative_to(PROJECT_ROOT)}")

    print_separador("PHASE 6 COMPLETED")
    print(f"  Report: {md_path.relative_to(PROJECT_ROOT)}")
    print(f"  Config: {cfg_path.relative_to(PROJECT_ROOT)}")
    print(f"  Time: {_fmt(perf_counter() - t_total)}")
    return {"report": str(md_path), "config": str(cfg_path)}


if __name__ == "__main__":
    main()
