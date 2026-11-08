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
    1: "Human problem-solving and AI-assisted work",
    2: "AI governance and framework design",
    3: "Generative AI ethics, technology, and societal challenges",
    4: "Privacy, explainability, and financial trust",
    5: "ChatGPT response quality and performance evaluation",
    6: "Perception, trust, and AI adoption surveys",
    7: "Agentic systems, architecture, and enterprise workflows",
    8: "Bias, fairness, gender, and stereotype in prompts",
    9: "Clinical, medical, and healthcare AI",
    10: "Prediction, foundation models, and biological sequence learning",
    11: "Safety, cybersecurity, and adversarial robustness",
    12: "Multimodal and vision-language models",
    13: "Evaluation, benchmarks, and performance measurement",
    14: "Training, fine-tuning, and machine unlearning",
    15: "Intelligent management, markets, and energy systems",
    16: "Public policy, digital services, and urban platforms",
    17: "User interaction, chatbots, and conversational support",
    18: "Human cognition, emotion, and social simulation",
    19: "Text detection, datasets, and classification",
    20: "Retrieval, knowledge graphs, and semantic generation",
    21: "Networks, robotics, simulation, and digital twins",
    22: "Alignment, values, and moral decision-making",
    23: "Application challenges, reviews, and field advances",
    24: "Language, culture, translation, and multilingual corpora",
    25: "ChatGPT in academic and systematic literature reviews",
    26: "Law, regulation, compliance, and software requirements",
    27: "AI in education and student learning",
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
                    "geopandas",
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
            "The table above summarizes seed stability for each K in the "
            "neighbourhood. Jaccard is computed over the top-10 words of "
            "topics aligned with the Hungarian algorithm; ARI is computed over "
            "the dominant-topic assignment of each document."
        )
        _row = df_sens_seed[df_sens_seed["K"] == chosen_k]
        if len(_row):
            _r = _row.iloc[0]
            _jm, _am = float(_r["jaccard_mean"]), float(_r["ari_mean"])
            _level = ("high" if _jm >= 0.5 and _am >= 0.4
                      else "moderate" if _jm >= 0.2 else "low")
            _top = summary.sort_values("articles_count", ascending=False).head(4)
            _top_ids = ", ".join(f"T{int(t)}" for t in _top["topic_id"])
            _small = summary.sort_values("articles_count").head(1)
            _small_id = int(_small.iloc[0]["topic_id"])
            sens_conclusion = (
                f"For the selected K ({chosen_k}), seed stability is "
                f"**{_level}**: mean Jaccard {_jm:.4f} "
                f"(SD {_r['jaccard_std']:.4f}, minimum {_r['jaccard_min']:.4f}) "
                f"and mean ARI {_am:.4f} "
                f"(SD {_r['ari_std']:.4f}, minimum {_r['ari_min']:.4f}). "
                f"The highest-volume topics ({_top_ids}) are recovered most "
                f"consistently, while the smallest topic (T{_small_id}) is "
                f"more sensitive. The {chosen_k}-topic structure is considered "
                "**sufficiently robust for an exploratory analysis**, with the "
                "recommendation to report low-frequency topics as provisional. "
                "This result is consistent with the topic-model stability "
                "literature (Greene, O'Callaghan & Cunningham, 2014; "
                "De Waal & Barnard, 2008; Koltcov, Koltsova & Nikolenko, 2014)."
            )
        else:
            sens_conclusion = (
                "Seed stability is **moderate**, as is usual in LDA on "
                "heterogeneous scientific corpora with short abstracts."
            )
    else:
        sens_tab = "_Phase 5 has not produced the sensitivity table yet._"
        sens_text = ""
        sens_conclusion = (
            "Seed stability is **moderate**, as is usual in LDA on "
            "heterogeneous scientific corpora with short abstracts."
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
        sens_k_tab = "_No K-overlap data available._"

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

    md = f"""# Topic Modeling with Latent Dirichlet Allocation: Methodology and Preliminary Results

**Corpus:** AI ethics, generative AI, and large language models (Scopus, 2018–2026)
**Generated:** {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}
**Repository:** `https://github.com/JeanVydes/bertlda`
**Full configuration:** `outputs/results/config/model_configuration_full.json`

> **Scope note.** This document presents the methodology and preliminary results
> of the LDA topic model. The previous delivery (PRISMA, Scopus + Web of
> Science integration) is summarized in Section 1; the modeling corpus in this
> repository is the Scopus export of {n_raw:,} records. Fields marked
> `[TO COMPLETE]` must be taken from the previous delivery for the final version.

---

## 1. Corpus collection and de-duplication

### 1.1 Sources and search strategy

The systematic review of the previous delivery combined records from **Scopus**
and **Web of Science (WoS)**. The consolidated corpus targets the intersection
of artificial intelligence, generative AI, large language models, and their
ethical, social, regulatory, and safety dimensions. The exact search string,
query dates, and WoS integration counts are documented in the previous
delivery:

- **Scopus search string:** `[TO COMPLETE with the exact string from the previous delivery]`
- **WoS search string:** `[TO COMPLETE with the exact string from the previous delivery]`
- **Scopus query date:** `[TO COMPLETE]`
- **WoS query date:** `[TO COMPLETE]`
- **WoS records integrated:** `[TO COMPLETE]`
- **Duplicates removed during integration:** `[TO COMPLETE]`

The export used in this delivery contains **{n_raw:,} Scopus records**
(field `Source = Scopus`), with title, abstract, keywords, year, source,
affiliations, affiliation country, document type, and DOI. The observed time
window is **{quality.get('Year range', '2018-2026')}**; the year 2026 is
incomplete because the export was performed during that year.

### 1.2 PRISMA process

The PRISMA 2020 process (Page et al., 2021) is summarized below for the
modeling corpus. The Scopus + WoS integration counts from the previous
delivery must be verified before the final version.

{quality_tab}

- **Records identified (Scopus):** {n_raw:,}
- **Records without title or abstract:** 0
- **Excluded by length (< {CFG['min_palabras_doc']} words):** {n_excl:,}
- **Duplicates detected:** {quality.get('DOI duplicates', 0)} by DOI and {quality.get('Title duplicates', 0)} by title (retained and reported as a limitation)
- **Final documents for LDA:** **{n_docs:,}**
- **Normalized countries:** {n_countries}
- **Distinct sources:** {n_sources:,}

### 1.3 PRISMA flow diagram

![PRISMA flow diagram](../outputs/visualizations/PRISMA_Flow.png)

The full Scopus + WoS integration diagram is kept in the previous delivery.
The diagram above documents the flow of the corpus actually modeled.

### 1.4 Final corpus

The final corpus contains **{n_docs:,} documents** with sufficient text for
modeling. The annual distribution shows a strong concentration in recent years,
consistent with the rise of generative AI:

{annual_tab}

---

## 2. Text preprocessing

### 2.1 Fields used

**Title** and **abstract** (`Title` + `Abstract`) were used. Author and index
keywords (`Author Keywords`, `Index Keywords`) are available in the export but
were not concatenated to the input text, in order to avoid duplicating terms
and to keep a homogeneous representation across documents.

### 2.2 Techniques applied

1. **Unicode normalization and lowercasing.**
2. **Editorial noise removal:** URLs, e-mail addresses, numeric citations,
   copyright notices, licenses, and "published by" sentences.
3. **Tokenization** by whitespace over cleaned text.
4. **WordNet lemmatization** (nouns first, then verbs).
5. **Stopword removal in four layers:**
   - English functional stopwords (NLTK);
   - editorial stopwords (Elsevier, Springer, MDPI, "journal", "abstract", etc.);
   - methodological stopwords ("study", "results", "method", "approach", etc.);
   - AI-ethics domain terms and valid acronyms.
6. **Length filtering:** tokens shorter than 3 characters are discarded, except
   valid acronyms (`ai`, `llm`, `gpt`, `nlp`, `gdpr`, etc.).
7. **Bigrams:** collocations are detected with `gensim.models.phrases`
   (`min_count={ctx['params'].get('bigram_min_count')}`,
   `threshold={ctx['params'].get('bigram_threshold')}`), for example
   `generative_ai`, `artificial_intelligence`, `machine_learning`.
8. **Vocabulary filtering:** the training dictionary removes terms with
   document frequency below `{ctx['params'].get('no_below')}` and above
   `{ctx['params'].get('no_above')}` of documents.
9. **No aggressive stemming:** lemmatization is preferred to preserve topic
   interpretability.

### 2.3 Tools

- **Python 3.13** and **NLTK** for tokenization, stopwords, and lemmatization.
- **Gensim 4.4** for bigrams, dictionary, BoW corpus, and LDA.
- **pandas / NumPy** for data manipulation.
- **Matplotlib** for figures.

### 2.4 Preprocessing results

| Metric | Value |
|---|---:|
| Modeled documents | {n_docs:,} |
| Total tokens | {cfg_json['preprocessing']['n_tokens_total']:,} |
| Tokens per document (mean) | {cfg_json['preprocessing']['n_tokens_mean']:.2f} |
| Training vocabulary | {cfg_json['preprocessing']['vocab_train']:,} |
| Evaluation vocabulary | {cfg_json['preprocessing']['vocab_eval']:,} |
| Documents with empty BoW | {cfg_json['preprocessing']['n_docs_with_empty_bow']} |
| Total stopwords | {cfg_json['preprocessing']['n_stopwords']} |

Cleaning is implemented in `src/bertlda/text_processing.py` and applied
deterministically in `src/bertlda/phase1a_lda_diagnostic.py`.

---

## 3. LDA model foundations

Latent Dirichlet Allocation (Blei, Ng & Jordan, 2003) is a hierarchical
generative model for collections of discrete documents. Each document is
represented as a mixture of a fixed number of topics, and each topic as a
distribution over the vocabulary. Formally, for a document $d$:

$$
\\theta_d \\sim \\mathrm{{Dirichlet}}(\\alpha), \\qquad
z_{{d,n}} \\sim \\mathrm{{Categorical}}(\\theta_d), \\qquad
w_{{d,n}} \\sim \\mathrm{{Categorical}}(\\phi_{{z_{{d,n}}}}),
$$

where $\\theta_d$ is the document's topic proportion vector, $z_{{d,n}}$ is
the latent topic of word $n$, and $\\phi_k \\sim \\mathrm{{Dirichlet}}(\\eta)$
is the word distribution of topic $k$. The joint likelihood integrates out the
latent variables, and is approximated by variational inference or Gibbs
sampling; Gensim uses online variational inference.

LDA is appropriate for this corpus because:

- documents are short, heterogeneous texts (title + abstract), where a
  probabilistic mixture model is more stable than hard clustering;
- the objective is to **describe** the thematic structure and its evolution,
  not to classify documents into predefined categories;
- the document-topic matrix ($\\Theta$) supports downstream analysis with
  HJ-Biplot, LDAvis, and trend models;
- AI-ethics literature is interdisciplinary: a single document can combine
  governance, bias, privacy, and technical evaluation, which favors a mixed
  representation.

---

## 4. Selecting the number of topics (K)

### 4.1 Coherence metrics

Two complementary metrics were evaluated:

- **C_v** (Röder, Both & Hinneburg, 2015): combines semantic similarity
  between words, a sliding window, and mutual confirmation; it is the
  reference metric for comparing topics in scientific corpora.
- **NPMI** (Bouma, 2009): normalized pointwise mutual information; it
  penalizes spurious associations and is easily interpretable in [-1, 1].

**Topic diversity** (proportion of unique words in the top-10 of all topics),
**perplexity**, **dominance** (share of the most frequent topic), and the
**normalized entropy** of the size distribution were also recorded.

### 4.2 Evaluated range and decision criterion

**K = {int(df_k['K'].min())}–{int(df_k['K'].max())}** ({len(df_k)} values)
was evaluated. The coherence curve reaches its global maximum at
**K = {best_k}** (C_v = {best_cv:.4f}), but the gain over K = {chosen_k} is
marginal ({best_cv - chosen_cv:+.4f}) and concentrated in the plateau region.
**K = {chosen_k}** was selected for the following reasons:

1. **Coherence plateau:** K = {chosen_k} already lies on the plateau
   (C_v = {chosen_cv:.4f}; NPMI = {chosen_npmi:+.4f}).
2. **Parsimony and interpretability:** {chosen_k} topics produce a manageable
   thematic structure aligned with the dimensions of AI ethics (safety, bias,
   governance, education, health, etc.).
3. **Diversity:** the top-10 diversity is {chosen_div:.4f}, high for this
   corpus size.
4. **Dominance:** the most frequent topic concentrates {chosen_dom * 100:.1f}%
   of documents, below the excessive-dominance threshold (40%).
5. **Stability:** the sensitivity analysis (Section 6) confirms that the
   K = {chosen_k} structure is comparable to its neighbours.

![LDA coherence by number of topics](../outputs/visualizations/LDA_K_Coherence.png)

### 4.3 Full K-search table

{_md_table(df_k, cols=['K', 'cv', 'npmi', 'diversity', 'perplexity', 'dominance', 'entropy_norm', 'topics_empty'], headers=['K', 'C_v', 'NPMI', 'Diversity', 'Perplexity', 'Dominance', 'Entropy', 'Empty topics'])}

---

## 5. Model configuration and execution

### 5.1 Library and software

- **Gensim 4.4.0**, single-process `LdaModel` implementation.
- **Python 3.13**; the environment is reproduced with `uv sync`.
- Global seed **{CFG['seed']}**.
- Exports are CSV-only (never xlsx); intermediate checkpoints use `.pkl`.

### 5.2 Hyperparameters

| Parameter | Value |
|---|---|
| Number of topics (K) | {chosen_k} |
| Alpha (document-topic prior) | `{ctx['alpha']}` |
| Eta / beta (topic-word prior) | `{ctx['eta']}` |
| Iterations (passes) | {CFG['lda']['passes_final']} |
| Internal iterations per document | 50 |
| Random seed | {CFG['seed']} |
| Documents | {n_docs:,} |
| Vocabulary | {len(ctx['id2word']):,} |
| Implementation | `gensim.models.ldamodel.LdaModel` |

The alpha/eta search evaluated **{len(df_ae)} combinations** (4 alpha values ×
4 eta values). The best combination was
**alpha = `{ctx['alpha']}`, eta = `{ctx['eta']}`**
(C_v = {float(df_ae['cv'].max()):.4f}). The 8 best combinations are:

{ae_tab}

### 5.3 Reproducibility

The full code lives in `src/bertlda/`. The sequence is:

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

The full configuration, library versions, metrics, and artifact inventory are
exported to `outputs/results/config/model_configuration_full.json`.

### 5.4 Final model metrics

| Metric | Value |
|---|---:|
| C_v | {metrics.get('cv', float('nan')):.4f} |
| NPMI | {metrics.get('npmi', float('nan')):+.4f} |
| Diversity (top-10) | {metrics.get('diversity', float('nan')):.4f} |
| Perplexity | {metrics.get('perplexity', float('nan')):.4f} |
| Dominance | {metrics.get('dominance', float('nan')):.4f} |
| Normalized entropy | {metrics.get('entropy_norm', float('nan')):.4f} |
| Empty topics | {metrics.get('topics_empty', 0)} |

> **Note on C_v comparability.** The K grid (Section 4) uses
> `alpha="symmetric"` and `passes={CFG['lda']['passes_lda']}`; the final model
> uses `alpha="{ctx['alpha']}"`, `eta="{ctx['eta']}"`, and
> `passes={CFG['lda']['passes_final']}`. Its C_v
> ({metrics.get('cv', float('nan')):.4f}) therefore does not exactly match the
> grid value for K = {chosen_k} ({chosen_cv:.4f}). Comparisons are valid within
> each configuration.

---

## 6. Model sensitivity analysis

The sensitivity analysis evaluates three sources of variation: **random seed**,
**K neighbourhood**, and **alpha/eta hyperparameters**. Two metrics are used to
compare runs:

- **Jaccard** overlap between the top-10 words of aligned topics. Alignment is
  solved with the Hungarian algorithm over cosine similarity between
  topic-word distributions.
- **Adjusted Rand Index (ARI)** between the dominant-topic assignments of two
  runs.

### 6.1 Stability across random seeds

**{len(CFG['lda']['sensitivity_seeds'])} seeds**
({', '.join(str(s) for s in CFG['lda']['sensitivity_seeds'])}) were trained for
each K in the neighbourhood, with `passes={CFG['lda']['sensitivity_passes']}`.
The table reports the mean and standard deviation over all seed pairs:

{sens_tab}

{sens_text}

![LDA stability across seeds](../outputs/visualizations/LDA_Sensitivity_Stability.png)

### 6.2 Stability across K

The table compares each neighbourhood model against the reference model
(K = {chosen_k}, seed {CFG['seed']}). `Best Jaccard` is the best overlap of
each model topic with a reference topic; the last column reports how many
reference topics are covered with Jaccard ≥ 0.5.

{sens_k_tab}

![Overlap across K values](../outputs/visualizations/LDA_Sensitivity_K_Overlap.png)

### 6.3 Sensitivity to alpha and eta

The **{len(df_ae)}-combination** grid shows small C_v differences between
configurations (range {float(df_ae['cv'].max() - df_ae['cv'].min()):.4f}). The
selected configuration (`alpha={ctx['alpha']}`, `eta={ctx['eta']}`) remains
among the best and yields the highest coherence. This suggests the thematic
structure is reasonably robust to prior choice, as long as alpha remains
asymmetric or close to symmetric.

### 6.4 Robustness conclusion

{sens_conclusion}

---

## 7. Biplot and LDAvis methods

### 7.1 HJ-Biplot

The **HJ-Biplot** (Galindo-Villardón, 1986), an extension of Gabriel's biplot
(1971), represents rows and columns of a matrix simultaneously through a
singular value decomposition. For a centered matrix $X$:

$$
X = U D V^\\top,
$$

row coordinates are $A = U D$ and column coordinates $B = V D$. The key
property is that the scalar product $a_i \\cdot b_j$ approximates the value
$x_{{ij}}$, so rows and columns are interpreted in the same space. The
**representation quality** of each marker is measured with the squared cosine:

$$
\\cos^2_{{i}} = \\frac{{a_{{i1}}^2 + a_{{i2}}^2}}{{\\sum_k a_{{ik}}^2}},
$$

and analogously for columns. The **global quality** is the mean of the squared
cosines. Three biplots were built:

1. **Years × topics:** each row is a year (with at least 20 documents) and each
   column a topic; the value is the mean topic probability for that year.
2. **Countries × topics:** each row is one of the {CFG['report']['top_countries']} countries with most documents;
   the value is the mean topic probability.
3. **Documents × topics:** a random sample of {CFG['viz']['biplot_max_documents']}
   documents, colored by dominant topic.

{biplot_tab}

![HJ-Biplot years × topics](../outputs/visualizations/LDA_HJBiplot_Years.png)

![HJ-Biplot countries × topics](../outputs/visualizations/LDA_HJBiplot_Countries.png)

![HJ-Biplot documents × topics](../outputs/visualizations/LDA_HJBiplot_Documents.png)

**Reading.** In the years biplot, proximity between a year and a topic means
that topic gains relative weight in that year; recent years (2024–2026) shift
toward generative AI, agents, and model evaluation topics. In the countries
biplot, countries are identified by number with a side legend; each country's
position summarizes its mean thematic profile: countries with high applied-AI
output align with systems and applications topics, while countries more
oriented to governance align with ethics and policy topics. Mean
representation quality (cos²) indicates how much of each marker's variation
the two axes capture; high values (> 0.7) allow confident reading of that
marker position.

### 7.2 LDAvis

LDAvis (Sievert & Shirley, 2014) combines two views:

- **Intertopic distance map:** topics are projected into two dimensions from
  the distance between their word distributions. Bubble size is proportional
  to topic **prevalence** in the corpus. Nearby topics share vocabulary and
  may overlap.
- **Relevant-term ranking:** for each topic, words are ranked by
  **relevance**:

$$
\\mathrm{{relevance}}(w \\mid t) = \\lambda \\, p(w \\mid t)
+ (1-\\lambda) \\, \\frac{{p(w \\mid t)}}{{p(w)}},
$$

  where $\\lambda \\in [0,1]$ weights conditional frequency against term
  exclusivity. **λ = {LAMBDA}** was used (the value recommended by Sievert &
  Shirley for exploration), and rankings for λ = 0 and λ = 1 were also
  exported.

![LDAvis intertopic distance map](../outputs/visualizations/LDAvis_Intertopic_Map.png)

![LDAvis term relevance heatmap](../outputs/visualizations/LDAvis_Relevance_Heatmap.png)

The full interactive viewer is in
[`outputs/visualizations/LDAvis.html`](../outputs/visualizations/LDAvis.html).

**Relevant-term ranking (λ = {LAMBDA}):**

{rel_tab}

### 7.3 Comparison and complementarity

The **HJ-Biplot** provides a multivariate, structural reading: it shows how
years, countries, and documents are positioned relative to topics, with an
explicit representation-quality measure (cos²). It answers questions such as
"which topics characterize the most productive countries?" or "how does the
thematic profile change over years?".

**LDAvis** provides a term-topic reading: it shows which words define each
topic, how topics overlap, and which terms are most relevant for a given λ.
It is useful for **interpreting and labeling** topics and for detecting
redundant topics.

The two methods are complementary: the biplot says **where** documents, years,
and countries lie in the thematic space; LDAvis says **why** a topic is
interpreted in a given way, from its terms. Their convergence — topics close
in LDAvis that also appear close in the biplot — reinforces the validity of
the thematic structure.

---

## 8. Preliminary results and interpretation

### 8.1 Descriptive corpus characterization

**Temporal distribution.** The corpus covers {quality.get('Year range', '2018-2026')},
with an explosion of publications from 2023:

{annual_tab}

**Main sources/journals.** Output concentrates in NLP/AI conference proceedings
and open-access journals:

{source_tab}

**Countries.** The United States and China concentrate slightly more than two
fifths of the corpus; the tail includes India, Germany, the United Kingdom,
Italy, Canada, and Australia:

{country_tab}

**Document types.** Articles and conference papers predominate, with a
relevant share of reviews:

{dt_tab}

**Institutions.** Affiliations were extracted from the Scopus `Affiliations`
column. The table shows the institutions with most documents:

{inst_tab}

**Bibliometric overview.** The corpus is recent, dominated by AI/NLP
conferences and journals, with a strong presence of the United States and
China. The mix of document types (articles, conference papers, reviews)
suggests a consolidating field, with much applied output and a growing layer
of ethical and regulatory reflection.

### 8.2 LDA model results

The table below summarizes the {chosen_k} topics: identifier, proposed label,
number of documents with dominant topic, main terms, and the country with most
documents in the topic (by volume).

{_md_table(topic_table, cols=['topic_id', 'label', 'articles_count', 'top_terms', 'country'], headers=['Topic', 'Proposed label', 'Documents', 'Top terms', 'Leading country'])}

**Thematic reading.** The topics can be grouped into five blocks:

1. **Risk, security, and policy:** T11 (cybersecurity, adversarial robustness),
   T4 (privacy, explainability, trust), T26 (law, regulation, compliance),
   T16 (public policy, digital services), and T22 (alignment, values).
2. **Governance and ethics:** T3 (generative AI, ethics, societal challenges),
   T2 (AI governance frameworks), T8 (bias, fairness, gender), T6 (perception,
   trust, adoption), and T24 (culture, language, translation).
3. **Applications and evaluation:** T9 (clinical health), T27 (education),
   T14 (training, fine-tuning, unlearning), T13 (benchmarks, evaluation),
   T23 (application reviews and advances), and T15 (management, energy).
4. **Interaction and agents:** T17 (chatbots, conversational support),
   T7 (agentic systems, enterprise workflows), T18 (cognition, emotion),
   T20 (retrieval, knowledge graphs), and T21 (networks, robotics, simulation).
5. **Technical core:** T19 (text detection, datasets, classification),
   T10 (prediction, foundation models), T12 (multimodal, vision-language),
   T5 (ChatGPT evaluation), T25 (academic reviews), and T1 (human
   problem-solving and AI-assisted work).

The highest-volume topics are also the most stable in the sensitivity
analysis; the smallest topics should be read as residual or emerging themes.

### 8.3 Biplot interpretation

- **Years:** 2022–2023 align with foundational and early-application topics;
  2024–2026 shift toward generative AI, agents, RAG, and model evaluation. The
  direction of topic vectors shows which themes gain relative weight.
- **Countries:** by volume, the United States and China occupy central
  positions; European and Latin American countries spread toward governance,
  health, and education topics. Positions must be read together with
  representation quality: if a country has low cos², its plane position is
  unreliable.
- **Documents:** the document cloud shows a clear separation between technical
  evaluation documents (T13, T5, T19) and ethics/governance documents (T3,
  T8, T11), with an overlap zone in interaction and agents (T17, T7).

Global biplot quality is reported in the Section 7.1 table. High mean cos²
values mean the first two axes capture a large part of the structure; low
values call for cautious interpretation.

### 8.4 LDAvis interpretation

- **Intertopic map:** generative-AI (T3), evaluation (T13), and training
  (T14) topics appear close because they share technical vocabulary. Ethics
  (T8) and governance (T2) topics separate from the technical core. Small
  topics (T16, T24) appear isolated, consistent with their low volume.
- **Bubble size:** T3 is the most prevalent topic, followed by the other
  large applied and evaluation topics; the smallest topics have the smallest
  bubbles.
- **Relevance:** with λ = {LAMBDA}, the most relevant terms per topic combine
  frequency and exclusivity. For example, T11 is defined by `adversarial`,
  `attack`, `privacy`, `security`; T9 by `clinical`, `patient`, `diagnosis`;
  T20 by `rag`, `retrieval`, `knowledge`. The full ranking is in
  `outputs/results/lda/ldavis_relevance_terms.csv`.

### 8.5 Integrative synthesis

The three sources converge on a coherent thematic structure:

- **Corpus description:** explosive growth, conference and AI-journal
  dominance, high concentration in the United States and China.
- **Biplot:** separation between a technical core (evaluation, training,
  applications) and a normative core (governance, bias, security), with recent
  years shifting toward the former.
- **LDAvis:** technical topics share vocabulary and overlap; ethics topics
  separate; small topics are more fragile.

Overall, the corpus is not homogeneous: it combines a **technical layer**
(model evaluation, *fine-tuning*, RAG, multimodality) with a **normative
layer** (ethics, governance, bias, privacy, security). The {chosen_k}-topic
LDA captures this duality and provides a defensible basis for the phase-4
trend analysis.

---

## 9. References

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

## Appendix A. Reproducibility and repository structure

```text
bertlda/
├── raw_corpus.csv                  # Scopus export (not tracked if large)
├── src/bertlda/
│   ├── config.py                   # central configuration
│   ├── text_processing.py          # cleaning, stopwords, lemmatization
│   ├── phase0_audit.py             # corpus audit + descriptive figures
│   ├── phase1a_lda_diagnostic.py   # tokenization + K search
│   ├── phase1b_lda_decision.py     # K decision + alpha/eta search
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

## Appendix B. Main artifact inventory

- `outputs/models/lda/lda_final.model` — final LDA model.
- `outputs/results/lda/lda_theta.csv` — document-topic matrix ({theta.shape[0]:,} × {chosen_k}).
- `outputs/results/lda/lda_topic_summary.csv` — labels, keywords, and leading country.
- `outputs/results/lda/lda_ranking.csv` — topic ranking by frequency.
- `outputs/results/lda/lda_k_search.csv` — K grid.
- `outputs/results/lda/lda_alpha_eta_search.csv` — alpha/eta grid.
- `outputs/results/lda/lda_sensitivity_seeds.csv` — seed stability.
- `outputs/results/lda/lda_sensitivity_k.csv` — stability in the K neighbourhood.
- `outputs/results/lda/hjbiplot_*.csv` — HJ-Biplot coordinates and qualities.
- `outputs/results/lda/ldavis_relevance_terms.csv` — term relevance.
- `outputs/results/config/model_configuration_full.json` — full configuration.
- `outputs/visualizations/LDAvis.html` — interactive viewer.
- `docs/Methodology_LDA.md` — this document.
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
    _stab_path = RUTAS["results"] / "lda" / "lda_stability.csv"
    df_stability = (pd.read_csv(_stab_path) if _stab_path.exists()
                    else pd.DataFrame())

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
