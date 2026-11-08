# -*- coding: utf-8 -*-
"""
phase5_sensitivity.py
=====================
Phase 5 - LDA sensitivity analysis (assignment section 6).

The analysis answers three questions:
  1. Are topics stable when the random seed changes (same K)?
  2. Are topics stable in a neighbourhood of the chosen K?
  3. How sensitive is the model to alpha/eta? (the grid itself is produced by
     phase1b; this phase summarises it and adds the stability diagnostics).

Metrics
-------
* Jaccard overlap between the top-10 words of aligned topics.
* Adjusted Rand Index (ARI) between the dominant-topic assignments of two runs.
* Topic alignment is solved with the Hungarian algorithm on the cosine
  similarity between topic-word distributions.

Outputs (CSV, PNG/SVG)
----------------------
outputs/results/lda/lda_sensitivity_pairs.csv
outputs/results/lda/lda_sensitivity_seeds.csv
outputs/results/lda/lda_sensitivity_k.csv
outputs/visualizations/LDA_Sensitivity_Stability.{png,svg}
outputs/visualizations/LDA_Sensitivity_K_Overlap.{png,svg}
outputs/checkpoints/ck_sensitivity.pkl  (resumable)
"""
from __future__ import annotations

import os
import sys
from itertools import combinations
from time import perf_counter

import numpy as np
import pandas as pd

try:
    _DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _DIR = os.getcwd()
sys.path.insert(0, _DIR)

from config import CFG, RUTAS
from pipeline_base import (
    asegurar_carpetas, cargar_checkpoint, guardar_checkpoint,
    guardar_csv, checkpoint_existe, _fmt, print_separador,
)

TOP_N = int(CFG["lda"]["top_words_stability"])
PASSES = int(CFG["lda"]["sensitivity_passes"])
K_NEAR = [int(k) for k in CFG["lda"]["sensitivity_k"]]
SEEDS = [int(s) for s in CFG["lda"]["sensitivity_seeds"]]
ALPHA = "asymmetric"
ETA = "symmetric"


def _jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _fit_one(k: int, seed: int, corpus, id2word):
    from gensim.models.ldamodel import LdaModel

    lda = LdaModel(
        corpus=corpus, num_topics=k, id2word=id2word,
        alpha=ALPHA, eta=ETA, passes=PASSES, random_state=seed,
    )
    top_list = [[w for w, _ in lda.show_topic(i, topn=TOP_N)] for i in range(k)]
    dominant = np.array([
        int(max(lda.get_document_topics(d, minimum_probability=0),
                key=lambda x: x[1])[0])
        for d in corpus
    ], dtype=np.int32)
    return {
        "top": [set(x) for x in top_list],
        "top_list": top_list,
        "dominant": dominant,
        "topics": lda.get_topics().astype(np.float64),
        "k": k,
        "seed": seed,
    }


def _align(ref, other):
    """Return a mapping other_topic -> ref_topic and the cosine matrix."""
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics.pairwise import cosine_similarity

    sim = cosine_similarity(ref["topics"], other["topics"])
    rows, cols = linear_sum_assignment(-sim)
    # rows: ref topic, cols: other topic
    mapping = {int(c): int(r) for r, c in zip(rows, cols)}
    return mapping, sim


def _pair_metrics(ref, other):
    from sklearn.metrics import adjusted_rand_score

    mapping, _ = _align(ref, other)
    jac = []
    for t_other, t_ref in mapping.items():
        jac.append(_jaccard(ref["top"][t_ref], other["top"][t_other]))
    aligned = np.array([mapping.get(int(t), int(t)) for t in other["dominant"]],
                       dtype=np.int32)
    ari = float(adjusted_rand_score(ref["dominant"], aligned))
    return float(np.mean(jac)), float(np.min(jac)), ari


def _coherence(topics_words, docs_tokens, dict_eval):
    from gensim.models.coherencemodel import CoherenceModel

    words = topics_words
    out = {}
    for key, measure in (("cv", "c_v"), ("npmi", "c_npmi")):
        try:
            out[key] = round(float(CoherenceModel(
                topics=words, texts=docs_tokens, dictionary=dict_eval,
                coherence=measure, processes=1).get_coherence()), 4)
        except Exception:
            out[key] = float("nan")
    return out


def _guardar_fig(fig, nombre):
    for fmt in CFG["viz"]["formatos"]:
        fig.savefig(RUTAS["visualizations"] / f"{nombre}.{fmt}",
                    dpi=CFG["viz"]["dpi"], bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  -> {nombre}.{{{','.join(CFG['viz']['formatos'])}}}")


def _plot_stability(df_seed, chosen_k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    x = df_seed["K"].values

    ax = axes[0]
    ax.errorbar(x, df_seed["jaccard_mean"], yerr=df_seed["jaccard_std"],
                marker="o", capsize=3, color="#2166AC")
    ax.axvline(chosen_k, ls="--", color="#B2182B", lw=1.2,
               label=f"chosen K={chosen_k}")
    ax.set_xlabel("K"); ax.set_ylabel("Jaccard (top-10 words)")
    ax.set_title("Seed stability: topic content")
    ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    ax.errorbar(x, df_seed["ari_mean"], yerr=df_seed["ari_std"],
                marker="o", capsize=3, color="#1B7837")
    ax.axvline(chosen_k, ls="--", color="#B2182B", lw=1.2)
    ax.set_xlabel("K"); ax.set_ylabel("Adjusted Rand Index")
    ax.set_title("Seed stability: document assignment")
    ax.set_ylim(-0.05, 1); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.errorbar(x, df_seed["cv_mean"], yerr=df_seed["cv_std"],
                marker="o", capsize=3, color="#762A83")
    ax.axvline(chosen_k, ls="--", color="#B2182B", lw=1.2)
    ax.set_xlabel("K"); ax.set_ylabel("C_v coherence")
    ax.set_title("Coherence across seeds")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _guardar_fig(fig, "LDA_Sensitivity_Stability")


def _plot_k_overlap(df_overlap, chosen_k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Pivot: rows = K, columns = reference topics
    mat = df_overlap.pivot_table(
        index="K", columns="reference_topic", values="jaccard_best",
        aggfunc="first",
    ).sort_index()
    fig, ax = plt.subplots(figsize=(max(7, mat.shape[1] * 0.5),
                                    max(3.5, mat.shape[0] * 0.55)))
    im = ax.imshow(mat.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels([f"R{int(c)}" for c in mat.columns], fontsize=8)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels([f"K={int(i)}" for i in mat.index], fontsize=8)
    ax.set_xlabel(f"Reference topics (K={chosen_k})")
    ax.set_ylabel("Model with K topics")
    ax.set_title("Best-match Jaccard overlap against the chosen model")
    cb = fig.colorbar(im, ax=ax, shrink=0.75)
    cb.set_label("Jaccard (top-10 words)", fontsize=9)
    fig.tight_layout()
    _guardar_fig(fig, "LDA_Sensitivity_K_Overlap")


def main() -> dict:
    asegurar_carpetas()
    print_separador("PHASE 5 — LDA SENSITIVITY ANALYSIS")
    t_total = perf_counter()

    from gensim.models.ldamodel import LdaModel

    ck_lda = cargar_checkpoint("ck_corpus_lda")
    docs_tokens = (ck_lda.get("docs_tokens") or ck_lda.get("docs_tok")
                   or ck_lda.get("docs_tokensens"))
    id2word = ck_lda["id2word"]
    corpus = ck_lda["corpus"]
    dict_eval = ck_lda["dict_eval"]

    try:
        chosen_k = int(cargar_checkpoint("ck_k_optimum"))
    except Exception:
        chosen_k = int(CFG["lda"].get("k_default", 14))

    global K_NEAR
    if chosen_k not in K_NEAR:
        K_NEAR = sorted(max(4, chosen_k + d) for d in (-2, -1, 0, 1, 2))
        print(f"  Sensitivity K neighbourhood adjusted to {K_NEAR}")

    print(f"  Documents: {len(corpus):,} | vocab: {len(id2word):,}")
    print(f"  K neighbourhood: {K_NEAR} | seeds: {SEEDS} | passes={PASSES}")
    print(f"  Reference K: {chosen_k}")

    ck_name = "ck_sensitivity"
    state = cargar_checkpoint(ck_name) if checkpoint_existe(ck_name) else {}
    if not isinstance(state, dict):
        state = {}

    # ------------------------------------------------------------------
    # 1. Fit models for each (K, seed) and compute coherence
    # ------------------------------------------------------------------
    for k in K_NEAR:
        entry = state.setdefault(str(k), {})
        for seed in SEEDS:
            key = str(seed)
            if key in entry:
                continue
            t0 = perf_counter()
            fitted = _fit_one(k, seed, corpus, id2word)
            met = _coherence(fitted["top_list"], docs_tokens, dict_eval)
            fitted.update(met)
            fitted["elapsed_s"] = round(perf_counter() - t0, 1)
            entry[key] = fitted
            guardar_checkpoint(state, ck_name)
            print(f"  K={k:>2} seed={seed} -> Cv={met['cv']:.4f} "
                  f"NPMI={met['npmi']:+.4f} ({_fmt(perf_counter() - t0)})")

    # ------------------------------------------------------------------
    # 2. Seed stability: pairwise Jaccard + ARI
    # ------------------------------------------------------------------
    rows_pairs = []
    rows_seed = []
    for k in K_NEAR:
        entry = state[str(k)]
        jac_means, ari_vals, cv_vals = [], [], []
        for s_a, s_b in combinations(SEEDS, 2):
            ref = entry[str(s_a)]
            other = entry[str(s_b)]
            jm, jmin, ari = _pair_metrics(ref, other)
            rows_pairs.append({
                "K": k, "seed_a": s_a, "seed_b": s_b,
                "jaccard_mean": round(jm, 4),
                "jaccard_min": round(jmin, 4),
                "ari": round(ari, 4),
                "cv_a": ref.get("cv"), "cv_b": other.get("cv"),
            })
            jac_means.append(jm); ari_vals.append(ari)
        cv_vals = [entry[str(s)].get("cv", np.nan) for s in SEEDS]
        rows_seed.append({
            "K": k,
            "jaccard_mean": round(float(np.mean(jac_means)), 4),
            "jaccard_std": round(float(np.std(jac_means)), 4),
            "jaccard_min": round(float(np.min(jac_means)), 4),
            "ari_mean": round(float(np.mean(ari_vals)), 4),
            "ari_std": round(float(np.std(ari_vals)), 4),
            "ari_min": round(float(np.min(ari_vals)), 4),
            "cv_mean": round(float(np.nanmean(cv_vals)), 4),
            "cv_std": round(float(np.nanstd(cv_vals)), 4),
            "n_seeds": len(SEEDS),
            "n_pairs": len(jac_means),
        })

    df_pairs = pd.DataFrame(rows_pairs)
    df_seed = pd.DataFrame(rows_seed).sort_values("K")
    guardar_csv(df_pairs, "lda_sensitivity_pairs.csv",
                subdir=RUTAS["results"] / "lda")
    guardar_csv(df_seed, "lda_sensitivity_seeds.csv",
                subdir=RUTAS["results"] / "lda")

    # ------------------------------------------------------------------
    # 3. Neighbourhood of K: best-match Jaccard against the reference K
    # ------------------------------------------------------------------
    rows_k = []
    ref = state.get(str(chosen_k), {}).get(str(SEEDS[0]))
    if ref is not None:
        for k in K_NEAR:
            model = state[str(k)][str(SEEDS[0])]
            mapping, _ = _align(ref, model)
            best = []
            for t_model in range(k):
                if t_model in mapping:
                    t_ref = mapping[t_model]
                    best.append(_jaccard(ref["top"][t_ref],
                                         model["top"][t_model]))
                else:
                    best.append(0.0)
            # coverage of the reference topics by any model topic
            ref_to_model = {int(r): int(m) for m, r in mapping.items()}
            covered = set(ref_to_model)
            rows_k.append({
                "K": k,
                "reference_K": chosen_k,
                "jaccard_best_mean": round(float(np.mean(best)), 4),
                "jaccard_best_median": round(float(np.median(best)), 4),
                "jaccard_best_min": round(float(np.min(best)), 4),
                "jaccard_best_max": round(float(np.max(best)), 4),
                "topics_matched_ge_0_5": int(sum(v >= 0.5 for v in best)),
                "topics_total": int(k),
                "reference_topics_covered_ge_0_5": int(sum(
                    1 for t in covered
                    if _jaccard(ref["top"][t],
                                model["top"][ref_to_model[t]]) >= 0.5
                )),
            })
            # One row per reference topic for the heatmap
            for t_model, t_ref in mapping.items():
                rows_k.append({
                    "K": k,
                    "reference_K": chosen_k,
                    "reference_topic": int(t_ref + 1),
                    "model_topic": int(t_model + 1),
                    "jaccard_best": round(_jaccard(
                        ref["top"][t_ref], model["top"][t_model]), 4),
                })
        # Separate summary and long table
        df_k_long = pd.DataFrame([r for r in rows_k if "reference_topic" in r])
        df_k_summary = pd.DataFrame([r for r in rows_k if "reference_topic" not in r])
        guardar_csv(df_k_summary, "lda_sensitivity_k.csv",
                    subdir=RUTAS["results"] / "lda")
        guardar_csv(df_k_long, "lda_sensitivity_k_overlap.csv",
                    subdir=RUTAS["results"] / "lda")
        _plot_k_overlap(df_k_long, chosen_k)
    else:
        print(f"  Reference model K={chosen_k} not found in sensitivity grid; "
              "skipping K-overlap table.")
        df_k_summary = pd.DataFrame()

    # ------------------------------------------------------------------
    # 4. Alpha/eta summary (produced by phase1b)
    # ------------------------------------------------------------------
    ae_path = RUTAS["results"] / "lda" / "lda_alpha_eta_search.csv"
    if ae_path.exists():
        ae = pd.read_csv(ae_path)
        print(f"  Alpha/eta grid: {len(ae)} combinations | "
              f"best Cv={ae['cv'].max():.4f} "
              f"(alpha={ae.loc[ae['cv'].idxmax(), 'alpha']}, "
              f"eta={ae.loc[ae['cv'].idxmax(), 'eta']})")
    else:
        ae = pd.DataFrame()

    # ------------------------------------------------------------------
    # 5. Plots
    # ------------------------------------------------------------------
    _plot_stability(df_seed, chosen_k)

    print_separador("PHASE 5 COMPLETED")
    print(f"  Seed pairs: {len(df_pairs)} | K values: {len(df_seed)}")
    if len(df_seed):
        best = df_seed.loc[df_seed["jaccard_mean"].idxmax()]
        print(f"  Highest content stability: K={int(best['K'])} "
              f"(Jaccard={best['jaccard_mean']:.4f}, ARI={best['ari_mean']:.4f})")
    print(f"  Time: {_fmt(perf_counter() - t_total)}")
    return {"n_pairs": len(df_pairs), "n_k": len(df_seed)}


if __name__ == "__main__":
    main()
