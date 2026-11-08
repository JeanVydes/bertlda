# -*- coding: utf-8 -*-
"""
phase1b_lda_decision.py
=======================
Phase 1b - decide K for LDA by coherence criteria, then search alpha/eta.

K decision uses three coherence-based criteria (plateau, parsimony and
interpretability) with declared weights; no cross-seed stability is run here
(seed/K stability is covered by phase5_sensitivity, which is the
assignment-required sensitivity analysis).

Reads:  ck_corpus_base.pkl, ck_corpus_lda.pkl
Writes: ck_k_optimum.pkl, ck_lda_final_params.pkl
        outputs/results/lda/lda_k_decision.csv
        outputs/results/lda/lda_alpha_eta_search.csv
        outputs/results/lda/lda_weight_sensitivity.csv
"""
from __future__ import annotations

import os
import sys
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
    guardar_csv, _fmt, print_separador,
)
from phase1a_lda_diagnostic import lda_metrics


def _norm(s):
    r = s.max() - s.min()
    return (s - s.min()) / r if r > 0 else pd.Series(0.5, index=s.index)


def criterio_plateau(df_k):
    cv = df_k.sort_values("K")["cv"].values
    ganancia = np.diff(cv, prepend=cv[0])
    score = 1 - _norm(pd.Series(np.abs(ganancia), index=df_k.sort_values("K").index))
    return score.reindex(df_k.index)


def criterio_parsimonia(df_k, k_min):
    return 1 - _norm(df_k["K"] - k_min)


def criterio_interpretabilidad(df_k):
    return (0.5 * _norm(df_k["npmi"]) + 0.3 * _norm(df_k["diversity"])
            + 0.2 * (1 - _norm(df_k["dominance"])))


def main() -> dict:
    asegurar_carpetas()
    print_separador("PHASE 1b — LDA K DECISION + ALPHA/ETA")
    t_total = perf_counter()

    from gensim.models.ldamodel import LdaModel

    ck_search = RUTAS["results"] / "lda" / "lda_k_search.csv"
    if not ck_search.exists():
        raise FileNotFoundError(f"{ck_search} not found. Run phase1a first.")
    df_k = pd.read_csv(ck_search).sort_values("K").reset_index(drop=True)

    ck_lda = cargar_checkpoint("ck_corpus_lda")
    docs_tokens = (ck_lda.get("docs_tokens") or ck_lda.get("docs_tok")
                   or ck_lda.get("docs_tokensens"))
    id2word = ck_lda["id2word"]
    corpus = ck_lda["corpus"]
    dict_eval = ck_lda["dict_eval"]

    base = cargar_checkpoint("ck_corpus_base")
    P = base["params"]

    # K decision: coherence criteria only (plateau, parsimony, interpretability)
    PESOS_K = {"plateau": 0.40, "parsimony": 0.30, "interpretability": 0.30}

    print(f"  Candidates evaluated: {df_k['K'].tolist()}")

    ncfg = CFG["compute"]["n_cores"]
    import multiprocessing
    if ncfg is None:
        ncfg = max(1, multiprocessing.cpu_count() - 1)
    NC = ncfg

    print("\n-- COHERENCE CRITERIA (all candidates) --")
    df_k["score_plateau"] = criterio_plateau(df_k)
    df_k["score_parsimony"] = criterio_parsimonia(df_k, df_k["K"].min())
    df_k["score_interpretability"] = criterio_interpretabilidad(df_k)
    df_k["score_total"] = (
        PESOS_K["plateau"] * df_k["score_plateau"]
        + PESOS_K["parsimony"] * df_k["score_parsimony"]
        + PESOS_K["interpretability"] * df_k["score_interpretability"]
    )
    # Decision rule (primary): smallest K whose C_v is within 1% of the
    # maximum C_v over the grid -- the first K on the coherence plateau.
    # The weighted composite above is kept as a descriptive record; the
    # plateau-tolerance rule is more defensible than maximising a composite
    # dominated by parsimony (which trivially favours the smallest K).
    cv_max = float(df_k["cv"].max())
    k_best_cv = int(df_k.loc[df_k["cv"].idxmax(), "K"])
    umbral = 0.99 * cv_max
    on_plateau = df_k[df_k["cv"] >= umbral].sort_values("K")
    ganador = on_plateau.iloc[0]
    k_final = int(ganador["K"])
    print(f"\n  C_v max = {cv_max:.4f} at K={k_best_cv}; plateau threshold "
          f"(99% of max) = {umbral:.4f}")
    print(f"  Plateau candidates: {on_plateau['K'].tolist()}")
    print(f"\n  K CHOSEN: {k_final} (smallest K within 1% of max C_v)")
    print(f"    Cv={ganador['cv']:.4f}  NPMI={ganador['npmi']:+.4f}  "
          f"diversity={ganador['diversity']:.4f}  dominance={ganador['dominance']:.4f}")

    # Weight sensitivity (no seed stability component)
    print("\n-- WEIGHT SENSITIVITY --")
    esquemas = {
        "declared": {"plateau": 0.40, "parsimony": 0.30, "interpretability": 0.30},
        "equal": {"plateau": 1/3, "parsimony": 1/3, "interpretability": 1/3},
        "pars_dom": {"plateau": 0.25, "parsimony": 0.50, "interpretability": 0.25},
        "plateau_dom": {"plateau": 0.50, "parsimony": 0.25, "interpretability": 0.25},
    }
    rows_sensitivity = []
    for nombre, w in esquemas.items():
        score_w = (w["plateau"] * df_k["score_plateau"]
                   + w["parsimony"] * df_k["score_parsimony"]
                   + w["interpretability"] * df_k["score_interpretability"])
        k_bajo = int(df_k.loc[score_w.idxmax(), "K"])
        rows_sensitivity.append({"scheme": nombre, **w, "K_winner": k_bajo})
        print(f"  {nombre:<12} -> K={k_bajo}")
    df_sens = pd.DataFrame(rows_sensitivity)
    ks_alt = set(df_sens["K_winner"]) - {k_final}
    if ks_alt:
        print(f"  NOTE: K changes under other weight schemes: {sorted(ks_alt)}")
    else:
        print(f"  K={k_final} holds across all {len(esquemas)} schemes.")
    guardar_csv(df_sens, "lda_weight_sensitivity.csv", subdir=RUTAS["results"] / "lda")

    guardar_csv(df_k, "lda_k_decision.csv", subdir=RUTAS["results"] / "lda")

    # Save K decision checkpoint
    guardar_checkpoint(k_final, "ck_k_optimum")

    # Alpha/eta search with chosen K
    print(f"\n-- ALPHA/ETA SEARCH (K={k_final}) --")
    t0 = perf_counter()
    # Single-process LdaModel for reproducibility (same as final training).
    Cls = LdaModel
    ex = {}

    rows_ae = []
    for alpha in P["alpha_grid"]:
        for eta in P["eta_grid"]:
            try:
                lda = Cls(corpus=corpus, num_topics=k_final, id2word=id2word,
                          alpha=alpha, eta=eta, passes=P["passes_fijo"],
                          random_state=CFG["seed"], **ex)
                m = lda_metrics(lda, corpus, docs_tokens, dict_eval, k_final)
                m.update({"alpha": str(alpha), "eta": str(eta)})
                rows_ae.append(m)
                print(f"  alpha={str(alpha):<12} eta={str(eta):<10} "
                      f"-> Cv={m['cv']:.4f} NPMI={m['npmi']:+.4f} dom={m['dominance']:.3f}")
            except Exception as e:
                print(f"  alpha={alpha} eta={eta} ERROR: {e}")
    df_ae = pd.DataFrame(rows_ae)
    guardar_csv(df_ae, "lda_alpha_eta_search.csv", subdir=RUTAS["results"] / "lda")
    mejor_ae = df_ae.loc[df_ae["cv"].idxmax()]
    alpha_final = mejor_ae["alpha"]
    eta_final = mejor_ae["eta"]
    # restore python types where possible
    def _cast(v):
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                return v
        return v
    alpha_final = _cast(alpha_final)
    eta_final = _cast(eta_final)
    print(f"\n  ALPHA/ETA CHOSEN: alpha={alpha_final}  eta={eta_final}  (Cv={mejor_ae['cv']:.4f})")

    guardar_checkpoint({"alpha": alpha_final, "eta": eta_final, "k": k_final},
                      "ck_lda_final_params")
    guardar_checkpoint({"k_final": k_final, "alpha_final": alpha_final,
                        "eta_final": eta_final}, "ck_lda_final_params_extra")

    print_separador("PHASE 1b COMPLETED")
    print(f"  K={k_final} | alpha={alpha_final} | eta={eta_final}")
    print(f"  Time: {_fmt(perf_counter() - t_total)}")
    print("  Next: phase1c (final LDA model)")
    return {"k_final": k_final, "alpha_final": alpha_final, "eta_final": eta_final}


if __name__ == "__main__":
    main()
