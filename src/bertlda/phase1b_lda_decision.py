# -*- coding: utf-8 -*-
"""
phase1b_lda_decision.py
=======================
Phase 1b - decide K for LDA, run stability (cross-seed), then search alpha/eta.

Reads:  ck_corpus_base.pkl, ck_corpus_lda.pkl
Writes: ck_k_optimum.pkl, ck_lda_final_params.pkl
        outputs/results/lda/lda_k_decision.csv
        outputs/results/lda/lda_alpha_eta_search.csv
        outputs/results/lda/lda_weight_sensitivity.csv
        outputs/results/lda/lda_stability.csv
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
    guardar_csv, checkpoint_existe, _fmt, print_separador,
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


def stability_k(k, corpus, id2word, escala, n_cores, seeds, passes, docs_tokens, dict_eval):
    from gensim.models.ldamodel import LdaModel
    from sklearn.metrics import adjusted_rand_score

    modelos, dominants, cv_per_seed = [], [], []
    for s in seeds:
        lda = LdaModel(corpus=corpus, num_topics=k, id2word=id2word,
                       alpha="symmetric", eta=None, passes=passes, random_state=s)
        modelos.append(lda)
        dom = [int(np.argmax([p for _, p in
                              lda.get_document_topics(d, minimum_probability=0)]))
              for d in corpus]
        dominants.append(dom)
        m = lda_metrics(lda, corpus, docs_tokens, dict_eval, k)
        cv_per_seed.append(m["cv"])

    ref = modelos[0]
    aris = []
    for i in range(1, len(modelos)):
        from scipy.optimize import linear_sum_assignment
        from sklearn.metrics.pairwise import cosine_similarity
        A = ref.get_topics()
        B = modelos[i].get_topics()
        S = cosine_similarity(A, B)
        rows, cols = linear_sum_assignment(-S)
        mapa = dict(zip(cols, rows))
        alineado = [mapa.get(t, t) for t in dominants[i]]
        aris.append(adjusted_rand_score(dominants[0], alineado))

    ari_medio = float(np.mean(aris)) if aris else float("nan")
    ari_min = float(np.min(aris)) if aris else float("nan")
    return ari_medio, ari_min, cv_per_seed


def main() -> dict:
    asegurar_carpetas()
    print_separador("PHASE 1b — LDA K DECISION + ALPHA/ETA")
    t_total = perf_counter()

    from gensim.models.ldamodel import LdaModel
    from gensim.models.ldamulticore import LdaMulticore

    ck_search = RUTAS["results"] / "lda" / "lda_k_search.csv"
    if not ck_search.exists():
        raise FileNotFoundError(f"{ck_search} not found. Run phase1a first.")
    df_k = pd.read_csv(ck_search).sort_values("K").reset_index(drop=True)

    ck_lda = cargar_checkpoint("ck_corpus_lda")
    docs_tokens = ck_lda["docs_tokens"]
    id2word = ck_lda["id2word"]
    corpus = ck_lda["corpus"]
    dict_eval = ck_lda["dict_eval"]

    base = cargar_checkpoint("ck_corpus_base")
    P = base["params"]

    # Standalone manual candidate override (None = automatic top-N)
    K_CANDIDATOS_MANUAL = None
    N_TOP_STABILITY = 5
    N_SEEDS = CFG["lda"]["n_corridas_stabilityilidad"]
    SEEDS = [CFG["seed"] + i for i in range(N_SEEDS)]
    PASSES_STABILITY = 10
    PESOS_K = {"plateau": 0.30, "stability": 0.30,
                "parsimony": 0.20, "interpretability": 0.20}

    # OVERRIDE_K support
    OVERRIDE_K = None

    print(f"  Candidates evaluated: {df_k['K'].tolist()}")

    ncfg = CFG["compute"]["n_cores"]
    import multiprocessing
    if ncfg is None:
        ncfg = max(1, multiprocessing.cpu_count() - 1)
    NC = ncfg

    ck_stability_path = None
    from pathlib import Path
    ck_stability_file = "ck_k_stability"
    stability = cargar_checkpoint(ck_stability_file) if checkpoint_existe(ck_stability_file) else {}

    if OVERRIDE_K is not None:
        print(f"\n  OVERRIDE_K={OVERRIDE_K}: skipping automatic selection.")
        k_final = OVERRIDE_K
        df_k["score_total"] = float("nan")
    else:
        print("\n-- CRITERIA (all candidates) --")
        df_k["score_plateau"] = criterio_plateau(df_k)
        df_k["score_parsimony"] = criterio_parsimonia(df_k, df_k["K"].min())
        df_k["score_interpretability"] = criterio_interpretabilidad(df_k)
        df_k["score_partial"] = (
            0.30 * df_k["score_plateau"]
            + 0.20 * df_k["score_parsimony"]
            + 0.20 * df_k["score_interpretability"]
        )

        # candidate selection
        if K_CANDIDATOS_MANUAL is not None:
            disponibles = set(df_k["K"])
            faltantes = set(K_CANDIDATOS_MANUAL) - disponibles
            if faltantes:
                raise ValueError(f"Manual candidates not evaluated: {sorted(faltantes)}")
            top_stability = df_k[df_k["K"].isin(K_CANDIDATOS_MANUAL)]
            print(f"  Manual candidates: {sorted(K_CANDIDATOS_MANUAL)}")
        else:
            top_stability = df_k.sort_values("score_partial", ascending=False).head(N_TOP_STABILITY)
            print(f"  Top {N_TOP_STABILITY} by partial score for stability: {sorted(top_stability['K'].tolist())}")

        print(f"\n-- STABILITY ({N_SEEDS} seeds, passes={PASSES_STABILITY}) --")
        for k in sorted(top_stability["K"].tolist()):
            k = int(k)
            entrada = stability.get(k)
            recalc = False
            if entrada is not None:
                if len(entrada.get("seeds", [])) != N_SEEDS:
                    recalc = True
                    print(f"  K={k}: cached seeds count mismatch -> recalc")
            else:
                recalc = True
            if not recalc:
                continue
            t_k = perf_counter()
            ari_medio, ari_min, cv_per_seed = stability_k(
                k, corpus, id2word, P["escala"], NC, SEEDS, PASSES_STABILITY,
                docs_tokens, dict_eval,
            )
            stability[k] = {
                "ARI_medio": ari_medio, "ARI_min": ari_min,
                "cv_per_seed": cv_per_seed,
                "seeds": list(SEEDS),
                "elapsed_s": round(perf_counter() - t_k, 1),
            }
            guardar_checkpoint(stability, ck_stability_file)
            print(f"  K={k:>3} -> ARI medio={ari_medio:.4f} min={ari_min:.4f} ({_fmt(perf_counter() - t_k)})")

        df_k["ARI_medio"] = df_k["K"].map(lambda k: stability.get(int(k), {}).get("ARI_medio", float("nan")))
        df_k["score_stability"] = df_k["ARI_medio"].fillna(0)
        evals = df_k["ARI_medio"].notna()
        if evals.sum() > 1:
            df_k.loc[evals, "score_stability"] = _norm(df_k.loc[evals, "ARI_medio"])

        def _cv_std(k):
            vals = stability.get(int(k), {}).get("cv_per_seed", [])
            return float(np.std(vals)) if len(vals) > 1 else float("nan")
        df_k["cv_std"] = df_k["K"].map(_cv_std)
        df_k["score_stability_cv"] = float("nan")
        con_std = df_k["cv_std"].notna()
        if con_std.sum() > 1:
            df_k.loc[con_std, "score_stability_cv"] = 1 - _norm(df_k.loc[con_std, "cv_std"])
        elif con_std.sum() == 1:
            df_k.loc[con_std, "score_stability_cv"] = 1.0

        df_k["score_total"] = np.where(
            evals,
            PESOS_K["plateau"] * df_k["score_plateau"]
            + PESOS_K["stability"] * df_k["score_stability"]
            + PESOS_K["parsimony"] * df_k["score_parsimony"]
            + PESOS_K["interpretability"] * df_k["score_interpretability"],
            float("nan"),
        )
        ganador = df_k.loc[df_k["score_total"].idxmax()]
        k_final = int(ganador["K"])
        print(f"\n  K CHOSEN: {k_final} (score_total={ganador['score_total']:.4f})")
        print(f"    Cv={ganador['cv']:.4f}  NPMI={ganador['npmi']:+.4f}  ARI={ganador['ARI_medio']:.4f}")

        # Weight sensitivity
        print("\n-- WEIGHT SENSITIVITY --")
        esquemas = {
            "declared": {"plateau": 0.30, "stability": 0.30, "parsimony": 0.20, "interpretability": 0.20},
            "equal": {"plateau": 0.25, "stability": 0.25, "parsimony": 0.25, "interpretability": 0.25},
            "stab_dom": {"plateau": 0.20, "stability": 0.40, "parsimony": 0.20, "interpretability": 0.20},
            "pars_dom": {"plateau": 0.20, "stability": 0.20, "parsimony": 0.40, "interpretability": 0.20},
        }
        rows_sensitivity = []
        for nombre, w in esquemas.items():
            score_w = np.where(
                evals,
                w["plateau"] * df_k["score_plateau"] + w["stability"] * df_k["score_stability"]
                + w["parsimony"] * df_k["score_parsimony"]
                + w["interpretability"] * df_k["score_interpretability"],
                float("nan"),
            )
            k_bajo = int(df_k.loc[np.nanargmax(score_w), "K"])
            rows_sensitivity.append({"scheme": nombre, **w, "K_winner": k_bajo})
            print(f"  {nombre:<12} -> K={k_bajo}")
        df_sens = pd.DataFrame(rows_sensitivity)
        ks_alt = set(df_sens["K_winner"]) - {k_final}
        if ks_alt:
            print(f"  WARNING: K changes under other weight schemes: {sorted(ks_alt)}")
        else:
            print(f"  K={k_final} holds across all {len(esquemas)} schemes.")
        guardar_csv(df_sens, "lda_weight_sensitivity.csv", subdir=RUTAS["results"] / "lda")

    # Save stability summary CSV
    rows_stab = []
    for k, v in stability.items():
        rows_stab.append({
            "K": k, "ARI_medio": v["ARI_medio"], "ARI_min": v["ARI_min"],
            "cv_seed_std": float(np.std(v["cv_per_seed"])) if v["cv_per_seed"] else float("nan"),
            "n_seeds": len(v["seeds"]), "elapsed_s": v["elapsed_s"],
        })
    if rows_stab:
        guardar_csv(pd.DataFrame(rows_stab).sort_values("K"),
                    "lda_stability.csv", subdir=RUTAS["results"] / "lda")

    guardar_csv(df_k, "lda_k_decision.csv", subdir=RUTAS["results"] / "lda")

    # Save K decision checkpoint
    guardar_checkpoint(k_final, "ck_k_optimum")

    # Alpha/eta search with chosen K
    print(f"\n-- ALPHA/ETA SEARCH (K={k_final}) --")
    t0 = perf_counter()
    Cls = LdaModel if (P["escala"] == "MICRO" or NC == 1) else LdaMulticore
    ex = {} if Cls is LdaModel else {"workers": NC}

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
