# AGENTS.md — bertlda operational notes

## Where to run

Repo (origin): `https://github.com/JeanVydes/bertlda`  
Branch: `main`  
Clone on new machine:
```bash
git clone https://github.com/JeanVydes/bertlda.git
cd bertlda
uv sync
```

## Inputs

- `raw_corpus.csv` (Scopus export, 20,000 docs). Excluded from Git tracking
  (ignored in `.gitignore`); copy it manually to the new machine.

## Current state (last session)

- Phase 0 **done** — `outputs/results/corpus/*` + `outputs/visualizations/*`
  (world map, annual distribution + trend, country distribution + heatmap,
  source ranking).
- LDA **done**:
  - K=14, alpha=asymmetric, eta=symmetric (chosen by operator override).
  - Model: `outputs/models/lda/lda_final.model`.
  - `outputs/results/lda/lda_theta.csv` (19,505 × 14 topics),
    `lda_topic_summary.csv`, `lda_ranking.csv`,
    `model_configuration_lda.json`.
  - Phase 3 LDA viz: similarity (cosine/Hellinger/Jensen-Shannon), top-words
    heatmap, topic × year, topic × country in `outputs/visualizations/`.
  - Phase 1a grid partial (K=4…23) retained in `lda_k_search.csv`.
- BERTopic (2a/2b/2c) and Phase 4 (trends) **not done**.

## Phase sequence and commands

```bash
# Preprocessing (optional if raw_corpus already committed as CSV)
bertlda --phase preprocess-corpus --input raw_corpus.csv
# or
uv run python src/bertlda/preprocess_corpus.py

# Corpus audit (sets countries from Affiliations, stopword candidates...)
bertlda --phase phase0

# LDA diagnostic (long, resumable)
bertlda --phase phase1a

# LDA decision (stability + K choice + alpha/eta)
bertlda --phase phase1b

# Final LDA model + Theta + ranking + config
bertlda --phase phase1c

# Optional: embeddings comparison (heavy, optional)
bertlda --phase phase2a

# BERTopic grid (heavy)
bertlda --phase phase2b

# After manually copying a row from etapa2_hdbscan.csv into config.py:
#   CFG['seleccion_bertopic'] = {n_neighbors, n_components, min_cluster_size,
#                                 min_samples, metodo, epsilon}
bertlda --phase phase2c

# Visualizations
bertlda --phase phase3

# Temporal trends (OLS / Mann-Kendall / beta regression)
bertlda --phase phase4
```

## Resumability notes

- Phase 1a: resumable via `ck_corpus_lda.pkl` + `lda_k_search.csv`
  (skips completed K values).
- Phase 1b: stability checkpoint saved per K (`ck_k_stability.pkl`); alpha/eta
  grid is small (~16 combos).
- Phase 2a/2b: resumable through checkpoints (`ck_embeddings_*.pkl`,
  `ck_etapa1_umap.pkl`, `ck_etapa2a/b_hdbscan_*.pkl`, `ck_proj_*.pkl`).
- Phase 1c/2c/3/4: not resumable; run end-to-end within the phase.

## Key deliverables (when complete)

- `outputs/models/lda/lda_final.model`
- `outputs/results/lda/lda_theta.csv` (document × topic matrix)
- `outputs/results/lda/lda_topic_summary.csv` (topic labels + keywords)
- `outputs/results/lda/lda_ranking.csv`
- `outputs/results/lda/model_configuration_lda.json`
- `outputs/visualizations/LDA_*.png/.svg` (Phase 3)
- BERTopic: `outputs/models/bertopic/bertopic_final/`,
  `outputs/results/bertopic/resultados_bertopic.csv`, topic info + ranking
- Trends: `outputs/results/trends/trend_analysis_*.csv` +
  `outputs/visualizations/Trend_Grid_*.png`

## Next steps to finish LDA (current session)

1. Wait for Phase 1a to finish (notification auto-fires; do not poll).
2. Run Phase 1b (`bertlda --phase phase1b`) to decide K (stability-weighted)
   and (only if needed) restrict via `CFG['lda']['k_candidatos']`.
3. Run Phase 1c to train the final model and export CSVs.
4. Run Phase 3 for visualizations.
5. Commit & push to origin.

## Note on device

- Embeddings use GPU if available (`CFG['compute']['embeddings_device'] = 'auto'`).
- UMAP, HDBSCAN and LDA are CPU-only (reproducibility / publicability reasons;
  cuML is non-deterministic). No GPU speedup for those.
