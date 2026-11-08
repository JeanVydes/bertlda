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
- LDA **done** (grid completo + decisión por coherencia):
  - K=27, alpha=asymmetric, eta=0.5. Decision rule: smallest K within 1% of
    max C_v (0.5696 at K=40) → plateau candidate K=27 (C_v 0.5640).
  - Model: `outputs/models/lda/lda_final.model`.
  - `outputs/results/lda/lda_theta.csv` (19,505 × 27 topics),
    `lda_topic_summary.csv`, `lda_ranking.csv`,
    `model_configuration_lda.json`.
  - Phase 3 LDA viz: similarity (cosine/Hellinger/Jensen-Shannon), top-words
    heatmap, topic × year, topic × country in `outputs/visualizations/`.
  - Phase 1a grid complete (K=4…40) in `lda_k_search.csv`.
- Sensitivity (Phase 5) **done**: K∈{25,26,27,28,29} × 5 seeds
  (`lda_sensitivity_pairs.csv`, `lda_sensitivity_seeds.csv`,
  `lda_sensitivity_k.csv`, `lda_sensitivity_k_overlap.csv`) + stability /
  K-overlap figures. Highest content stability at K=25.
- Deliverables (Phase 6) **done** for K=27: `docs/Methodology_LDA.md` (the
  assignment report in English, academic style, sections 1–9 + annexes), full
  config `outputs/results/config/model_configuration_full.json`, HJ-Biplot
  (years/countries/documents × topics + cos² tables), LDAvis
  (`outputs/visualizations/LDAvis.html` + intertopic map + relevance tables
  for λ∈{0, 0.6, 1}), PRISMA flow diagram, descriptive extras
  (document types, publishers, institutions).
- BERTopic was **removed entirely** (code, config, CLI, deps, outputs).
  Phase 4 (trends) exists for LDA only, not required by the assignment.

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

# LDA decision (coherence plateau + alpha/eta)
bertlda --phase phase1b

# Final LDA model + Theta + ranking + config
bertlda --phase phase1c

# Visualizations
bertlda --phase phase3

# Temporal trends (OLS / Mann-Kendall / beta regression)
bertlda --phase phase4

# LDA sensitivity (seed stability + K neighbourhood) — resumable
bertlda --phase phase5

# Final deliverables: config JSON, HJ-Biplot, LDAvis, Methodology_LDA.md
bertlda --phase phase6
```

## Resumability notes

- Phase 1a: resumable via `ck_corpus_lda.pkl` + `lda_k_search.csv`
  (skips completed K values).
- Phase 1b: alpha/eta grid is small (~16 combos).
- Phase 5: resumable via `ck_sensitivity.pkl` (skips completed K/seed fits).
- Phase 1c/3/4/6: not resumable; run end-to-end within the phase.

## Key deliverables (when complete)

- `outputs/models/lda/lda_final.model`
- `outputs/results/lda/lda_theta.csv` (document × topic matrix)
- `outputs/results/lda/lda_topic_summary.csv` (topic labels + keywords)
- `outputs/results/lda/lda_ranking.csv`
- `outputs/results/lda/model_configuration_lda.json`
- `outputs/visualizations/LDA_*.png/.svg` (Phase 3)
- Trends (LDA only): `outputs/results/trends/trend_analysis_*.csv` +
  `outputs/visualizations/Trend_Grid_*.png`

## Next steps (current session)

1. Commit & push to origin (already pushed — verify with `git status`).
2. The `[TO COMPLETE]` fields in Section 1.1 of `docs/Methodology_LDA.md`
   (exact Scopus/WoS search strings, query dates, WoS counts) must be filled
   from the previous delivery.

## Note on device

- LDA is CPU-only with single-process `LdaModel` for reproducibility.
- Setting `CFG['compute']['n_cores']` only affects tokenization parallelism.
