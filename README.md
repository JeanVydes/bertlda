# bertlda

Multi-stage topic modeling pipeline (LDA + BERTopic) for Scopus CSV exports,
adapted to CSV-only workflows. Corpus topic: AI ethics.

## Setup

```bash
git clone https://github.com/JeanVydes/bertlda.git
cd bertlda
uv sync
```

## Pipeline overview

Phases run independently (resumable via checkpoints and CSVs):

| Phase | Script | What it does |
|-------|--------|--------------|
| prep | `preprocess_corpus` | Preprocess Scopus CSV: `Title + Abstract` -> tokens. |
| 0 | `phase0_audit` | Corpus audit: doc_id, countries (from Affiliations), years, sources, stopword candidates, world map + charts. |
| 1a | `phase1a_lda_diagnostic` | LDA tokenization, bigrams, K search (Cv/NPMI/diversity). |
| 1b | `phase1b_lda_decision` | Cross-seed stability, weight sensitivity, final K + alpha/eta. |
| 1c | `phase1c_lda_final` | Final LDA model: training, Theta matrix, topics, ranking, config. |
| 2a | `phase2a_embeddings_compare` | Compare embedding models (optional, heavy). |
| 2b | `phase2b_bertopic_grid` | BERTopic grid: UMAP + HDBSCAN coarse-to-fine. |
| 2c | `phase2c_bertopic_final` | Final BERTopic model from a manually selected grid row. |
| 3 | `phase3_visualizations` | LDA + BERTopic visualizations (PNG/SVG/HTML). |
| 4 | `phase4_trends` | Temporal trend analysis (OLS, Mann-Kendall, beta regression). |
| 5 | `phase5_sensitivity` | LDA sensitivity: seed stability (Jaccard/ARI) + K neighbourhood. |
| 6 | `phase6_report` | Deliverables: full config JSON, HJ-Biplot, LDAvis, `docs/Methodology_LDA.md`. |

Invoke via CLI:

```bash
bertlda --list
bertlda --phase phase0
bertlda --phase phase1a
bertlda --phase phase1b
bertlda --phase phase1c
bertlda --phase phase2b
bertlda --phase phase2c   # after editing CFG['seleccion_bertopic'] in config.py
bertlda --phase phase3
bertlda --phase phase4
bertlda --phase phase5
bertlda --phase phase6   # add --skip-ldavis to skip the heavy pyLDAvis HTML step
bertlda --phase preprocess-corpus --input raw_corpus.csv
```

Or directly:

```bash
uv run python src/bertlda/phase0_audit.py
```

## Outputs

- `outputs/results/` - CSV tables (corpus, LDA, BERTopic, trends).
- `outputs/visualizations/` - PNG/SVG (and HTML for BERTopic).
- `outputs/models/` - trained models.
- `outputs/checkpoints/` - resumable pipeline state (`.pkl`).

## Notes

- All exports are CSV-only (no xlsx).
- Country is extracted from the `Affiliations` column (semicolon-separated segments).
- UMAP/HDBSCAN run on CPU for reproducibility; embeddings use GPU when available.
- Heavy phases (1a/2b/2c) are long and resumable — run them individually rather
  than end-to-end.