# bertlda — LDA topic modeling (Scopus only)

Multi-stage topic-modeling pipeline using **only LDA** (no BERTopic) on a
Scopus export of the AI-ethics / generative-AI literature. All exports are
CSV-only (never xlsx); heavy phases are resumable via checkpoints.

Final model: **K=27** topics (coherence plateau, smallest K within 1% of max
C_v), alpha=asymmetric, eta=0.5 (see `outputs/results/model_configuration_lda.json`,
full configuration in `outputs/results/config/model_configuration_full.json`,
and the report in `docs/Methodology_LDA.md`).

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
| 1a | `phase1a_lda_diagnostic` | LDA tokenization, bigrams, K search K=4–40 (Cv/NPMI/diversity). |
| 1b | `phase1b_lda_decision` | K decision (coherence plateau) + alpha/eta search. |
| 1c | `phase1c_lda_final` | Final LDA model: training, Theta matrix, topics, ranking, config. |
| 3 | `phase3_visualizations` | LDA visualizations (PNG/SVG). |
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

- `outputs/results/` - CSV tables (corpus, LDA, trends).
- `outputs/visualizations/` - PNG/SVG (and HTML for LDAvis).
- `outputs/models/` - trained models.
- `outputs/checkpoints/` - resumable pipeline state (`.pkl`).

## Notes

- All exports are CSV-only (no xlsx).
- Country is extracted from the `Affiliations` column (semicolon-separated segments).
- LDA runs on CPU with single-process `LdaModel` for reproducibility.
- Heavy phases (1a/5) are long and resumable — run them individually rather
  than end-to-end.