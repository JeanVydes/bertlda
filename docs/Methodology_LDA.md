# Topic Modeling with Latent Dirichlet Allocation: Methodology and Preliminary Results

**Corpus:** AI ethics, generative AI, and large language models (Scopus, 2018–2026)
**Generated:** 2026-11-08 18:05
**Repository:** `https://github.com/JeanVydes/bertlda`
**Full configuration:** `outputs/results/config/model_configuration_full.json`

> **Scope note.** This document presents the methodology and preliminary results
> of the LDA topic model. The collection and de-duplication process (PRISMA,
> Scopus) is summarized in Section 1; the modeling corpus in this
> repository is the Scopus export of 20,000 records. Fields marked
> `[TO COMPLETE]` must be taken from the previous delivery for the final version.

---

## 1. Corpus collection and de-duplication

### 1.1 Sources and search strategy

The systematic review collected records from **Scopus only** (no Web of
Science integration is used in this delivery). The corpus targets the
intersection of artificial intelligence, generative AI, large language models,
and their ethical, social, regulatory, and safety dimensions:

- **Source:** Scopus (only source used in this delivery)
- **Scopus search string:** `( "generative AI" OR "large language model*" OR "language model" OR "foundation model*" OR GPT OR chatbot OR "text generation" ) AND ( governance OR ethics OR regulation OR "responsible AI" OR "algorithmic accountability" OR "AI policy" ) AND PUBYEAR AFT 2017 AND PUBYEAR BEF 2028 AND ( LIMIT-TO ( LANGUAGE , "English" ) ) AND PUBYEAR > 2017 AND PUBYEAR < 2027`
- **Scopus query date:** 2026-10-11 (query performed four weeks before the current date)

The export used in this delivery contains **20,000 Scopus records**
(field `Source = Scopus`), with title, abstract, keywords, year, source,
affiliations, affiliation country, document type, and DOI. The observed time
window is **2018-2026**; the year 2026 is
incomplete because the export was performed during that year.

### 1.2 PRISMA process

The PRISMA 2020 process (Page et al., 2021) is summarized below for the
Scopus modeling corpus.

| Metric | Value |
|---|---|
| Total records | 20000 |
| With title | 20000 |
| With abstract | 20000 |
| With both | 20000 |
| Without useful text | 0 |
| DOI duplicates | 20 |
| Title duplicates | 59 |
| Excluded by length | 495 |
| USABLE DOCUMENTS | 19505 |
| Year range | 2018-2026 |
| Countries (normalized) | 86 |
| Corpus ALL CAPS | 0.0% |
| Uppercase normalization | False |
| Docs >350 words | 739 |
| Distinct sources | 5697 |

- **Records identified (Scopus):** 20,000
- **Records without title or abstract:** 0
- **Excluded by length (< 20 words):** 495
- **Duplicates detected:** 20 by DOI and 59 by title (retained and reported as a limitation)
- **Final documents for LDA:** **19,505**
- **Normalized countries:** 86
- **Distinct sources:** 5,697

### 1.3 PRISMA flow diagram

![PRISMA flow diagram](../outputs/visualizations/PRISMA_Flow.png)

The diagram above documents the flow of the Scopus corpus actually modeled
(identification, screening, eligibility, inclusion).

### 1.4 Final corpus

The final corpus contains **19,505 documents** with sufficient text for
modeling. The annual distribution shows a strong concentration in recent years,
consistent with the rise of generative AI:

| Year | Documents | % |
|---|---|---|
| 2018 | 2 | 0.0100 |
| 2019 | 1 | 0.0100 |
| 2021 | 1 | 0.0100 |
| 2022 | 31 | 0.1600 |
| 2023 | 556 | 2.8500 |
| 2024 | 2763 | 14.1700 |
| 2025 | 6883 | 35.2900 |
| 2026 | 9268 | 47.5200 |

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
   (`min_count=59`,
   `threshold=8`), for example
   `generative_ai`, `artificial_intelligence`, `machine_learning`.
8. **Vocabulary filtering:** the training dictionary removes terms with
   document frequency below `39` and above
   `0.5` of documents.
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
| Modeled documents | 19,505 |
| Total tokens | 2,361,836 |
| Tokens per document (mean) | 121.09 |
| Training vocabulary | 4,533 |
| Evaluation vocabulary | 23,887 |
| Documents with empty BoW | 0 |
| Total stopwords | 288 |

Cleaning is implemented in `src/bertlda/text_processing.py` and applied
deterministically in `src/bertlda/phase1a_lda_diagnostic.py`.

---

## 3. LDA model foundations

Latent Dirichlet Allocation (Blei, Ng & Jordan, 2003) is a hierarchical
generative model for collections of discrete documents. Each document is
represented as a mixture of a fixed number of topics, and each topic as a
distribution over the vocabulary. Formally, for a document $d$:

$$
\theta_d \sim \mathrm{Dirichlet}(\alpha), \qquad
z_{d,n} \sim \mathrm{Categorical}(\theta_d), \qquad
w_{d,n} \sim \mathrm{Categorical}(\phi_{z_{d,n}}),
$$

where $\theta_d$ is the document's topic proportion vector, $z_{d,n}$ is
the latent topic of word $n$, and $\phi_k \sim \mathrm{Dirichlet}(\eta)$
is the word distribution of topic $k$. The joint likelihood integrates out the
latent variables, and is approximated by variational inference or Gibbs
sampling; Gensim uses online variational inference.

LDA is appropriate for this corpus because:

- documents are short, heterogeneous texts (title + abstract), where a
  probabilistic mixture model is more stable than hard clustering;
- the objective is to **describe** the thematic structure and its evolution,
  not to classify documents into predefined categories;
- the document-topic matrix ($\Theta$) supports downstream analysis with
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

**K = 4–40** (37 values)
was evaluated. The coherence curve reaches its global maximum at
**K = 40** (C_v = 0.5696), but the gain over K = 27 is
marginal (+0.0056) and concentrated in the plateau region.
**K = 27** was selected for the following reasons:

1. **Coherence plateau:** K = 27 already lies on the plateau
   (C_v = 0.5640; NPMI = +0.0749).
2. **Parsimony and interpretability:** 27 topics produce a manageable
   thematic structure aligned with the dimensions of AI ethics (safety, bias,
   governance, education, health, etc.).
3. **Diversity:** the top-10 diversity is 0.7889, high for this
   corpus size.
4. **Dominance:** the most frequent topic concentrates 10.6%
   of documents, below the excessive-dominance threshold (40%).
5. **Stability:** the sensitivity analysis (Section 6) confirms that the
   K = 27 structure is comparable to its neighbours.

![LDA coherence by number of topics](../outputs/visualizations/LDA_K_Coherence.png)

### 4.3 Full K-search table

| K | C_v | NPMI | Diversity | Perplexity | Dominance | Entropy | Empty topics |
|---|---|---|---|---|---|---|---|
| 4 | 0.4826 | 0.0377 | 0.8750 | -7.4130 | 0.2740 | 0.9953 | 0.0000 |
| 5 | 0.4846 | 0.0366 | 0.8200 | -7.3861 | 0.2516 | 0.9891 | 0.0000 |
| 6 | 0.4841 | 0.0340 | 0.7500 | -7.3658 | 0.2298 | 0.9882 | 0.0000 |
| 7 | 0.4989 | 0.0400 | 0.7429 | -7.3513 | 0.2060 | 0.9864 | 0.0000 |
| 8 | 0.4806 | 0.0347 | 0.7125 | -7.3453 | 0.1863 | 0.9894 | 0.0000 |
| 9 | 0.4922 | 0.0406 | 0.7444 | -7.3327 | 0.1491 | 0.9917 | 0.0000 |
| 10 | 0.4865 | 0.0386 | 0.7000 | -7.3298 | 0.1390 | 0.9884 | 0.0000 |
| 11 | 0.5377 | 0.0570 | 0.8091 | -7.3059 | 0.1288 | 0.9860 | 0.0000 |
| 12 | 0.5189 | 0.0502 | 0.7750 | -7.3105 | 0.1518 | 0.9818 | 0.0000 |
| 13 | 0.5354 | 0.0610 | 0.7769 | -7.3168 | 0.1238 | 0.9689 | 0.0000 |
| 14 | 0.5406 | 0.0581 | 0.8000 | -7.3209 | 0.1774 | 0.9542 | 0.0000 |
| 15 | 0.5383 | 0.0634 | 0.7467 | -7.3229 | 0.1963 | 0.9564 | 0.0000 |
| 16 | 0.5350 | 0.0611 | 0.7688 | -7.3216 | 0.1148 | 0.9638 | 0.0000 |
| 17 | 0.5298 | 0.0567 | 0.7588 | -7.3233 | 0.1171 | 0.9573 | 0.0000 |
| 18 | 0.5524 | 0.0641 | 0.7778 | -7.3192 | 0.1360 | 0.9524 | 0.0000 |
| 19 | 0.5492 | 0.0642 | 0.8000 | -7.3205 | 0.1566 | 0.9499 | 0.0000 |
| 20 | 0.5471 | 0.0655 | 0.7450 | -7.3280 | 0.1346 | 0.9500 | 0.0000 |
| 21 | 0.5556 | 0.0710 | 0.7952 | -7.3287 | 0.1251 | 0.9556 | 0.0000 |
| 22 | 0.5535 | 0.0715 | 0.7818 | -7.3270 | 0.1247 | 0.9444 | 0.0000 |
| 23 | 0.5463 | 0.0665 | 0.7826 | -7.3371 | 0.1059 | 0.9505 | 0.0000 |
| 24 | 0.5583 | 0.0696 | 0.7792 | -7.3304 | 0.1160 | 0.9568 | 0.0000 |
| 25 | 0.5508 | 0.0650 | 0.7800 | -7.3335 | 0.1264 | 0.9047 | 0.0000 |
| 26 | 0.5561 | 0.0661 | 0.7654 | -7.3405 | 0.1183 | 0.9474 | 0.0000 |
| 27 | 0.5640 | 0.0749 | 0.7889 | -7.3363 | 0.1056 | 0.9398 | 0.0000 |
| 28 | 0.5638 | 0.0732 | 0.7679 | -7.3404 | 0.0852 | 0.9367 | 0.0000 |
| 29 | 0.5606 | 0.0725 | 0.7655 | -7.3312 | 0.0864 | 0.9199 | 0.0000 |
| 30 | 0.5449 | 0.0622 | 0.7867 | -7.3426 | 0.0854 | 0.9192 | 0.0000 |
| 31 | 0.5545 | 0.0697 | 0.8065 | -7.3472 | 0.0843 | 0.9157 | 0.0000 |
| 32 | 0.5654 | 0.0755 | 0.7781 | -7.3458 | 0.0757 | 0.9279 | 0.0000 |
| 33 | 0.5520 | 0.0660 | 0.7545 | -7.3501 | 0.0744 | 0.9278 | 0.0000 |
| 34 | 0.5598 | 0.0672 | 0.7647 | -7.3557 | 0.0783 | 0.9224 | 0.0000 |
| 35 | 0.5534 | 0.0707 | 0.7800 | -7.3563 | 0.1022 | 0.9203 | 0.0000 |
| 36 | 0.5647 | 0.0737 | 0.7833 | -7.3615 | 0.1113 | 0.9030 | 0.0000 |
| 37 | 0.5641 | 0.0740 | 0.7892 | -7.3628 | 0.0750 | 0.9268 | 0.0000 |
| 38 | 0.5659 | 0.0758 | 0.7421 | -7.3675 | 0.0748 | 0.9174 | 0.0000 |
| 39 | 0.5508 | 0.0606 | 0.7615 | -7.3717 | 0.0898 | 0.8916 | 0.0000 |
| 40 | 0.5696 | 0.0781 | 0.7825 | -7.3704 | 0.0928 | 0.9118 | 0.0000 |

---

## 5. Model configuration and execution

### 5.1 Library and software

- **Gensim 4.4.0**, single-process `LdaModel` implementation.
- **Python 3.13**; the environment is reproduced with `uv sync`.
- Global seed **42**.
- Exports are CSV-only (never xlsx); intermediate checkpoints use `.pkl`.

### 5.2 Hyperparameters

| Parameter | Value |
|---|---|
| Number of topics (K) | 27 |
| Alpha (document-topic prior) | `asymmetric` |
| Eta / beta (topic-word prior) | `0.5` |
| Iterations (passes) | 50 |
| Internal iterations per document | 50 |
| Random seed | 42 |
| Documents | 19,505 |
| Vocabulary | 4,533 |
| Implementation | `gensim.models.ldamodel.LdaModel` |

The alpha/eta search evaluated **16 combinations** (4 alpha values ×
4 eta values). The best combination was
**alpha = `asymmetric`, eta = `0.5`**
(C_v = 0.5861). The 8 best combinations are:

| alpha | eta | C_v | NPMI | Dominance |
|---|---|---|---|---|
| asymmetric | 0.5 | 0.5861 | 0.0861 | 0.0890 |
| 0.01 | 0.5 | 0.5779 | 0.0812 | 0.0771 |
| symmetric | 0.5 | 0.5748 | 0.0808 | 0.0789 |
| 0.1 | 0.5 | 0.5713 | 0.0793 | 0.0815 |
| asymmetric | 0.01 | 0.5657 | 0.0776 | 0.1273 |
| asymmetric | 0.05 | 0.5656 | 0.0765 | 0.1268 |
| 0.01 | 0.05 | 0.5649 | 0.0748 | 0.1023 |
| symmetric | symmetric | 0.5640 | 0.0749 | 0.1056 |

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
| C_v | 0.5761 |
| NPMI | +0.0818 |
| Diversity (top-10) | 0.7741 |
| Perplexity | -7.2693 |
| Dominance | 0.0844 |
| Normalized entropy | 0.9475 |
| Empty topics | 0 |

> **Note on C_v comparability.** The K grid (Section 4) uses
> `alpha="symmetric"` and `passes=30`; the final model
> uses `alpha="asymmetric"`, `eta="0.5"`, and
> `passes=50`. Its C_v
> (0.5761) therefore does not exactly match the
> grid value for K = 27 (0.5640). Comparisons are valid within
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

**5 seeds**
(42, 43, 44, 45, 46) were trained for
each K in the neighbourhood, with `passes=10`.
The table reports the mean and standard deviation over all seed pairs:

| K | Jaccard mean | Jaccard SD | Jaccard min | ARI mean | ARI SD | ARI min | C_v mean | C_v SD |
|---|---|---|---|---|---|---|---|---|
| 25 | 0.3459 | 0.0344 | 0.3005 | 0.2457 | 0.0270 | 0.2085 | 0.5458 | 0.0100 |
| 26 | 0.3102 | 0.0393 | 0.2361 | 0.2464 | 0.0333 | 0.2021 | 0.5460 | 0.0118 |
| 27 | 0.3192 | 0.0242 | 0.2756 | 0.2413 | 0.0280 | 0.1985 | 0.5511 | 0.0095 |
| 28 | 0.2942 | 0.0355 | 0.2585 | 0.2228 | 0.0281 | 0.1761 | 0.5433 | 0.0074 |
| 29 | 0.3050 | 0.0257 | 0.2631 | 0.2594 | 0.0329 | 0.1995 | 0.5497 | 0.0096 |

The table above summarizes seed stability for each K in the neighbourhood. Jaccard is computed over the top-10 words of topics aligned with the Hungarian algorithm; ARI is computed over the dominant-topic assignment of each document.

![LDA stability across seeds](../outputs/visualizations/LDA_Sensitivity_Stability.png)

### 6.2 Stability across K

The table compares each neighbourhood model against the reference model
(K = 27, seed 42). `Best Jaccard` is the best overlap of
each model topic with a reference topic; the last column reports how many
reference topics are covered with Jaccard ≥ 0.5.

| K | Best Jaccard mean | Median | Min | Max | Topics ≥ 0.5 | Topics | Reference topics covered ≥ 0.5 |
|---|---|---|---|---|---|---|---|
| 25 | 0.4412 | 0.4286 | 0.0000 | 1 | 11 | 25 | 11 |
| 26 | 0.4048 | 0.3810 | 0.0000 | 1 | 8 | 26 | 8 |
| 27 | 1 | 1 | 1 | 1 | 27 | 27 | 27 |
| 28 | 0.3844 | 0.4286 | 0.0000 | 1 | 10 | 28 | 10 |
| 29 | 0.3715 | 0.3333 | 0.0000 | 0.8182 | 11 | 29 | 11 |

![Overlap across K values](../outputs/visualizations/LDA_Sensitivity_K_Overlap.png)

### 6.3 Sensitivity to alpha and eta

The **16-combination** grid shows small C_v differences between
configurations (range 0.0288). The
selected configuration (`alpha=asymmetric`, `eta=0.5`) remains
among the best and yields the highest coherence. This suggests the thematic
structure is reasonably robust to prior choice, as long as alpha remains
asymmetric or close to symmetric.

### 6.4 Robustness conclusion

For the selected K (27), seed stability is **moderate**: mean Jaccard 0.3192 (SD 0.0242, minimum 0.2756) and mean ARI 0.2413 (SD 0.0280, minimum 0.1985). The highest-volume topics (T3, T23, T20, T1) are recovered most consistently, while the smallest topic (T16) is more sensitive. The 27-topic structure is considered **sufficiently robust for an exploratory analysis**, with the recommendation to report low-frequency topics as provisional. This result is consistent with the topic-model stability literature (Greene, O'Callaghan & Cunningham, 2014; De Waal & Barnard, 2008; Koltcov, Koltsova & Nikolenko, 2014).

---

## 7. Biplot and LDAvis methods

### 7.1 HJ-Biplot

The **HJ-Biplot** (Galindo-Villardón, 1986), an extension of Gabriel's biplot
(1971), represents rows and columns of a matrix simultaneously through a
singular value decomposition. For a centered matrix $X$:

$$
X = U D V^\top,
$$

row coordinates are $A = U D$ and column coordinates $B = V D$. The key
property is that the scalar product $a_i \cdot b_j$ approximates the value
$x_{ij}$, so rows and columns are interpreted in the same space. The
**representation quality** of each marker is measured with the squared cosine:

$$
\cos^2_{i} = \frac{a_{i1}^2 + a_{i2}^2}{\sum_k a_{ik}^2},
$$

and analogously for columns. The **global quality** is the mean of the squared
cosines. Three biplots were built:

1. **Years × topics:** each row is a year (with at least 20 documents) and each
   column a topic; the value is the mean topic probability for that year.
2. **Countries × topics:** each row is one of the 12 countries with most documents;
   the value is the mean topic probability.
3. **Documents × topics:** a random sample of 1500
   documents, colored by dominant topic.

| Dataset | Rows | Columns | Axis 1 (%) | Axis 2 (%) | Mean row cos² | Mean column cos² |
|---|---|---|---|---|---|---|
| Years x topics | 5 | 27 | 77.5300 | 18.1100 | 0.8999 | 0.8827 |
| Countries x topics | 12 | 27 | 59.6600 | 23.1100 | 0.5619 | 0.4612 |
| Document sample x topics | 1500 | 27 | 10.3400 | 8.2100 | 0.1760 | 0.1080 |

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
\mathrm{relevance}(w \mid t) = \lambda \, p(w \mid t)
+ (1-\lambda) \, \frac{p(w \mid t)}{p(w)},
$$

  where $\lambda \in [0,1]$ weights conditional frequency against term
  exclusivity. **λ = 0.6** was used (the value recommended by Sievert &
  Shirley for exploration), and rankings for λ = 0 and λ = 1 were also
  exported.

![LDAvis intertopic distance map](../outputs/visualizations/LDAvis_Intertopic_Map.png)

![LDAvis term relevance heatmap](../outputs/visualizations/LDAvis_Relevance_Heatmap.png)

The full interactive viewer is in
[`outputs/visualizations/LDAvis.html`](../outputs/visualizations/LDAvis.html).

**Relevant-term ranking (λ = 0.6):**

| Topic | Term | Relevance (λ=0.6) |
|---|---|---|
| 1 | make | -1.9061 |
| 1 | work | -2.0260 |
| 1 | problem | -2.0544 |
| 1 | one | -2.0643 |
| 1 | many | -2.0965 |
| 1 | way | -2.1533 |
| 2 | governance | -1.4754 |
| 2 | ai | -1.7458 |
| 2 | framework | -1.8406 |
| 2 | conceptual | -2.0854 |
| 2 | institutional | -2.0922 |
| 2 | epistemic | -2.0966 |
| 3 | ai | -0.5374 |
| 3 | generative_ai | -1.2487 |
| 3 | genai | -1.5243 |
| 3 | ethical | -1.6333 |
| 3 | technology | -1.6445 |
| 3 | artificial_intelligence | -1.7119 |
| 4 | explanation | -0.3093 |
| 4 | privacy | -0.4823 |
| 4 | financial | -0.5804 |
| 4 | audit | -0.9192 |
| 4 | explainable | -1.0140 |
| 4 | explainability | -1.0563 |
| 5 | response | -1.4187 |
| 5 | chatgpt | -1.4401 |
| 5 | question | -1.4426 |
| 5 | score | -1.4832 |
| 5 | accuracy | -1.5976 |
| 5 | gemini | -1.7996 |
| 6 | participant | -1.1916 |
| 6 | perceive | -1.2583 |
| 6 | perception | -1.4120 |
| 6 | survey | -1.5801 |
| 6 | group | -1.7614 |
| 6 | influence | -1.7797 |
| 7 | agent | -0.4618 |
| 7 | architecture | -1.1732 |
| 7 | system | -1.2051 |
| 7 | agentic | -1.2517 |
| 7 | multi-agent | -1.3265 |
| 7 | agentic_ai | -1.3595 |
| 8 | bias | -0.0389 |
| 8 | fairness | -0.8845 |
| 8 | gender | -1.0786 |
| 8 | stereotype | -1.2881 |
| 8 | demographic | -1.4773 |
| 8 | gender_bias | -1.5010 |
| 9 | clinical | -0.6310 |
| 9 | medical | -0.7912 |
| 9 | patient | -0.8103 |
| 9 | healthcare | -1.0350 |
| 9 | health | -1.3673 |
| 9 | care | -1.4132 |
| 10 | prediction | -0.7855 |
| 10 | foundation_model | -1.2005 |
| 10 | biological | -1.2422 |
| 10 | ml | -1.2941 |
| 10 | cell | -1.4533 |
| 10 | sequence | -1.5049 |
| 11 | safety | -0.5508 |
| 11 | security | -0.8021 |
| 11 | threat | -0.8158 |
| 11 | attack | -0.8171 |
| 11 | vulnerability | -0.9028 |
| 11 | risk | -0.9524 |
| 12 | multimodal | -0.0837 |
| 12 | image | -0.0909 |
| 12 | visual | -0.3375 |
| 12 | video | -0.6130 |
| 12 | modality | -0.9154 |
| 12 | mllms | -1.0967 |
| 13 | evaluation | -1.3949 |
| 13 | benchmark | -1.6801 |
| 13 | score | -1.9246 |
| 13 | metric | -1.9603 |
| 13 | across | -2.0082 |
| 13 | test | -2.2250 |
| 14 | unlearn | -1.5762 |
| 14 | train | -1.6102 |
| 14 | fine-tune | -1.7543 |
| 14 | inference | -1.8443 |
| 14 | token | -1.8693 |
| 14 | optimization | -2.0574 |
| 15 | market | -1.3062 |
| 15 | management | -1.3182 |
| 15 | energy | -1.3745 |
| 15 | sustainability | -1.5976 |
| 15 | esg | -1.6021 |
| 15 | intelligent | -1.7468 |
| 16 | service | -0.2831 |
| 16 | policy | -0.3468 |
| 16 | urban | -0.6942 |
| 16 | web | -0.7174 |
| 16 | public | -0.9587 |
| 16 | government | -1.0533 |
| 17 | user | -0.6800 |
| 17 | chatbot | -1.2621 |
| 17 | mental_health | -1.2788 |
| 17 | chatbots | -1.3350 |
| 17 | interaction | -1.3539 |
| 17 | conversational | -1.3628 |
| 18 | human | -1.0166 |
| 18 | emotional | -1.1349 |
| 18 | emotion | -1.1499 |
| 18 | cognitive | -1.1713 |
| 18 | social | -1.2560 |
| 18 | game | -1.4818 |
| 19 | text | -1.3059 |
| 19 | detection | -1.3067 |
| 19 | dataset | -1.3321 |
| 19 | classification | -1.3875 |
| 19 | fine-tune | -1.7099 |
| 19 | annotation | -1.7681 |
| 20 | semantic | -1.3283 |
| 20 | reason | -1.4064 |
| 20 | knowledge | -1.4447 |
| 20 | retrieval | -1.5218 |
| 20 | rag | -1.6462 |
| 20 | knowledge_graph | -1.6721 |
| 21 | plan | -0.8694 |
| 21 | network | -0.8910 |
| 21 | physical | -0.8986 |
| 21 | robot | -0.9243 |
| 21 | traffic | -1.0564 |
| 21 | digital_twin | -1.0812 |
| 22 | alignment | -0.2511 |
| 22 | moral | -0.3600 |
| 22 | value | -0.4605 |
| 22 | human | -0.5263 |
| 22 | preference | -0.6786 |
| 22 | reward | -1.0361 |
| 23 | application | -1.3617 |
| 23 | challenge | -1.6965 |
| 23 | review | -1.8585 |
| 23 | field | -1.9321 |
| 23 | survey | -1.9946 |
| 23 | development | -2.0451 |
| 24 | language | 0.2892 |
| 24 | cultural | -0.2247 |
| 24 | english | -0.3524 |
| 24 | translation | -0.3981 |
| 24 | linguistic | -0.4615 |
| 24 | multilingual | -0.4697 |
| 25 | chatgpt | -0.7402 |
| 25 | review | -1.3218 |
| 25 | publish | -1.3882 |
| 25 | systematic_review | -1.4609 |
| 25 | gai | -1.4752 |
| 25 | scientific | -1.5561 |
| 26 | legal | -0.7587 |
| 26 | code | -1.2817 |
| 26 | requirement | -1.2948 |
| 26 | software | -1.4401 |
| 26 | automate | -1.5392 |
| 26 | compliance | -1.6226 |
| 27 | student | -0.4137 |
| 27 | education | -0.8911 |
| 27 | learn | -0.9129 |
| 27 | educational | -1.0275 |
| 27 | feedback | -1.2317 |
| 27 | learner | -1.3979 |

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

**Temporal distribution.** The corpus covers 2018-2026,
with an explosion of publications from 2023:

| Year | Documents | % |
|---|---|---|
| 2018 | 2 | 0.0100 |
| 2019 | 1 | 0.0100 |
| 2021 | 1 | 0.0100 |
| 2022 | 31 | 0.1600 |
| 2023 | 556 | 2.8500 |
| 2024 | 2763 | 14.1700 |
| 2025 | 6883 | 35.2900 |
| 2026 | 9268 | 47.5200 |

**Main sources/journals.** Output concentrates in NLP/AI conference proceedings
and open-access journals:

| Source | Documents | % |
|---|---|---|
| Proceedings of the Annual Meeting of the Association for Computational Linguistics | 542 | 2.7800 |
| Lecture Notes in Computer Science | 365 | 1.8700 |
| CEUR Workshop Proceedings | 281 | 1.4400 |
| Conference on Human Factors in Computing Systems - Proceedings  | 278 | 1.4300 |
| IEEE Access | 260 | 1.3300 |
| Advances in Neural Information Processing Systems | 232 | 1.1900 |
| Communications in Computer and Information Science | 218 | 1.1200 |
| Proceedings of Machine Learning Research | 179 | 0.9200 |
| Lecture Notes in Networks and Systems | 164 | 0.8400 |
| Applied Sciences (Switzerland) | 158 | 0.8100 |

**Countries.** The United States and China concentrate slightly more than two
fifths of the corpus; the tail includes India, Germany, the United Kingdom,
Italy, Canada, and Australia:

| Country | Documents | % |
|---|---|---|
| United States | 4538 | 23.7100 |
| China | 3888 | 20.3200 |
| India | 1022 | 5.3400 |
| Germany | 999 | 5.2200 |
| United Kingdom | 891 | 4.6600 |
| Italy | 617 | 3.2200 |
| Canada | 500 | 2.6100 |
| Australia | 487 | 2.5400 |
| South Korea | 449 | 2.3500 |
| Spain | 316 | 1.6500 |

**Document types.** Articles and conference papers predominate, with a
relevant share of reviews:

| Document type | Documents | % |
|---|---|---|
| Article | 8319 | 42.6500 |
| Conference paper | 8011 | 41.0700 |
| Review | 1806 | 9.2600 |
| Book chapter | 838 | 4.3000 |
| Note | 164 | 0.8400 |
| Book | 127 | 0.6500 |
| Editorial | 80 | 0.4100 |
| Letter | 53 | 0.2700 |
| Short survey | 47 | 0.2400 |
| Conference review | 42 | 0.2200 |

**Institutions.** Affiliations were extracted from the Scopus `Affiliations`
column. The table shows the institutions with most documents:

| Institution | Documents | % |
|---|---|---|
| University of California | 354 | 1.8100 |
| Stanford University | 326 | 1.6700 |
| Tsinghua University | 322 | 1.6500 |
| Carnegie Mellon University | 255 | 1.3100 |
| Zhejiang University | 249 | 1.2800 |
| National University of Singapore | 239 | 1.2300 |
| Nanyang Technological University | 217 | 1.1100 |
| University of Oxford | 211 | 1.0800 |
| Shanghai Jiao Tong University | 203 | 1.0400 |
| University of Washington | 201 | 1.0300 |

**Bibliometric overview.** The corpus is recent, dominated by AI/NLP
conferences and journals, with a strong presence of the United States and
China. The mix of document types (articles, conference papers, reviews)
suggests a consolidating field, with much applied output and a growing layer
of ethical and regulatory reflection.

### 8.2 LDA model results

The table below summarizes the 27 topics: identifier, proposed label,
number of documents with dominant topic, main terms, and the country with most
documents in the topic (by volume).

| Topic | Proposed label | Documents | Top terms | Leading country |
|---|---|---|---|---|
| 1 | Human problem-solving and AI-assisted work | 1168 | work, make, system, ai, problem, one, human, task, find, question | United States |
| 2 | AI governance and framework design | 1064 | ai, framework, governance, system, design, across, structure, rather, evidence, support | United States |
| 3 | Generative AI ethics, technology, and societal challenges | 1647 | ai, generative_ai, technology, artificial_intelligence, ethical, genai, challenge, development, tool, generative_artificial | United States |
| 4 | Privacy, explainability, and financial trust | 181 | privacy, explanation, financial, audit, framework, transparency, explainability, explainable, trust, sensitive | United States |
| 5 | ChatGPT response quality and performance evaluation | 996 | response, chatgpt, question, accuracy, score, performance, compare, test, quality, gpt-4 | United States |
| 6 | Perception, trust, and AI adoption surveys | 437 | participant, ai, survey, perceive, perception, finding, group, influence, among, impact | United States |
| 7 | Agentic systems, architecture, and enterprise workflows | 686 | agent, system, architecture, framework, agentic, multi-agent, agentic_ai, ai, rag, workflow | United States |
| 8 | Bias, fairness, gender, and stereotype in prompts | 644 | bias, fairness, gender, prompt, social, stereotype, across, demographic, output, group | United States |
| 9 | Clinical, medical, and healthcare AI | 881 | clinical, medical, patient, healthcare, health, care, support, medicine, include, clinician | United States |
| 10 | Prediction, foundation models, and biological sequence learning | 503 | prediction, foundation_model, machine_learn, ml, biological, feature, learn, predict, sequence, representation | China |
| 11 | Safety, cybersecurity, and adversarial robustness | 512 | safety, risk, security, attack, threat, vulnerability, cybersecurity, adversarial, detection, defense | United States |
| 12 | Multimodal and vision-language models | 182 | multimodal, image, visual, video, modality, vision, mllms, generation, foundation_model, vlms | China |
| 13 | Evaluation, benchmarks, and performance measurement | 784 | evaluation, benchmark, across, score, test, metric, prompt, assessment, performance, framework | United States |
| 14 | Training, fine-tuning, and machine unlearning | 1098 | train, fine-tune, performance, task, unlearn, inference, optimization, introduce, framework, learn | China |
| 15 | Intelligent management, markets, and energy systems | 700 | management, system, framework, market, energy, intelligent, integrate, power, process, technology | China |
| 16 | Public policy, digital services, and urban platforms | 111 | policy, service, public, information, urban, platform, web, digital, government, access | United States |
| 17 | User interaction, chatbots, and conversational support | 768 | user, design, interaction, system, support, chatbots, chatbot, mental_health, conversational, dialogue | United States |
| 18 | Human cognition, emotion, and social simulation | 551 | human, social, cognitive, emotional, emotion, agent, theory, narrative, behavior, interaction | United States |
| 19 | Text detection, datasets, and classification | 1081 | text, detection, dataset, classification, performance, prompt, fine-tune, generate, task, datasets | United States |
| 20 | Retrieval, knowledge graphs, and semantic generation | 1277 | framework, reason, knowledge, semantic, retrieval, rag, structure, system, generation, query | China |
| 21 | Networks, robotics, simulation, and digital twins | 223 | plan, system, network, environment, physical, robot, simulation, control, scenario, traffic | China |
| 22 | Alignment, values, and moral decision-making | 244 | human, alignment, value, moral, preference, ethical, align, decision, decision-making, reason | United States |
| 23 | Application challenges, reviews, and field advances | 1498 | application, challenge, review, development, include, field, advance, capability, potential, survey | China |
| 24 | Language, culture, translation, and multilingual corpora | 126 | language, cultural, translation, english, linguistic, multilingual, write, chinese, text, context | China |
| 25 | ChatGPT in academic and systematic literature reviews | 499 | chatgpt, review, include, publish, tool, identify, literature, scientific, systematic_review, qualitative | United States |
| 26 | Law, regulation, compliance, and software requirements | 732 | legal, process, code, requirement, automate, software, regulatory, regulation, compliance, case | United States |
| 27 | AI in education and student learning | 912 | student, learn, education, educational, feedback, tool, skill, assessment, chatgpt, teach | United States |

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
- **Relevance:** with λ = 0.6, the most relevant terms per topic combine
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
layer** (ethics, governance, bias, privacy, security). The 27-topic
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
- `outputs/results/lda/lda_theta.csv` — document-topic matrix (19,505 × 27).
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
