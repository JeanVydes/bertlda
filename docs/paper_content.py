# -*- coding: utf-8 -*-
"""Content of the final paper 2611.00002, in document order.

This module is generated once from the manuscript and is the sole
source of text/tables for make_paper_docx.py. No DOCX template is used.
"""

PAPER_ITEMS = [{'type': 'p', 'text': 'Article', 'style': 'Normal', 'align': None, 'img': None},
 {'type': 'p',
  'text': 'Topic Modeling of the Generative AI and Large Language Model Ethics and Governance Literature '
          'Using Latent Dirichlet Allocation, HJ-Biplot, and LDAvis',
  'style': 'Normal',
  'align': 'LEFT (0)',
  'img': None},
 {'type': 'p', 'text': 'Jean Vides 1,*', 'style': 'Normal', 'align': 'LEFT (0)', 'img': None},
 {'type': 'p',
  'text': '1 Systems Engineering Program, Faculty of Engineering, Universidad del Magdalena, Santa Marta, '
          'Colombia',
  'style': 'Normal',
  'align': 'LEFT (0)',
  'img': None},
 {'type': 'p',
  'text': '* Correspondence: jcvides@unimagdalena.edu.co',
  'style': 'Normal',
  'align': 'LEFT (0)',
  'img': None},
 {'type': 'p',
  'text': 'Abstract: The rapid diffusion of generative artificial intelligence (AI) and large language '
          'models (LLMs) has produced a fast-growing, interdisciplinary literature on their ethical, '
          'regulatory, and safety implications, yet its thematic structure has rarely been mapped with '
          'reproducible, stability-checked methods. We applied Latent Dirichlet Allocation (LDA) to the '
          'titles and abstracts of 19,505 Scopus documents (2018–2026) retrieved with a Boolean query on '
          'generative AI, language models, governance, and ethics, following PRISMA-style screening. The '
          'number of topics (K) was selected within K = 4–40 using Cv and NPMI coherence, topic diversity, '
          'and dominance, and robustness was examined across random seeds, neighbouring K, and Dirichlet '
          'priors. Topics were explored with HJ-Biplots and LDAvis. A 27-topic model (Cv = 0.576; NPMI = '
          '0.082) separated a large technical layer (evaluation, fine-tuning, retrieval, agents) from a '
          'smaller normative layer (governance, bias, privacy, security, regulation). Of the documents, '
          '82.8% were published in 2025–2026, and the United States and China contributed 44.0%. Seed '
          'stability was moderate (mean Jaccard = 0.319; mean ARI = 0.241). The literature is dominated by '
          'technical work on models, with ethics and governance forming a distinct minority. The model is '
          'suitable for exploratory mapping, but low-volume topics should be treated as provisional.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'Keywords: topic modeling; latent Dirichlet allocation; generative AI; large language models; AI '
          'ethics; AI governance; topic coherence; HJ-Biplot; LDAvis; bibliometric analysis',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p', 'text': '', 'style': 'Normal', 'align': None, 'img': None},
 {'type': 'p',
  'text': 'Academic Editor: [to be assigned]',
  'style': 'Normal',
  'align': 'LEFT (0)',
  'img': None},
 {'type': 'p',
  'text': 'Received: [date]; Revised: [date]; Accepted: [date]; Published: [date]',
  'style': 'Normal',
  'align': 'LEFT (0)',
  'img': None},
 {'type': 'p',
  'text': 'Copyright: © 2026 by the authors. Submitted to Information for possible open access publication '
          'under the terms and conditions of the Creative Commons Attribution (CC BY) license '
          '(https://creativecommons.org/licenses/by/4.0/).',
  'style': 'Normal',
  'align': 'LEFT (0)',
  'img': None},
 {'type': 'p', 'text': '', 'style': 'Normal', 'align': None, 'img': None},
 {'type': 'p', 'text': '1. Introduction', 'style': 'Heading 1', 'align': None, 'img': None},
 {'type': 'p',
  'text': 'Scholarship on the ethical, social, and regulatory implications of artificial intelligence (AI) '
          'has grown from principle-setting exercises, such as the global landscape of AI ethics guidelines '
          '[1] and the AI4People ethical framework [2], into a broad field that now includes critical '
          'assessments of how guidelines are translated into practice [3,4] and analyses of the risks posed '
          'by large language models (LLMs) [5,6]. The public release of widely accessible generative AI '
          'systems has accelerated this output: in the corpus analyzed here, 82.8% of the documents were '
          'published in 2025 and 2026.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'Mapping this literature has largely relied on qualitative reviews of guidelines and policy '
          'documents. Jobin et al. [1] identified a convergence around a limited set of principles, and '
          'Hagendorff [3] compared major guidelines and questioned their practical effectiveness. '
          'Computational text mining has also been applied to ethical AI policy documents: Saheb and Saheb '
          '[7] used Latent Dirichlet Allocation (LDA) on 57 national policy documents from 24 countries and '
          'synthesized 19 topics into six themes. In the scholarly literature, topic-model-based reviews '
          'have been published for specific application domains such as AI in auditing [8], where '
          'ethics-related topics accounted for a small share of the text. Text-as-data methods [9], and '
          'topic models in particular [10], offer a reproducible alternative to manual reading once a corpus '
          'reaches tens of thousands of documents.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'Three gaps remain. First, within the sources reviewed here, we found no study that maps the '
          'scholarly (as opposed to policy) literature jointly addressing generative AI and LLMs with '
          'governance and ethics at the scale reached since 2023. Second, topic-model applications often '
          'report a single fitted model without examining its sensitivity to the number of topics, random '
          'initialization, or priors, although LDA solutions are known to vary between runs [11,12] and its '
          'priors matter [13]. Third, coherence scores alone are an incomplete guide to interpretability '
          '[14], which motivates complementary multivariate and interactive diagnostics.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'The objectives of this study are therefore: (i) to build and document a reproducible corpus of '
          'Scopus records on generative AI, LLMs, and their governance and ethical dimensions; (ii) to fit '
          'and select an LDA topic model using multiple coherence and structure diagnostics; (iii) to '
          'quantify the robustness of the solution to random seeds, neighbouring numbers of topics, and '
          'Dirichlet priors; and (iv) to interpret the resulting thematic structure, including its temporal '
          'and geographic patterns, with HJ-Biplots and LDAvis. Section 2 describes the materials and '
          'methods, Section 3 presents the results, Section 4 discusses them in relation to the literature, '
          'and Section 5 concludes.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p', 'text': '2. Materials and Methods', 'style': 'Heading 1', 'align': None, 'img': None},
 {'type': 'p', 'text': '2.1. Corpus Retrieval', 'style': 'Heading 2', 'align': None, 'img': None},
 {'type': 'p',
  'text': 'Records were retrieved from Scopus [15] on 11 October 2026 with the following Boolean query:',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '( "generative AI" OR "large language model*" OR "language model" OR "foundation model*" OR GPT OR '
          'chatbot OR "text generation" ) AND ( governance OR ethics OR regulation OR "responsible AI" OR '
          '"algorithmic accountability" OR "AI policy" ) AND PUBYEAR AFT 2017 AND PUBYEAR BEF 2028 AND ( '
          'LIMIT-TO ( LANGUAGE , "English" ) ) AND PUBYEAR > 2017 AND PUBYEAR < 2027',
  'style': 'Normal',
  'align': 'LEFT (0)',
  'img': None},
 {'type': 'p',
  'text': 'The query intersects terms for generative AI and language models with terms for governance, '
          'ethics, regulation, and policy, and restricts the publication years (2018–2026) and the language '
          '(English). The number of records was capped at 20,000 because of the computational budget and '
          'processing time available for model fitting, the K search, and the sensitivity analyses; the '
          'modeling corpus is therefore a capped subset of the full Scopus result set. The export contained '
          'title, abstract, author and index keywords, publication year, source, affiliations, affiliation '
          'country, document type, and DOI. The year 2026 is incomplete because the export was made during '
          'that year.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '2.2. Screening and Corpus Definition',
  'style': 'Heading 2',
  'align': None,
  'img': None},
 {'type': 'p',
  'text': 'Screening followed the flow-diagram conventions of PRISMA 2020 [16], as summarized in Table 1 and '
          'Figure 1. All 20,000 records contained both title and abstract. Records with fewer than 20 words '
          'of useful text (n = 495) were excluded. Duplicates were detected by DOI (n = 20) and by title (n '
          '= 59) but were flagged and retained; their implications are discussed in Section 4.5. The final '
          'modeling corpus comprises 19,505 documents from 86 normalized countries and 5,697 distinct '
          'sources.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'Table 1. Corpus screening summary.',
  'style': 'Normal',
  'align': 'LEFT (0)',
  'img': None},
 {'type': 'tbl',
  'rows': [['Metric', 'Value'],
           ['Records identified (Scopus)', '20,000'],
           ['Records with title and abstract', '20,000'],
           ['Records without useful text', '0'],
           ['DOI duplicates (flagged, retained)', '20'],
           ['Title duplicates (flagged, retained)', '59'],
           ['Records excluded by length (< 20 words)', '495'],
           ['Documents in the modeling corpus', '19,505'],
           ['Documents with more than 350 words', '739'],
           ['Countries (normalized)', '86'],
           ['Distinct sources', '5,697']]},
 {'type': 'p', 'text': '', 'style': 'Normal', 'align': None, 'img': None},
 {'type': 'p', 'text': '', 'style': 'Normal', 'align': 'CENTER (1)', 'img': 'PRISMA_Flow.png'},
 {'type': 'p',
  'text': 'Figure 1. PRISMA-style flow diagram of the Scopus records screened and modeled. Generated by the '
          'pipeline script phase6_report.py.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p', 'text': '2.3. Text Preprocessing', 'style': 'Heading 2', 'align': None, 'img': None},
 {'type': 'p',
  'text': 'Title and abstract were concatenated into a single text field. Author and index keywords were '
          'available but were not appended, to avoid duplicating terms and to keep a homogeneous '
          'representation across documents. Because preprocessing choices materially affect unsupervised '
          'text models [17] and stopword lists should be adapted to the corpus [18], all steps were '
          'deterministic and are listed below, following standard information-retrieval practice [19] and '
          'implemented with NLTK [20]:',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'Unicode normalization and lowercasing.',
  'style': 'List Paragraph',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'Removal of editorial noise: URLs, e-mail addresses, numeric citations, copyright and license '
          'notices, and “published by” sentences.',
  'style': 'List Paragraph',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'Whitespace tokenization of the cleaned text.',
  'style': 'List Paragraph',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'WordNet [21] lemmatization (nouns first, then verbs); stemming was not applied, to preserve '
          'interpretability.',
  'style': 'List Paragraph',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'Four-layer stopword removal (288 stopwords in total): English functional stopwords (NLTK); '
          'editorial terms (e.g., publisher names, “journal”, “abstract”); methodological terms (e.g., '
          '“study”, “results”, “method”); and generic AI-ethics domain terms, while preserving valid '
          'acronyms.',
  'style': 'List Paragraph',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'Length filtering: tokens shorter than three characters were discarded, except valid acronyms (ai, '
          'llm, gpt, nlp, gdpr).',
  'style': 'List Paragraph',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'Bigram detection with the phrase-scoring approach of Mikolov et al. [22] as implemented in Gensim '
          '[23] (minimum count 59, threshold 8), producing terms such as generative_ai and machine_learning.',
  'style': 'List Paragraph',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'Vocabulary filtering: terms appearing in fewer than 39 documents or in more than 50% of documents '
          'were removed from the training dictionary.',
  'style': 'List Paragraph',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'Processing used Python 3.13 with NLTK, Gensim 4.4, pandas [24], and NumPy [25]. Table 2 '
          'summarizes the output.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p', 'text': 'Table 2. Preprocessing output.', 'style': 'Normal', 'align': 'LEFT (0)', 'img': None},
 {'type': 'tbl',
  'rows': [['Metric', 'Value'],
           ['Modeled documents', '19,505'],
           ['Total tokens', '2,361,836'],
           ['Tokens per document (mean)', '121.09'],
           ['Training vocabulary (terms)', '4,533'],
           ['Evaluation vocabulary (terms)', '23,887'],
           ['Documents with empty bag-of-words', '0'],
           ['Total stopwords', '288']]},
 {'type': 'p', 'text': '', 'style': 'Normal', 'align': None, 'img': None},
 {'type': 'p', 'text': '2.4. Latent Dirichlet Allocation', 'style': 'Heading 2', 'align': None, 'img': None},
 {'type': 'p',
  'text': 'LDA [26] is a hierarchical generative model in which each document is a mixture of a fixed number '
          'of latent topics and each topic is a distribution over the vocabulary. For document d with Nd '
          'words, the generative process is',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p', 'text': 'θ_d ~ Dirichlet(α)\t\t(1)', 'style': 'Normal', 'align': None, 'img': None},
 {'type': 'p', 'text': 'z_{d,n} ~ Categorical(θ_d)\t\t(2)', 'style': 'Normal', 'align': None, 'img': None},
 {'type': 'p',
  'text': 'w_{d,n} ~ Categorical(φ_{z_{d,n}})\t\t(3)',
  'style': 'Normal',
  'align': None,
  'img': None},
 {'type': 'p', 'text': 'φ_k ~ Dirichlet(η)\t\t(4)', 'style': 'Normal', 'align': None, 'img': None},
 {'type': 'p',
  'text': 'where θd is the topic-proportion vector of document d, zd,n is the latent topic of its n-th word '
          'wd,n, and φk is the word distribution of topic k. Exact posterior inference is intractable and '
          'was approximated with online variational Bayes [27] as implemented in Gensim’s LdaModel [23]. LDA '
          'was chosen because titles and abstracts are short, heterogeneous texts for which a probabilistic '
          'mixture is more stable than hard clustering; because the aim is to describe the thematic '
          'structure of the field rather than to classify documents into predefined categories; because the '
          'document–topic matrix (Θ) feeds the downstream HJ-Biplot and LDAvis analyses; and because ethics '
          'scholarship is interdisciplinary, so that one paper may combine governance, bias, privacy, and '
          'technical evaluation, which favors a mixed-membership representation.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '2.5. Selection of the Number of Topics and Priors',
  'style': 'Heading 2',
  'align': None,
  'img': None},
 {'type': 'p',
  'text': 'Two coherence metrics were computed on the top-10 words of each topic: Cv, which combines a '
          'sliding window, indirect cosine similarity, and normalized pointwise mutual information [28], and '
          'NPMI [29], defined for a word pair as',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'NPMI(w_i, w_j) = log[P(w_i,w_j)/(P(w_i)P(w_j))] − log P(w_i,w_j)\t\t(5)',
  'style': 'Normal',
  'align': None,
  'img': None},
 {'type': 'p',
  'text': 'and bounded in [−1, 1]. Automatic coherence measures were designed to approximate human judgments '
          'of topic quality [30], although human interpretability remains the ultimate criterion [14]. Four '
          'structural diagnostics were also recorded: topic diversity (the proportion of unique words among '
          'the top-10 words of all topics), perplexity, dominance (the share of documents whose dominant '
          'topic is the most frequent one), and the normalized entropy of the topic-size distribution,',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'H̃ = −Σ_{k=1}^{K} p_k log p_k / log K\t\t(6)',
  'style': 'Normal',
  'align': None,
  'img': None},
 {'type': 'p',
  'text': 'where pk is the share of documents whose dominant topic is k. The number of topics was evaluated '
          'for K = 4–40 (37 values) with a symmetric α and 30 passes. The operational rule was to select the '
          'smallest K whose Cv lies within 1% of the grid maximum, provided that diversity was high, '
          'dominance remained below an excessive-dominance threshold of 40%, no topic was empty, and the '
          'neighbouring values of K were stable (Section 2.7). For the selected K, a grid of 16 prior '
          'combinations was evaluated: α ∈ {0.01, 0.1, symmetric, asymmetric} and η ∈ {0.01, 0.05, 0.5, '
          'symmetric}. The configuration with the highest Cv was retained, and asymmetric document–topic '
          'priors were included because they can improve topic models [13].',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p', 'text': '2.6. Final Model Configuration', 'style': 'Heading 2', 'align': None, 'img': None},
 {'type': 'p',
  'text': 'The final model was trained with the configuration in Table 3, using a single-process CPU '
          'implementation and a fixed random seed for reproducibility.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'Table 3. Final LDA configuration.',
  'style': 'Normal',
  'align': 'LEFT (0)',
  'img': None},
 {'type': 'tbl',
  'rows': [['Parameter', 'Value'],
           ['Number of topics (K)', '27'],
           ['α (document–topic prior)', 'asymmetric'],
           ['η (topic–word prior)', '0.5'],
           ['Passes', '50'],
           ['Iterations per document', '50'],
           ['Random seed', '42'],
           ['Documents / vocabulary', '19,505 / 4,533'],
           ['Implementation', 'Gensim 4.4.0, LdaModel']]},
 {'type': 'p', 'text': '', 'style': 'Normal', 'align': None, 'img': None},
 {'type': 'p',
  'text': '2.7. Sensitivity and Stability Analysis',
  'style': 'Heading 2',
  'align': None,
  'img': None},
 {'type': 'p',
  'text': 'Three sources of variation were examined, following the stability literature [12,31,32]. (a) '
          'Random seed: five seeds (42–46) were trained for each K in the neighbourhood 25–29, with 10 '
          'passes per run. (b) Neighbouring K: every model was compared with the reference model (K = 27, '
          'seed 42). (c) Priors: the 16-combination α/η grid of Section 2.5. Topics from two runs were '
          'aligned with the Hungarian algorithm [33] applied to the cosine similarity of their topic–word '
          'distributions. The overlap of aligned topics was measured with the Jaccard index of their top-10 '
          'word sets,',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p', 'text': 'J(A,B) = |A ∩ B| / |A ∪ B|\t\t(7)', 'style': 'Normal', 'align': None, 'img': None},
 {'type': 'p',
  'text': 'and the agreement between dominant-topic assignments of the same documents with the Adjusted Rand '
          'Index (ARI) [34]. For the neighbouring-K comparison, each reference topic was matched to its best '
          'Jaccard partner, and a reference topic was counted as covered when that Jaccard was at least 0.5.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p', 'text': '2.8. HJ-Biplot and LDAvis', 'style': 'Heading 2', 'align': None, 'img': None},
 {'type': 'p',
  'text': 'The HJ-Biplot [35], an extension of the classical biplot [36], represents rows and columns of a '
          'matrix simultaneously in a low-dimensional space. For a column-centered matrix Xc after centered '
          'log-ratio (CLR) transformation of the topic proportions, the singular value decomposition is',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p', 'text': 'X_c = U · D · Vᵀ\t\t(8)', 'style': 'Normal', 'align': None, 'img': None},
 {'type': 'p',
  'text': 'with row coordinates A = UD and column coordinates B = VD. The representation quality of row i on '
          'the first two axes is the squared cosine,',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'cos²_i = (a_{i1}² + a_{i2}²) / Σ_k a_{ik}²\t\t(9)',
  'style': 'Normal',
  'align': None,
  'img': None},
 {'type': 'p',
  'text': 'computed analogously for columns; the global quality is the mean squared cosine. Three biplots '
          'were built: years × topics (years with at least 20 documents; each cell is the CLR-transformed '
          'mean topic proportion), countries × topics (the 12 most productive countries), and documents × '
          'topics (a random sample of 1,500 documents colored by dominant topic).',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'In LDAvis [37], topics are projected onto an intertopic distance map in which bubble area is '
          'proportional to topic prevalence, and terms are ranked by relevance,',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'r(w|t) = λ·log p(w|t) + (1−λ)·log[p(w|t)/p(w)]\t\t(10)',
  'style': 'Normal',
  'align': None,
  'img': None},
 {'type': 'p',
  'text': 'where λ ∈ [0, 1] trades off within-topic frequency against exclusivity. Relevance values are '
          'reported on the logarithmic scale; λ = 0.6 was used for exploration, and rankings for λ = 0 and λ '
          '= 1 were also exported.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '2.9. Software, Reproducibility, and Use of Generative AI',
  'style': 'Heading 2',
  'align': None,
  'img': None},
 {'type': 'p',
  'text': 'The analysis was organized as scripted pipeline phases (corpus audit, K diagnostic, K decision, '
          'final model, trend analysis, sensitivity analysis, and reporting) executed with a global seed of '
          '42 in an environment reproduced with uv sync. All figures are direct outputs of the pipeline, '
          'generated from the corpus data with Matplotlib [38]: Figures 1, 2, and 5–8 by phase6_report.py '
          'and Figures 3 and 4 by phase5_sensitivity.py. Tables are exported as CSV files, and the full '
          'configuration, library versions, and metrics are stored in a single JSON file. The code is '
          'available in the project repository [39]. Generative AI tools were used only to draft and format '
          'the manuscript text from the group’s methodology document, as declared in the Acknowledgments; '
          'they were not used to generate results or figures.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p', 'text': '3. Results', 'style': 'Heading 1', 'align': None, 'img': None},
 {'type': 'p',
  'text': '3.1. Bibliometric Profile of the Corpus',
  'style': 'Heading 2',
  'align': None,
  'img': None},
 {'type': 'p',
  'text': 'The modeled corpus is very recent (Table 4). Only four documents date from 2018 to 2021, and 2022 '
          'contributes 31. Output then rises from 556 documents in 2023 to 2,763 in 2024, 6,883 in 2025, and '
          '9,268 in 2026, so that 82.8% of the corpus was published in the last two years; the 2026 count is '
          'a partial-year figure.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'Table 4. Documents per year (* 2026 is incomplete).',
  'style': 'Normal',
  'align': 'LEFT (0)',
  'img': None},
 {'type': 'tbl',
  'rows': [['Year', 'Documents', '%'],
           ['2018', '2', '0.01'],
           ['2019', '1', '0.01'],
           ['2021', '1', '0.01'],
           ['2022', '31', '0.16'],
           ['2023', '556', '2.85'],
           ['2024', '2,763', '14.17'],
           ['2025', '6,883', '35.29'],
           ['2026*', '9,268', '47.52']]},
 {'type': 'p', 'text': '', 'style': 'Normal', 'align': None, 'img': None},
 {'type': 'p',
  'text': 'The United States (23.71%) and China (20.32%) together account for 44.0% of the corpus, followed '
          'by India, Germany, and the United Kingdom (Table 5). Articles (42.65%) and conference papers '
          '(41.07%) dominate the document types (Table 6), and reviews account for 9.26%. The most frequent '
          'single source is the proceedings of the Annual Meeting of the Association for Computational '
          'Linguistics (542 documents; 2.78%), followed by Lecture Notes in Computer Science, the CEUR '
          'Workshop Proceedings, and the ACM Conference on Human Factors in Computing Systems (Table 7). The '
          'most productive institutions are the University of California system (354 documents), Stanford '
          'University (326), Tsinghua University (322), Carnegie Mellon University (255), and Zhejiang '
          'University (249).',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'Table 5. Ten most productive countries.',
  'style': 'Normal',
  'align': 'LEFT (0)',
  'img': None},
 {'type': 'tbl',
  'rows': [['Country', 'Documents', '%'],
           ['United States', '4,538', '23.71'],
           ['China', '3,888', '20.32'],
           ['India', '1,022', '5.34'],
           ['Germany', '999', '5.22'],
           ['United Kingdom', '891', '4.66'],
           ['Italy', '617', '3.22'],
           ['Canada', '500', '2.61'],
           ['Australia', '487', '2.54'],
           ['South Korea', '449', '2.35'],
           ['Spain', '316', '1.65']]},
 {'type': 'p', 'text': '', 'style': 'Normal', 'align': None, 'img': None},
 {'type': 'p', 'text': 'Table 6. Main document types.', 'style': 'Normal', 'align': 'LEFT (0)', 'img': None},
 {'type': 'tbl',
  'rows': [['Document type', 'Documents', '%'],
           ['Article', '8,319', '42.65'],
           ['Conference paper', '8,011', '41.07'],
           ['Review', '1,806', '9.26'],
           ['Book chapter', '838', '4.30'],
           ['Note', '164', '0.84'],
           ['Book', '127', '0.65']]},
 {'type': 'p', 'text': '', 'style': 'Normal', 'align': None, 'img': None},
 {'type': 'p',
  'text': 'Table 7. Eight most frequent sources.',
  'style': 'Normal',
  'align': 'LEFT (0)',
  'img': None},
 {'type': 'tbl',
  'rows': [['Source', 'Documents', '%'],
           ['Proceedings of the Annual Meeting of the Association for Computational Linguistics',
            '542',
            '2.78'],
           ['Lecture Notes in Computer Science', '365', '1.87'],
           ['CEUR Workshop Proceedings', '281', '1.44'],
           ['Conference on Human Factors in Computing Systems', '278', '1.43'],
           ['IEEE Access', '260', '1.33'],
           ['Advances in Neural Information Processing Systems', '232', '1.19'],
           ['Communications in Computer and Information Science', '218', '1.12'],
           ['Proceedings of Machine Learning Research', '179', '0.92']]},
 {'type': 'p', 'text': '', 'style': 'Normal', 'align': None, 'img': None},
 {'type': 'p',
  'text': '3.2. Model Selection and Final Model',
  'style': 'Heading 2',
  'align': None,
  'img': None},
 {'type': 'p',
  'text': 'Coherence rises quickly up to about K = 11–14 and then plateaus (Figure 2, Table 8; the complete '
          'grid is in Appendix A, Table A1). The global maximum occurs at K = 40 (Cv = 0.5696), but the '
          'advantage over K = 27 (Cv = 0.5640; NPMI = 0.0749) is only 0.0056, and K = 27 is the smallest K '
          'within 1% of that maximum. It also shows high top-10 diversity (0.7889), a dominance of 10.6% '
          '(well below the 40% threshold), no empty topics, and stable neighbours (Section 3.4). Perplexity '
          'varied little across the grid (from −7.41 to −7.37). K = 27 was therefore selected.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p', 'text': '', 'style': 'Normal', 'align': 'CENTER (1)', 'img': 'LDA_K_Coherence.png'},
 {'type': 'p',
  'text': 'Figure 2. Cv and NPMI coherence as a function of the number of topics K (symmetric α, 30 passes). '
          'Generated by phase6_report.py.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'Table 8. Selected rows of the K search (symmetric α, 30 passes; the selected K is in bold).',
  'style': 'Normal',
  'align': 'LEFT (0)',
  'img': None},
 {'type': 'tbl',
  'rows': [['K', 'Cv', 'NPMI', 'Diversity', 'Dominance'],
           ['4', '0.4826', '0.0377', '0.8750', '0.2740'],
           ['8', '0.4806', '0.0347', '0.7125', '0.1863'],
           ['12', '0.5189', '0.0502', '0.7750', '0.1518'],
           ['16', '0.5350', '0.0611', '0.7688', '0.1148'],
           ['20', '0.5471', '0.0655', '0.7450', '0.1346'],
           ['24', '0.5583', '0.0696', '0.7792', '0.1160'],
           ['27', '0.5640', '0.0749', '0.7889', '0.1056'],
           ['30', '0.5449', '0.0622', '0.7867', '0.0854'],
           ['34', '0.5598', '0.0672', '0.7647', '0.0783'],
           ['38', '0.5659', '0.0758', '0.7421', '0.0748'],
           ['40', '0.5696', '0.0781', '0.7825', '0.0928']]},
 {'type': 'p', 'text': '', 'style': 'Normal', 'align': None, 'img': None},
 {'type': 'p',
  'text': 'The α/η grid (Table 9) gave the best coherence for α = asymmetric and η = 0.5 (Cv = 0.5861), and '
          'the Cv range across all 16 combinations was only 0.0288. The final model (Table 10) reaches Cv = '
          '0.5761 and NPMI = +0.0818 with no empty topics. Its Cv differs from the grid value for K = 27 '
          '(0.5640) because the grid used a symmetric α and 30 passes, so comparisons are valid only within '
          'one configuration.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'Table 9. The eight best α/η combinations at K = 27.',
  'style': 'Normal',
  'align': 'LEFT (0)',
  'img': None},
 {'type': 'tbl',
  'rows': [['α', 'η', 'Cv', 'NPMI', 'Dominance'],
           ['asymmetric', '0.5', '0.5861', '0.0861', '0.0890'],
           ['0.01', '0.5', '0.5779', '0.0812', '0.0771'],
           ['symmetric', '0.5', '0.5748', '0.0808', '0.0789'],
           ['0.1', '0.5', '0.5713', '0.0793', '0.0815'],
           ['asymmetric', '0.01', '0.5657', '0.0776', '0.1273'],
           ['asymmetric', '0.05', '0.5656', '0.0765', '0.1268'],
           ['0.01', '0.05', '0.5649', '0.0748', '0.1023'],
           ['symmetric', 'symmetric', '0.5640', '0.0749', '0.1056']]},
 {'type': 'p', 'text': '', 'style': 'Normal', 'align': None, 'img': None},
 {'type': 'p',
  'text': 'Table 10. Metrics of the final model.',
  'style': 'Normal',
  'align': 'LEFT (0)',
  'img': None},
 {'type': 'tbl',
  'rows': [['Metric', 'Value', 'Metric', 'Value'],
           ['Cv', '0.5761', 'Dominance', '0.0844'],
           ['NPMI', '+0.0818', 'Normalized entropy', '0.9475'],
           ['Diversity (top-10)', '0.7741', 'Empty topics', '0'],
           ['Perplexity', '−7.2693', '', '']]},
 {'type': 'p', 'text': '', 'style': 'Normal', 'align': None, 'img': None},
 {'type': 'p', 'text': '3.3. Topic Structure', 'style': 'Heading 2', 'align': None, 'img': None},
 {'type': 'p',
  'text': 'Table 11 lists the 27 topics with their proposed labels, document counts, top terms, and leading '
          'country. The largest topics by number of documents are T3 (generative AI ethics, 1,647 documents; '
          '8.4% of the corpus), T23 (application challenges and reviews, 1,498), T20 (retrieval and '
          'knowledge graphs, 1,277), T1 (human problem-solving, 1,168), and T14 (training and unlearning, '
          '1,098); the smallest are T16 (public policy, 111), T24 (language and culture, 126), T4 (privacy '
          'and explainability, 181), T12 (multimodal models, 182), and T21 (networks and robotics, 223). The '
          'United States leads 19 of the 27 topics by document volume and China leads the remaining eight '
          '(T10, T12, T14, T15, T20, T21, T23, and T24). Topic labels were assigned by the authors from the '
          'top terms and, where needed, from the most representative documents.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'Table 11. The 27 LDA topics: proposed label, number of documents with the topic as dominant, top '
          'ten terms by frequency, and the country contributing the most documents.',
  'style': 'Normal',
  'align': 'LEFT (0)',
  'img': None},
 {'type': 'tbl',
  'rows': [['ID', 'Proposed label', 'Docs', 'Top terms', 'Leading country'],
           ['1',
            'Human problem-solving and AI-assisted work',
            '1,168',
            'work, make, system, ai, problem, one, human, task, find, question',
            'United States'],
           ['2',
            'AI governance and framework design',
            '1,064',
            'ai, framework, governance, system, design, across, structure, rather, evidence, support',
            'United States'],
           ['3',
            'Generative AI ethics, technology, and societal challenges',
            '1,647',
            'ai, generative_ai, technology, artificial_intelligence, ethical, genai, challenge, development, '
            'tool, generative_artificial',
            'United States'],
           ['4',
            'Privacy, explainability, and financial trust',
            '181',
            'privacy, explanation, financial, audit, framework, transparency, explainability, explainable, '
            'trust, sensitive',
            'United States'],
           ['5',
            'ChatGPT response quality and performance evaluation',
            '996',
            'response, chatgpt, question, accuracy, score, performance, compare, test, quality, gpt-4',
            'United States'],
           ['6',
            'Perception, trust, and AI adoption surveys',
            '437',
            'participant, ai, survey, perceive, perception, finding, group, influence, among, impact',
            'United States'],
           ['7',
            'Agentic systems, architecture, and enterprise workflows',
            '686',
            'agent, system, architecture, framework, agentic, multi-agent, agentic_ai, ai, rag, workflow',
            'United States'],
           ['8',
            'Bias, fairness, gender, and stereotype in prompts',
            '644',
            'bias, fairness, gender, prompt, social, stereotype, across, demographic, output, group',
            'United States'],
           ['9',
            'Clinical, medical, and healthcare AI',
            '881',
            'clinical, medical, patient, healthcare, health, care, support, medicine, include, clinician',
            'United States'],
           ['10',
            'Prediction, foundation models, and biological sequence learning',
            '503',
            'prediction, foundation_model, machine_learn, ml, biological, feature, learn, predict, sequence, '
            'representation',
            'China'],
           ['11',
            'Safety, cybersecurity, and adversarial robustness',
            '512',
            'safety, risk, security, attack, threat, vulnerability, cybersecurity, adversarial, detection, '
            'defense',
            'United States'],
           ['12',
            'Multimodal and vision-language models',
            '182',
            'multimodal, image, visual, video, modality, vision, mllms, generation, foundation_model, vlms',
            'China'],
           ['13',
            'Evaluation, benchmarks, and performance measurement',
            '784',
            'evaluation, benchmark, across, score, test, metric, prompt, assessment, performance, framework',
            'United States'],
           ['14',
            'Training, fine-tuning, and machine unlearning',
            '1,098',
            'train, fine-tune, performance, task, unlearn, inference, optimization, introduce, framework, '
            'learn',
            'China'],
           ['15',
            'Intelligent management, markets, and energy systems',
            '700',
            'management, system, framework, market, energy, intelligent, integrate, power, process, '
            'technology',
            'China'],
           ['16',
            'Public policy, digital services, and urban platforms',
            '111',
            'policy, service, public, information, urban, platform, web, digital, government, access',
            'United States'],
           ['17',
            'User interaction, chatbots, and conversational support',
            '768',
            'user, design, interaction, system, support, chatbots, chatbot, mental_health, conversational, '
            'dialogue',
            'United States'],
           ['18',
            'Human cognition, emotion, and social simulation',
            '551',
            'human, social, cognitive, emotional, emotion, agent, theory, narrative, behavior, interaction',
            'United States'],
           ['19',
            'Text detection, datasets, and classification',
            '1,081',
            'text, detection, dataset, classification, performance, prompt, fine-tune, generate, task, '
            'datasets',
            'United States'],
           ['20',
            'Retrieval, knowledge graphs, and semantic generation',
            '1,277',
            'framework, reason, knowledge, semantic, retrieval, rag, structure, system, generation, query',
            'China'],
           ['21',
            'Networks, robotics, simulation, and digital twins',
            '223',
            'plan, system, network, environment, physical, robot, simulation, control, scenario, traffic',
            'China'],
           ['22',
            'Alignment, values, and moral decision-making',
            '244',
            'human, alignment, value, moral, preference, ethical, align, decision, decision-making, reason',
            'United States'],
           ['23',
            'Application challenges, reviews, and field advances',
            '1,498',
            'application, challenge, review, development, include, field, advance, capability, potential, '
            'survey',
            'China'],
           ['24',
            'Language, culture, translation, and multilingual corpora',
            '126',
            'language, cultural, translation, english, linguistic, multilingual, write, chinese, text, '
            'context',
            'China'],
           ['25',
            'ChatGPT in academic and systematic literature reviews',
            '499',
            'chatgpt, review, include, publish, tool, identify, literature, scientific, systematic_review, '
            'qualitative',
            'United States'],
           ['26',
            'Law, regulation, compliance, and software requirements',
            '732',
            'legal, process, code, requirement, automate, software, regulatory, regulation, compliance, case',
            'United States'],
           ['27',
            'AI in education and student learning',
            '912',
            'student, learn, education, educational, feedback, tool, skill, assessment, chatgpt, teach',
            'United States']]},
 {'type': 'p', 'text': '', 'style': 'Normal', 'align': None, 'img': None},
 {'type': 'p',
  'text': 'For interpretation, the topics were grouped into five thematic blocks: risk, security, and policy '
          '(T4, T11, T16, T22, T26); governance and ethics (T2, T3, T6, T8, T24); applications and '
          'evaluation (T9, T13, T14, T15, T23, T27); interaction and agents (T7, T17, T18, T20, T21); and a '
          'technical core (T1, T5, T10, T12, T19, T25). Summing dominant-topic counts per block (Table 12) '
          'shows that applications and evaluation form the largest block (30.1%), followed by the technical '
          'core (22.7%) and governance and ethics (20.1%).',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'Table 12. Thematic blocks, derived by summing dominant-topic counts.',
  'style': 'Normal',
  'align': 'LEFT (0)',
  'img': None},
 {'type': 'tbl',
  'rows': [['Block', 'Topics', 'Documents', '%'],
           ['Risk, security, and policy', 'T4, T11, T16, T22, T26', '1,780', '9.1'],
           ['Governance and ethics', 'T2, T3, T6, T8, T24', '3,918', '20.1'],
           ['Applications and evaluation', 'T9, T13, T14, T15, T23, T27', '5,873', '30.1'],
           ['Interaction and agents', 'T7, T17, T18, T20, T21', '3,505', '18.0'],
           ['Technical core', 'T1, T5, T10, T12, T19, T25', '4,429', '22.7']]},
 {'type': 'p', 'text': '', 'style': 'Normal', 'align': None, 'img': None},
 {'type': 'p', 'text': '3.4. Sensitivity and Robustness', 'style': 'Heading 2', 'align': None, 'img': None},
 {'type': 'p',
  'text': 'For the selected K = 27, the mean Jaccard overlap between aligned topics across seed pairs is '
          '0.3192 (SD = 0.0242; minimum 0.2756), and the mean ARI is 0.2413 (SD = 0.0280; minimum 0.1985) '
          '(Table 13, Figure 3). Neighbouring values of K behave similarly, so K = 27 is neither notably '
          'more nor less stable than its neighbours.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'Table 13. Seed stability by K, mean (SD) over seed pairs (10 passes per run).',
  'style': 'Normal',
  'align': 'LEFT (0)',
  'img': None},
 {'type': 'tbl',
  'rows': [['K', 'Jaccard', 'ARI', 'Cv'],
           ['25', '0.346 (0.034)', '0.246 (0.027)', '0.546 (0.010)'],
           ['26', '0.310 (0.039)', '0.246 (0.033)', '0.546 (0.012)'],
           ['27', '0.319 (0.024)', '0.241 (0.028)', '0.551 (0.010)'],
           ['28', '0.294 (0.036)', '0.223 (0.028)', '0.543 (0.007)'],
           ['29', '0.305 (0.026)', '0.259 (0.033)', '0.550 (0.010)']]},
 {'type': 'p', 'text': '', 'style': 'Normal', 'align': None, 'img': None},
 {'type': 'p', 'text': '', 'style': 'Normal', 'align': 'CENTER (1)', 'img': 'LDA_Sensitivity_Stability.png'},
 {'type': 'p',
  'text': 'Figure 3. Stability of the LDA solution across random seeds for K = 25–29. Generated by '
          'phase5_sensitivity.py.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'Compared with the reference model, neighbouring models recover only part of the topic set at a '
          'Jaccard threshold of 0.5: 11 of 27 reference topics for K = 25, 8 for K = 26, 10 for K = 28, and '
          '11 for K = 29 (Table 14, Figure 4). Topics therefore split and merge as K changes, although '
          'several persist across models (a best-match Jaccard of 1.0 is reached for K = 25, 26, and 28).',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'Table 14. Best-match Jaccard against the reference model (K = 27, seed 42).',
  'style': 'Normal',
  'align': 'LEFT (0)',
  'img': None},
 {'type': 'tbl',
  'rows': [['K', 'Mean', 'Median', 'Reference topics covered (≥ 0.5)'],
           ['25', '0.4412', '0.4286', '11 / 27'],
           ['26', '0.4048', '0.3810', '8 / 27'],
           ['27', '1.0000', '1.0000', '27 / 27'],
           ['28', '0.3844', '0.4286', '10 / 27'],
           ['29', '0.3715', '0.3333', '11 / 27']]},
 {'type': 'p', 'text': '', 'style': 'Normal', 'align': None, 'img': None},
 {'type': 'p', 'text': '', 'style': 'Normal', 'align': 'CENTER (1)', 'img': 'LDAvis_Intertopic_Map.png'},
 {'type': 'p',
  'text': 'Figure 4. Overlap between neighbouring-K models and the reference model (K = 27). Generated by '
          'phase5_sensitivity.py.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'The coherence differences across the 16 prior combinations are small (range 0.0288; Table 9), and '
          'the selected configuration is the best of the grid. Overall, seed stability is moderate. The '
          'highest-volume topics (T3, T23, T20, and T1) are recovered most consistently, whereas the '
          'smallest topic (T16) is the most sensitive.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p', 'text': '3.5. HJ-Biplot Analysis', 'style': 'Heading 2', 'align': None, 'img': None},
 {'type': 'p',
  'text': 'Table 15 reports the quality of the three biplots after the centered log-ratio transformation. '
          'The years × topics biplot is well represented (the first two axes retain 97.5% of the variance; '
          'mean squared cosines of 0.93 for rows and 0.95 for columns), the countries × topics biplot is '
          'moderately represented (59.4% of the variance; mean squared cosines of 0.43 and 0.47), and the '
          'documents × topics biplot is poorly represented (about 16.8%; mean squared cosines of 0.17 and '
          '0.13). Table 16 summarizes the interpretation of the axes of the first two biplots from their row '
          'and column coordinates.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p', 'text': 'Table 15. HJ-Biplot quality.', 'style': 'Normal', 'align': 'LEFT (0)', 'img': None},
 {'type': 'tbl',
  'rows': [['Biplot', 'Rows × columns', 'Axes 1 / 2 (% variance)', 'Mean cos² rows', 'Mean cos² columns'],
           ['Years × topics', '5 × 27', '74.7 / 22.8', '0.931', '0.952'],
           ['Countries × topics', '12 × 27', '32.8 / 26.7', '0.434', '0.468'],
           ['Documents (sample) × topics', '1,500 × 27', '9.2 / 7.7', '0.169', '0.133']]},
 {'type': 'p', 'text': '', 'style': 'Normal', 'align': None, 'img': None},
 {'type': 'p',
  'text': 'Table 16. Interpretation of the HJ-Biplot axes (poles listed by decreasing coordinate magnitude).',
  'style': 'Normal',
  'align': 'LEFT (0)',
  'img': None},
 {'type': 'tbl',
  'rows': [['Biplot, axis', 'Negative pole', 'Positive pole'],
           ['Years, axis 1 (74.7%)', '2022; T1, T24, T22, T8, T14', '2026, 2025; T5, T4, T9, T7, T20, T21'],
           ['Years, axis 2 (22.8%)',
            '2023, 2024; T1, T25, T5, T23, T3, T27',
            '2026, 2022; T7, T2, T13, T15, T21, T20'],
           ['Countries, axis 1 (32.8%)',
            'UK, Australia, Germany, Italy, Canada, Spain, United States; T3, T1, T2, T26',
            'China; T20, T14, T12, T15, T10, T21'],
           ['Countries, axis 2 (26.7%)', 'United States, India; T1, T3, T17, T4', 'Turkey; T5, T9, T13']]},
 {'type': 'p', 'text': '', 'style': 'Normal', 'align': None, 'img': None},
 {'type': 'p',
  'text': 'In the CLR-transformed years biplot, axis 1 is a temporal gradient that orders the years '
          'chronologically from 2022 (−4.79) to 2026 (+1.93). The early pole is associated with T1 (human '
          'problem-solving), multilingual corpora (T24), value alignment (T22), bias (T8) and training '
          '(T14), whereas the recent pole is dominated by ChatGPT response evaluation (T5, the longest '
          'vector), privacy (T4), clinical applications (T9), agentic systems (T7), retrieval and knowledge '
          'graphs (T20) and robotics/digital twins (T21). Axis 2 separates 2026 (+2.02) from 2023 (−1.85) '
          'and 2024 (−1.06), and is associated with agentic systems (T7), governance frameworks (T2) and '
          'evaluation (T13) at its positive pole, and with human problem-solving (T1), literature reviews '
          '(T25) and education (T27) at its negative pole. The 2022 point rests on only 31 documents; its '
          'extreme position should therefore be interpreted with caution.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p', 'text': '', 'style': 'Normal', 'align': 'CENTER (1)', 'img': 'LDA_HJBiplot_Years.png'},
 {'type': 'p',
  'text': 'Figure 5. HJ-Biplot of years × topics (CLR-transformed topic proportions, column-centered). '
          'Generated by phase6_report.py.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'In the CLR-transformed countries biplot, axis 1 is dominated by China (+2.44; squared cosine '
          '0.87), aligned with retrieval and knowledge graphs (T20), training (T14) and multimodal models '
          '(T12); South Korea (+1.16) and Japan (+0.85) lie on the same side. The negative pole gathers most '
          'Western countries (United Kingdom, Australia, Germany, Italy, Canada, Spain and the United '
          'States), aligned with generative AI ethics (T3), human problem-solving (T1), governance (T2) and '
          'law (T26). Axis 2 is dominated by Turkey (+2.57; squared cosine 0.97), aligned with ChatGPT '
          'response evaluation (T5), clinical applications (T9) and evaluation (T13), whereas China, South '
          'Korea and Japan lie near the centre of this axis. Most countries other than China and Turkey lie '
          'close to the origin with squared cosines below 0.6 (for example, the United States 0.18, India '
          '0.15 and Spain 0.26), so their relative positions should not be interpreted.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p', 'text': '', 'style': 'Normal', 'align': 'CENTER (1)', 'img': 'LDA_HJBiplot_Countries.png'},
 {'type': 'p',
  'text': 'Figure 6. HJ-Biplot of countries × topics (12 most productive countries). Generated by '
          'phase6_report.py.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'The documents biplot (Figure 7) suggests a separation between technical-evaluation documents (T5, '
          'T13, T19) and ethics and governance documents (T3, T8, T11), with an overlap zone around '
          'interaction and agents (T7, T17). Given its low representation quality, this reading is '
          'indicative only.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p', 'text': '', 'style': 'Normal', 'align': 'CENTER (1)', 'img': 'LDA_HJBiplot_Documents.png'},
 {'type': 'p',
  'text': 'Figure 7. HJ-Biplot of a random sample of 1,500 documents colored by dominant topic. Generated by '
          'phase6_report.py.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p', 'text': '3.6. LDAvis Analysis', 'style': 'Heading 2', 'align': None, 'img': None},
 {'type': 'p',
  'text': 'On the intertopic distance map (Figure 8), the generative AI (T3), evaluation (T13), and training '
          '(T14) topics lie close together because they share technical vocabulary, whereas the ethics (T8) '
          'and governance (T2) topics separate from the technical core. Small topics (T16 and T24) appear '
          'isolated, consistent with their low volume, and T3 has the largest bubble.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p', 'text': '', 'style': 'Normal', 'align': 'CENTER (1)', 'img': 'LDAvis_Relevance_Heatmap.png'},
 {'type': 'p',
  'text': 'Figure 8. LDAvis intertopic distance map (bubble size proportional to topic prevalence). '
          'Generated by phase6_report.py.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'With λ = 0.6, the most relevant terms combine frequency and exclusivity. T11 is characterized by '
          'safety, security, threat, and attack; T9 by clinical, medical, and patient; T20 by semantic, '
          'reason, knowledge, and retrieval; and T8 by bias and fairness. T24 is the only topic whose '
          'leading term (language) has a positive relevance score. By contrast, the leading terms of T1 '
          '(make, work, problem) and T23 (application, challenge, review) are generic, indicating low '
          'specificity.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p', 'text': '3.7. Integrative Synthesis', 'style': 'Heading 2', 'align': None, 'img': None},
 {'type': 'p',
  'text': 'The two visualizations answer different questions. The HJ-Biplot has a multivariate foundation: '
          'it shows where years, countries, or documents lie relative to topics and, through the squared '
          'cosines, how reliable each position is. LDAvis supports the exploration of why a topic is '
          'interpreted as it is, through the intertopic map, topic prevalence, and relevance-ranked terms.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'The three sources of evidence converge. The bibliometric profile shows a corpus dominated by '
          '2025–2026 and by United States and Chinese output, made up mostly of articles and conference '
          'papers, with computational linguistics and human–computer interaction venues among the leading '
          'sources. The LDA solution is consistent with that profile: applications and evaluation and the '
          'technical core together account for 52.8% of the documents (Table 12). The years biplot, the best '
          'represented of the three, places recent years close to retrieval, evaluation, agent, and '
          'governance-framework topics, and the LDAvis map shows technical topics sharing a vocabulary-dense '
          'region while ethics and governance topics are more separated. Together they support a structure '
          'with a large technical layer and a smaller, more distinct normative layer, with the strongest '
          'evidence for the high-volume topics.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p', 'text': '4. Discussion', 'style': 'Heading 1', 'align': None, 'img': None},
 {'type': 'p', 'text': '4.1. Principal Findings', 'style': 'Heading 2', 'align': None, 'img': None},
 {'type': 'p',
  'text': 'The 27-topic solution indicates that the literature linking generative AI and LLMs with '
          'governance and ethics is a composite of two layers rather than a homogeneous “AI ethics” field. A '
          'technical layer—benchmarking (T13), training and unlearning (T14), retrieval and knowledge graphs '
          '(T20), multimodality (T12), and agentic architectures (T7)—shares a large, overlapping '
          'vocabulary. A normative layer—governance (T2), bias and fairness (T8), safety and security (T11), '
          'alignment and moral decision-making (T22), and law and compliance (T26)—occupies more clearly '
          'separated regions of the topic space. The convergence of the biplot and LDAvis readings '
          'strengthens this interpretation.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'The block shares qualify the field’s self-image: although the corpus was retrieved with '
          'governance- and ethics-related terms, applications and evaluation (30.1%) and the technical core '
          '(22.7%) together outweigh governance and ethics (20.1%) and risk, security, and policy (9.1%). '
          'This may reflect the breadth of the model-related terms in the search string, but we did not test '
          'this explanation. Domain applications form sizeable, distinct topics, notably education (T27; '
          '4.7% of documents) and health (T9; 4.5%). Temporally, the biplot shows a shift from early topics '
          'centered on human problem-solving, detection, and bias toward retrieval, evaluation, agents, and '
          'governance frameworks, although the 2022 point rests on 31 documents. Geographically, China is '
          'associated with model-building topics (retrieval, training, intelligent management), whereas '
          'Western countries are associated with ethics-centered and governance topics; leadership by volume '
          'largely mirrors total output, however, and is not evidence of specialization.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '4.2. Comparison with Previous Literature',
  'style': 'Heading 2',
  'align': None,
  'img': None},
 {'type': 'p',
  'text': 'Previous mappings of AI ethics were based on guidelines and policy documents. Jobin et al. [1] '
          'reported convergence around transparency, justice and fairness, non-maleficence, responsibility, '
          'and privacy. The normative topics we recovered from the scholarly literature correspond to '
          'several of these principles—privacy and explainability (T4), bias and fairness (T8), safety and '
          'security (T11), and governance (T2)—which suggests that the principle-oriented vocabulary has '
          'been taken up by research on generative AI. Hagendorff [3] questioned the practical effectiveness '
          'of guidelines, and Mittelstadt [4] argued that principles alone cannot guarantee ethical AI '
          'because translating them into practice remains difficult. Our finding that technical evaluation, '
          'training, and application topics outweigh the normative blocks is compatible with the concern '
          'that ethics discourse is largely absorbed by technical questions, although our design, which is '
          'descriptive and based on titles and abstracts, cannot test that claim.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'The closest methodological precedent is the LDA analysis of 57 national policy documents by Saheb '
          'and Saheb [7], whose themes (principles, personal data protection, governance and monitoring, '
          'procedural, and epistemological issues) are normative by construction. In our scholarly corpus, '
          'topics for privacy, governance, bias, and explainability coexist with large technical clusters, '
          'which is the expected difference between a research corpus and a policy corpus. The comparison '
          'with Binh [8], who applied LDA to 465 AI-auditing papers, is similar: technology-centred topics '
          'accounted for most of the content and ethics-related topics for a minority. Because corpora, '
          'queries, and units of measurement (tokens versus dominant-topic documents) differ, the magnitudes '
          'are not comparable, but the qualitative pattern is shared.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'The topics also map onto several risk areas described for language models. Bias and fairness (T8) '
          'correspond to discrimination and toxicity, privacy (T4) and security (T11) to information hazards '
          'and malicious uses, and the interaction topics (T17 and T18) to human–computer interaction harms '
          'in the taxonomy of Weidinger et al. [6]. In contrast, none of the 27 topics is characterized by '
          'terms on the environmental or labor-market consequences of generative AI, which Bender et al. [5] '
          'and Weidinger et al. [6] highlight and which Binh [8] also found among the least represented '
          'topics in AI auditing (0.4% of tokens each for environmental ethics and racial and ethnic '
          'disparities). T15 concerns energy and market applications rather than the footprint of the models '
          'themselves. This absence, which could be tested with a dedicated query, may indicate a gap in the '
          'literature or a limitation of a model with K = 27 and of title-and-abstract text.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '4.3. Methodological Considerations',
  'style': 'Heading 2',
  'align': None,
  'img': None},
 {'type': 'p',
  'text': 'The coherence curve plateaus, so the choice of K = 27 rests on parsimony and interpretability as '
          'much as on coherence: the best-scoring K (40) offers only a marginal gain. Topic identity is only '
          'moderately stable: a seed-pair Jaccard of about 0.32 and an ARI of about 0.24 mean that '
          'individual topics, especially small ones, may split or merge across runs. This agrees with '
          'reports that LDA topics vary between runs [11,12], so topic-level claims should be restricted to '
          'the higher-volume topics. The small coherence range across priors (0.0288) suggests that the '
          'structure is reasonably robust to the Dirichlet priors in this grid, even though priors can '
          'matter in general [13]. The sensitivity runs used 10 passes against 50 in the final model, so the '
          'reported stability may understate that of the final solution; we did not test this.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'The biplots were computed on column-centered mean topic probabilities. Because topic proportions '
          'are compositional data, log-ratio transformations are generally recommended for biplots [40]; the '
          'present biplots should therefore be regarded as exploratory, and repeating them after a log-ratio '
          'transformation is a priority. Coherence is not a substitute for human judgment [14], and the '
          'topic labels in Table 11 were assigned by the authors and have not been validated by independent '
          'coders. Finally, embedding-based approaches such as BERTopic [41] could be compared with LDA on '
          'the same corpus.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p', 'text': '4.4. Implications', 'style': 'Heading 2', 'align': None, 'img': None},
 {'type': 'p',
  'text': 'Theoretically, the two-layer structure suggests that studies of “AI ethics” should distinguish '
          'between the technical research on which ethical questions are built and the normative scholarship '
          'that addresses them, since aggregate counts mix the two. Practically, the absence of dedicated '
          'topics on environmental and labor-market impacts, and the small share of risk, security, and '
          'policy topics (9.1%), point to areas where funders and journals could encourage work. '
          'Methodologically, reporting stability and sensitivity alongside coherence, as done here, gives '
          'readers a way to judge which topics can be trusted.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p', 'text': '4.5. Limitations', 'style': 'Heading 2', 'align': None, 'img': None},
 {'type': 'p',
  'text': 'Single source and language. Only English-language Scopus records were used, which omits other '
          'indexes and languages.',
  'style': 'List Paragraph',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'Capped corpus. The corpus was limited to 20,000 records for reasons of computational budget and '
          'time, so the findings describe this subset and not necessarily the full literature matching the '
          'query.',
  'style': 'List Paragraph',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'Incomplete final year. The year 2026 is a partial year (query date: 11 October 2026), and '
          '2018–2022 contribute only 35 documents, so temporal readings mostly compare 2023–2026.',
  'style': 'List Paragraph',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'Duplicates. Twenty DOI duplicates and 59 title duplicates were flagged but retained.',
  'style': 'List Paragraph',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'Text representation. Only titles and abstracts were modeled; keywords were excluded.',
  'style': 'List Paragraph',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'Hard assignment. Document counts per topic use the dominant topic only and ignore mixed '
          'membership.',
  'style': 'List Paragraph',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'Biplot quality. The countries and documents biplots have moderate and low representation quality, '
          'respectively, after the centered log-ratio transformation.',
  'style': 'List Paragraph',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'Subjective labels. Topic labels and thematic blocks were assigned by the authors.',
  'style': 'List Paragraph',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p', 'text': '5. Conclusions', 'style': 'Heading 1', 'align': None, 'img': None},
 {'type': 'p',
  'text': 'This study documented a reproducible LDA pipeline applied to 19,505 Scopus documents on '
          'generative AI, LLMs, and their ethical and governance dimensions. Regarding the objectives, (i) '
          'the corpus was built, screened, and documented with a PRISMA-style flow; (ii) a 27-topic model '
          'was selected on the coherence plateau (Cv = 0.576; NPMI = 0.082); (iii) robustness was moderate '
          '(mean Jaccard of 0.319 and ARI of 0.241 across seeds, with neighbouring models recovering 8–11 of '
          'the 27 reference topics), so the solution is suitable for exploratory mapping with low-volume '
          'topics treated as provisional; and (iv) HJ-Biplot and LDAvis showed a large technical layer and a '
          'smaller, more distinct normative layer, with publication concentrated in 2025–2026 and in the '
          'United States and China.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'The main recommendations are to validate the topic labels with independent coders, to repeat the '
          'biplots with a log-ratio transformation, and to compare LDA with embedding-based topic models. '
          'Future work includes a dedicated temporal trend analysis once 2026 is complete, '
          'country-normalized thematic profiles, modeling of the full Scopus result set as computational '
          'resources allow, de-duplication, the inclusion of keyword fields, and additional bibliographic '
          'sources.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'Supplementary Materials: The code, configuration, and derived outputs (CSV tables and figures) '
          'are available in the project repository [39].',
  'style': 'Normal',
  'align': None,
  'img': None},
 {'type': 'p',
  'text': 'Author Contributions: Conceptualization, J.V.; methodology, J.V.; software, J.V.; validation, '
          'J.V.; formal analysis, J.V.; investigation, J.V.; data curation, J.V.; writing—original draft '
          'preparation, J.V.; writing—review and editing, J.V.; visualization, J.V.; supervision, J.V. All '
          'authors have read and agreed to the published version of the manuscript.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'Funding: This research received no external funding.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'Institutional Review Board Statement: Not applicable.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'Informed Consent Statement: Not applicable.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'Data Availability Statement: The code and the derived results (CSV tables, document–topic matrix, '
          'and figures) presented in this study are openly available in GitHub at '
          'https://github.com/JeanVydes/bertlda. The raw bibliographic records are third-party data obtained '
          'from Scopus (Elsevier) and are subject to the provider’s terms of use; the search query used to '
          'retrieve them is reported in Section 2.1 so that readers with Scopus access can reproduce the '
          'corpus.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'Acknowledgments: During the preparation of this manuscript, the authors used Claude Sonnet 5.5 '
          '(Anthropic) for the purposes of restructuring the group’s methodology document into the journal '
          'format, drafting and language editing of the text, and formatting of tables and equations. No '
          'generative AI tool was used to produce results, analyses, or figures; all figures were generated '
          'by the authors’ Python pipeline from the corpus data. The authors have reviewed and edited the '
          'output and take full responsibility for the content of this publication.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': 'Conflicts of Interest: The authors declare no conflicts of interest.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p', 'text': 'Appendix A', 'style': 'Heading 1', 'align': None, 'img': None},
 {'type': 'p',
  'text': 'Table A1. Complete K search (symmetric α, 30 passes). No value of K produced empty topics; the '
          'selected K = 27 is in bold.',
  'style': 'Normal',
  'align': 'LEFT (0)',
  'img': None},
 {'type': 'tbl',
  'rows': [['K', 'Cv', 'NPMI', 'Diversity', 'Perplexity', 'Dominance', 'Entropy'],
           ['4', '0.4826', '0.0377', '0.8750', '-7.4130', '0.2740', '0.9953'],
           ['5', '0.4846', '0.0366', '0.8200', '-7.3861', '0.2516', '0.9891'],
           ['6', '0.4841', '0.0340', '0.7500', '-7.3658', '0.2298', '0.9882'],
           ['7', '0.4989', '0.0400', '0.7429', '-7.3513', '0.2060', '0.9864'],
           ['8', '0.4806', '0.0347', '0.7125', '-7.3453', '0.1863', '0.9894'],
           ['9', '0.4922', '0.0406', '0.7444', '-7.3327', '0.1491', '0.9917'],
           ['10', '0.4865', '0.0386', '0.7000', '-7.3298', '0.1390', '0.9884'],
           ['11', '0.5377', '0.0570', '0.8091', '-7.3059', '0.1288', '0.9860'],
           ['12', '0.5189', '0.0502', '0.7750', '-7.3105', '0.1518', '0.9818'],
           ['13', '0.5354', '0.0610', '0.7769', '-7.3168', '0.1238', '0.9689'],
           ['14', '0.5406', '0.0581', '0.8000', '-7.3209', '0.1774', '0.9542'],
           ['15', '0.5383', '0.0634', '0.7467', '-7.3229', '0.1963', '0.9564'],
           ['16', '0.5350', '0.0611', '0.7688', '-7.3216', '0.1148', '0.9638'],
           ['17', '0.5298', '0.0567', '0.7588', '-7.3233', '0.1171', '0.9573'],
           ['18', '0.5524', '0.0641', '0.7778', '-7.3192', '0.1360', '0.9524'],
           ['19', '0.5492', '0.0642', '0.8000', '-7.3205', '0.1566', '0.9499'],
           ['20', '0.5471', '0.0655', '0.7450', '-7.3280', '0.1346', '0.9500'],
           ['21', '0.5556', '0.0710', '0.7952', '-7.3287', '0.1251', '0.9556'],
           ['22', '0.5535', '0.0715', '0.7818', '-7.3270', '0.1247', '0.9444'],
           ['23', '0.5463', '0.0665', '0.7826', '-7.3371', '0.1059', '0.9505'],
           ['24', '0.5583', '0.0696', '0.7792', '-7.3304', '0.1160', '0.9568'],
           ['25', '0.5508', '0.0650', '0.7800', '-7.3335', '0.1264', '0.9047'],
           ['26', '0.5561', '0.0661', '0.7654', '-7.3405', '0.1183', '0.9474'],
           ['27', '0.5640', '0.0749', '0.7889', '-7.3363', '0.1056', '0.9398'],
           ['28', '0.5638', '0.0732', '0.7679', '-7.3404', '0.0852', '0.9367'],
           ['29', '0.5606', '0.0725', '0.7655', '-7.3312', '0.0864', '0.9199'],
           ['30', '0.5449', '0.0622', '0.7867', '-7.3426', '0.0854', '0.9192'],
           ['31', '0.5545', '0.0697', '0.8065', '-7.3472', '0.0843', '0.9157'],
           ['32', '0.5654', '0.0755', '0.7781', '-7.3458', '0.0757', '0.9279'],
           ['33', '0.5520', '0.0660', '0.7545', '-7.3501', '0.0744', '0.9278'],
           ['34', '0.5598', '0.0672', '0.7647', '-7.3557', '0.0783', '0.9224'],
           ['35', '0.5534', '0.0707', '0.7800', '-7.3563', '0.1022', '0.9203'],
           ['36', '0.5647', '0.0737', '0.7833', '-7.3615', '0.1113', '0.9030'],
           ['37', '0.5641', '0.0740', '0.7892', '-7.3628', '0.0750', '0.9268'],
           ['38', '0.5659', '0.0758', '0.7421', '-7.3675', '0.0748', '0.9174'],
           ['39', '0.5508', '0.0606', '0.7615', '-7.3717', '0.0898', '0.8916'],
           ['40', '0.5696', '0.0781', '0.7825', '-7.3704', '0.0928', '0.9118']]},
 {'type': 'p', 'text': '', 'style': 'Normal', 'align': None, 'img': None},
 {'type': 'p', 'text': 'References', 'style': 'Heading 1', 'align': None, 'img': None},
 {'type': 'p',
  'text': '1.\tJobin, A.; Ienca, M.; Vayena, E. The Global Landscape of AI Ethics Guidelines. Nat. Mach. '
          'Intell. 2019, 1, 389–399. https://doi.org/10.1038/s42256-019-0088-2.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '2.\tFloridi, L.; Cowls, J.; Beltrametti, M.; Chatila, R.; Chazerand, P.; Dignum, V.; Luetge, C.; '
          'Madelin, R.; Pagallo, U.; Rossi, F.; et al. AI4People—An Ethical Framework for a Good AI Society: '
          'Opportunities, Risks, Principles, and Recommendations. Minds Mach. 2018, 28, 689–707. '
          'https://doi.org/10.1007/s11023-018-9482-5.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '3.\tHagendorff, T. The Ethics of AI Ethics: An Evaluation of Guidelines. Minds Mach. 2020, 30, '
          '99–120. https://doi.org/10.1007/s11023-020-09517-8.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '4.\tMittelstadt, B. Principles Alone Cannot Guarantee Ethical AI. Nat. Mach. Intell. 2019, 1, '
          '501–507. https://doi.org/10.1038/s42256-019-0114-4.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '5.\tBender, E.M.; Gebru, T.; McMillan-Major, A.; Shmitchell, S. On the Dangers of Stochastic '
          'Parrots: Can Language Models Be Too Big? In Proceedings of the 2021 ACM Conference on Fairness, '
          'Accountability, and Transparency, Virtual Event, 3–10 March 2021; ACM: New York, NY, USA, 2021; '
          'pp. 610–623. https://doi.org/10.1145/3442188.3445922.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '6.\tWeidinger, L.; Mellor, J.; Rauh, M.; Griffin, C.; Uesato, J.; Huang, P.-S.; Cheng, M.; '
          'Glaese, M.; Balle, B.; Kasirzadeh, A.; et al. Ethical and Social Risks of Harm from Language '
          'Models. arXiv 2021, arXiv:2112.04359.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '7.\tSaheb, T.; Saheb, T. Mapping Ethical Artificial Intelligence Policy Landscape: A Mixed Method '
          'Analysis. Sci. Eng. Ethics 2024, 30, 9. https://doi.org/10.1007/s11948-024-00472-6.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '8.\tBinh, N.T.T. Transforming Auditing in the AI Era: A Comprehensive Review. Information 2025, '
          '16, 400. https://doi.org/10.3390/info16050400.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '9.\tGrimmer, J.; Stewart, B.M. Text as Data: The Promise and Pitfalls of Automatic Content '
          'Analysis Methods for Political Texts. Polit. Anal. 2013, 21, 267–297. '
          'https://doi.org/10.1093/pan/mps028.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '10.\tDebortoli, S.; Müller, O.; Junglas, I.; vom Brocke, J. Text Mining for Information Systems '
          'Researchers: An Annotated Topic Modeling Tutorial. Commun. Assoc. Inf. Syst. 2016, 39, 110–135. '
          'https://doi.org/10.17705/1CAIS.03907.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '11.\tAgrawal, A.; Fu, W.; Menzies, T. What Is Wrong with Topic Modeling? (and How to Fix It Using '
          'Search-Based Software Engineering). Inf. Softw. Technol. 2018, 98, 74–88. '
          'https://doi.org/10.1016/j.infsof.2018.02.005.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '12.\tGreene, D.; O’Callaghan, D.; Cunningham, P. How Many Topics? Stability Analysis for Topic '
          'Models. In Machine Learning and Knowledge Discovery in Databases; Calders, T., Esposito, F., '
          'Hüllermeier, E., Meo, R., Eds.; Lecture Notes in Computer Science; Springer: Berlin/Heidelberg, '
          'Germany, 2014; Volume 8724, pp. 498–513. https://doi.org/10.1007/978-3-662-44848-9_32.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '13.\tWallach, H.M.; Mimno, D.; McCallum, A. Rethinking LDA: Why Priors Matter. In Advances in '
          'Neural Information Processing Systems 22; Bengio, Y., Schuurmans, D., Lafferty, J., Williams, C., '
          'Culotta, A., Eds.; Curran Associates: Red Hook, NY, USA, 2009; pp. 1973–1981.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '14.\tChang, J.; Gerrish, S.; Wang, C.; Boyd-Graber, J.; Blei, D. Reading Tea Leaves: How Humans '
          'Interpret Topic Models. In Advances in Neural Information Processing Systems 22; Bengio, Y., '
          'Schuurmans, D., Lafferty, J., Williams, C., Culotta, A., Eds.; Curran Associates: Red Hook, NY, '
          'USA, 2009; pp. 288–296.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '15.\tBaas, J.; Schotten, M.; Plume, A.; Côté, G.; Karimi, R. Scopus as a Curated, High-Quality '
          'Bibliometric Data Source for Academic Research in Quantitative Science Studies. Quant. Sci. Stud. '
          '2020, 1, 377–386. https://doi.org/10.1162/qss_a_00019.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '16.\tPage, M.J.; McKenzie, J.E.; Bossuyt, P.M.; Boutron, I.; Hoffmann, T.C.; Mulrow, C.D.; '
          'Shamseer, L.; Tetzlaff, J.M.; Akl, E.A.; Brennan, S.E.; et al. The PRISMA 2020 Statement: An '
          'Updated Guideline for Reporting Systematic Reviews. BMJ 2021, 372, n71. '
          'https://doi.org/10.1136/bmj.n71.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '17.\tDenny, M.J.; Spirling, A. Text Preprocessing for Unsupervised Learning: Why It Matters, When '
          'It Misleads, and What to Do About It. Polit. Anal. 2018, 26, 168–189. '
          'https://doi.org/10.1017/pan.2017.44.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '18.\tSchofield, A.; Magnusson, M.; Mimno, D. Pulling Out the Stops: Rethinking Stopword Removal '
          'for Topic Models. In Proceedings of the 15th Conference of the European Chapter of the '
          'Association for Computational Linguistics: Volume 2, Short Papers, Valencia, Spain, 3–7 April '
          '2017; ACL: Stroudsburg, PA, USA, 2017; pp. 432–436.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '19.\tManning, C.D.; Raghavan, P.; Schütze, H. Introduction to Information Retrieval; Cambridge '
          'University Press: Cambridge, UK, 2008.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '20.\tBird, S.; Klein, E.; Loper, E. Natural Language Processing with Python; O’Reilly Media: '
          'Sebastopol, CA, USA, 2009.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '21.\tMiller, G.A. WordNet: A Lexical Database for English. Commun. ACM 1995, 38, 39–41. '
          'https://doi.org/10.1145/219717.219748.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '22.\tMikolov, T.; Sutskever, I.; Chen, K.; Corrado, G.S.; Dean, J. Distributed Representations of '
          'Words and Phrases and Their Compositionality. In Advances in Neural Information Processing '
          'Systems 26; Burges, C.J.C., Bottou, L., Welling, M., Ghahramani, Z., Weinberger, K.Q., Eds.; '
          'Curran Associates: Red Hook, NY, USA, 2013; pp. 3111–3119.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '23.\tŘehůřek, R.; Sojka, P. Software Framework for Topic Modelling with Large Corpora. In '
          'Proceedings of the LREC 2010 Workshop on New Challenges for NLP Frameworks, Valletta, Malta, 22 '
          'May 2010; ELRA: Paris, France, 2010; pp. 45–50.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '24.\tMcKinney, W. Data Structures for Statistical Computing in Python. In Proceedings of the 9th '
          'Python in Science Conference, Austin, TX, USA, 28 June–3 July 2010; pp. 56–61.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '25.\tHarris, C.R.; Millman, K.J.; van der Walt, S.J.; Gommers, R.; Virtanen, P.; Cournapeau, D.; '
          'Wieser, E.; Taylor, J.; Berg, S.; Smith, N.J.; et al. Array Programming with NumPy. Nature 2020, '
          '585, 357–362. https://doi.org/10.1038/s41586-020-2649-2.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '26.\tBlei, D.M.; Ng, A.Y.; Jordan, M.I. Latent Dirichlet Allocation. J. Mach. Learn. Res. 2003, '
          '3, 993–1022.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '27.\tHoffman, M.; Bach, F.; Blei, D. Online Learning for Latent Dirichlet Allocation. In Advances '
          'in Neural Information Processing Systems 23; Lafferty, J., Williams, C., Shawe-Taylor, J., Zemel, '
          'R., Culotta, A., Eds.; Curran Associates: Red Hook, NY, USA, 2010; pp. 856–864.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '28.\tRöder, M.; Both, A.; Hinneburg, A. Exploring the Space of Topic Coherence Measures. In '
          'Proceedings of the Eighth ACM International Conference on Web Search and Data Mining, Shanghai, '
          'China, 2–6 February 2015; ACM: New York, NY, USA, 2015; pp. 399–408. '
          'https://doi.org/10.1145/2684822.2685324.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '29.\tBouma, G. Normalized (Pointwise) Mutual Information in Collocation Extraction. In '
          'Proceedings of the Biennial GSCL Conference 2009; Chiarcos, C., Eckart de Castilho, R., Stede, '
          'M., Eds.; Gunter Narr Verlag: Tübingen, Germany, 2009; pp. 31–40.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '30.\tLau, J.H.; Newman, D.; Baldwin, T. Machine Reading Tea Leaves: Automatically Evaluating '
          'Topic Coherence and Topic Model Quality. In Proceedings of the 14th Conference of the European '
          'Chapter of the Association for Computational Linguistics, Gothenburg, Sweden, 26–30 April 2014; '
          'ACL: Stroudsburg, PA, USA, 2014; pp. 530–539. https://doi.org/10.3115/v1/E14-1056.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '31.\tDe Waal, A.; Barnard, E. Evaluating Topic Models with Stability. In Proceedings of the 19th '
          'Annual Symposium of the Pattern Recognition Association of South Africa, Cape Town, South Africa, '
          '2008; pp. 79–84.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '32.\tKoltcov, S.; Koltsova, O.; Nikolenko, S. Latent Dirichlet Allocation: Stability and '
          'Applications to Studies of User-Generated Content. In Proceedings of the 2014 ACM Conference on '
          'Web Science, Bloomington, IN, USA, 23–26 June 2014; ACM: New York, NY, USA, 2014; pp. 161–165. '
          'https://doi.org/10.1145/2615569.2615680.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '33.\tKuhn, H.W. The Hungarian Method for the Assignment Problem. Nav. Res. Logist. Q. 1955, 2, '
          '83–97. https://doi.org/10.1002/nav.3800020109.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '34.\tHubert, L.; Arabie, P. Comparing Partitions. J. Classif. 1985, 2, 193–218. '
          'https://doi.org/10.1007/BF01908075.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '35.\tGalindo-Villardón, M.P. Una Alternativa de Representación Simultánea: HJ-Biplot. Qüestiió '
          '1986, 10, 13–23.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '36.\tGabriel, K.R. The Biplot Graphic Display of Matrices with Application to Principal Component '
          'Analysis. Biometrika 1971, 58, 453–467. https://doi.org/10.1093/biomet/58.3.453.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '37.\tSievert, C.; Shirley, K. LDAvis: A Method for Visualizing and Interpreting Topics. In '
          'Proceedings of the Workshop on Interactive Language Learning, Visualization, and Interfaces, '
          'Baltimore, MD, USA, 27 June 2014; ACL: Stroudsburg, PA, USA, 2014; pp. 63–70. '
          'https://doi.org/10.3115/v1/W14-3110.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '38.\tHunter, J.D. Matplotlib: A 2D Graphics Environment. Comput. Sci. Eng. 2007, 9, 90–95. '
          'https://doi.org/10.1109/MCSE.2007.55.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '39.\tVides, J. bertlda: LDA Topic-Modeling Pipeline for the AI-Ethics and Generative-AI '
          'Literature. Available online: https://github.com/JeanVydes/bertlda (accessed on 19 November '
          '2026).',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '40.\tAitchison, J.; Greenacre, M. Biplots of Compositional Data. J. R. Stat. Soc. Ser. C Appl. '
          'Stat. 2002, 51, 375–392. https://doi.org/10.1111/1467-9876.00275.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None},
 {'type': 'p',
  'text': '41.\tGrootendorst, M. BERTopic: Neural Topic Modeling with a Class-Based TF-IDF Procedure. arXiv '
          '2022, arXiv:2203.05794.',
  'style': 'Normal',
  'align': 'JUSTIFY (3)',
  'img': None}]
