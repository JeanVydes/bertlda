# -*- coding: utf-8 -*-
"""
text_processing.py
=================
Text cleaning utilities shared by every phase of the pipeline.

Adapted from the legacy procesador_v3.py utility (Spanish codebase) but
rewritten for the bertlda case:
  * works on the preprocessed CSV produced by preprocess_corpus.py
  * outputs only CSV-friendly objects (lists / strings, never pandas DataFrame)
  * no SciPy/Sklearn imports at module load
  * English identifiers, no dead code
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

import nltk
from nltk.stem import WordNetLemmatizer

# -----------------------------------------------------------------------------
# 1. NLTK resources (lazy, quiet)
# -----------------------------------------------------------------------------
def _ensure_nltk():
    for name, marker in {
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
        "stopwords": "corpora/stopwords",
    }.items():
        try:
            nltk.data.find(marker)
        except LookupError:
            nltk.download(name, quiet=True)


_ensure_nltk()
_LEMM = WordNetLemmatizer()

# -----------------------------------------------------------------------------
# 2. Stopword layers (3 layers: functional / editorial / methodological)
# -----------------------------------------------------------------------------
def functional_stopwords() -> set:
    """Layer 1: NLTK English (179 terms)."""
    from nltk.corpus import stopwords

    try:
        return set(stopwords.words("english"))
    except Exception:
        return set()


EDITORIAL_STOPWORDS = {
    "elsevier", "springer", "wiley", "taylor", "francis", "informa",
    "oxford", "cambridge", "sage", "emerald", "frontiers", "mdpi",
    "gmbh", "llc", "ltd", "limited", "press", "publisher", "publishing",
    "publication", "copyright", "reserved", "licence", "license",
    "licensee", "behalf", "author", "authors", "journal", "article",
    "abstract", "introduction", "conclusion", "discussion",
    "doi", "issn", "isbn", "volume", "issue",
}

METHODOLOGICAL_STOPWORDS = {
    "study", "result", "results", "paper", "research", "analysis",
    "method", "methods", "approach", "data", "using", "used", "use",
    "show", "shown", "found", "present", "provide", "based",
    "suggest", "indicate", "demonstrate", "reveal", "report",
    "describe", "examine", "investigate", "evaluate", "assess",
    "obtain", "perform", "conduct", "propose", "proposed",
    "however", "therefore", "thus", "furthermore", "moreover",
    "also", "may", "can", "well", "due", "new", "different",
    "fig", "figure", "table", "section", "et", "al",
}


def build_stopwords(
    domain: set | None = None,
    include_methodological: bool = True,
    extra: set | None = None,
) -> set:
    """Assemble the 3 stopword layers into one set."""
    stops = set(functional_stopwords()) | set(EDITORIAL_STOPWORDS)
    if include_methodological:
        stops |= set(METHODOLOGICAL_STOPWORDS)
    if domain:
        stops |= set(domain)
    if extra:
        stops |= set(extra)
    return {w for w in stops if w and w != "nan"}


def derive_domain_stopwords(
    texts: Iterable[str],
    threshold: float = 0.60,
    min_len: int = 4,
    top_n: int = 80,
) -> list[tuple[str, float]]:
    """Empirical layer 3: terms present in >= `threshold` of documents."""
    base = functional_stopwords() | EDITORIAL_STOPWORDS | METHODOLOGICAL_STOPWORDS
    n = len(list(texts)) if not isinstance(texts, list) else len(texts)
    counter = Counter()
    for t in texts:
        tokens = set(re.findall(rf"[a-z]{{{min_len},}}", str(t).lower()))
        counter.update(tokens)
    out = []
    for w, c in counter.items():
        rate = c / max(n, 1)
        if rate >= threshold and w not in base:
            out.append((w, round(rate * 100, 1)))
    out.sort(key=lambda x: -x[1])
    return out[:top_n]


# -----------------------------------------------------------------------------
# 3. Regex for LDA + embeddings
# -----------------------------------------------------------------------------
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_EMAIL = re.compile(r"\S+@\S+")
RE_SPACES = re.compile(r"\s+")
RE_CITATION = re.compile(r"\[\s*\d+(?:\s*[,-]\s*\d+)*\s*\]")
RE_COPYRIGHT = re.compile(
    r"(?:\(\s*c\s*\)|©)\s*\d{4}[^.]*?(?:\.|$)"
    r"|all\s+rights?\s+reserved[^.]*?(?:\.|$)"
    r"|exclusive\s+licen[sc]e\s+to[^.]*?(?:\.|$)"
    r"|published\s+by\s+(?:elsevier|springer|wiley|taylor|informa|mdpi|"
    r"oxford|cambridge|sage|emerald|frontiers)[^.]*?(?:\.|$)"
    r"|on\s+behalf\s+of\s+[^.]{0,80}?(?:\.|$)"
    r"|this\s+is\s+an\s+open\s+access\s+article[^.]*?(?:\.|$)",
    re.IGNORECASE,
)
RE_NO_ALNUM = re.compile(r"[^a-z0-9\-\s]")
RE_EXTRA_DASH = re.compile(r"(^-+)|(-+$)")
RE_ONLY_DIGIT = re.compile(r"^[\d\-]+$")
RE_TITLE_TRANSLATED = re.compile(r"\s*;\s*\[.*?\]\s*$", re.DOTALL)
SENTENCE_INITIAL = re.compile(r"(^|(?<=[.!?])\s+)([a-z])")

VALID_ACRONYMS = {
    "dna", "rna", "gdp", "gis", "iot", "sst", "mpa", "eez", "ghg", "co2",
    "ndc", "sdg", "sme", "gps", "led", "uav", "auv", "rov", "ai", "ml",
    "gas", "oil", "air", "ice", "sea", "bay", "law", "eco", "usa",
    "eu", "un", "ph", "uv", "tv",
}

# AI-ethics specific acronyms (extends the legacy set)
VALID_ACRONYMS_ETICA = VALID_ACRONYMS | {
    "llm", "llms", "nlp", "gpt", "genai", "gdpr", "eu", "un", "usa",
    "hipaa", "ccpa", "pipa", "api", "ux", "hci", "ip",
}


def lemmatize(word: str) -> str:
    n = _LEMM.lemmatize(word, pos="n")
    if n != word:
        return n
    return _LEMM.lemmatize(word, pos="v")


def clean_for_lda(text, stops: set, min_len: int = 3,
                  acronyms: set | None = None) -> list[str]:
    """Returns a list of lemmatized tokens. Lemmatizes BEFORE filtering stopwords."""
    if acronyms is None:
        acronyms = VALID_ACRONYMS_ETICA
    t = str(text).lower()
    t = RE_URL.sub(" ", t)
    t = RE_EMAIL.sub(" ", t)
    t = RE_CITATION.sub(" ", t)
    t = RE_COPYRIGHT.sub(" ", t)
    t = RE_NO_ALNUM.sub(" ", t)

    out = []
    for w in t.split():
        w = RE_EXTRA_DASH.sub("", w)
        if not w or RE_ONLY_DIGIT.match(w):
            continue
        lemma = lemmatize(w)
        if lemma in stops:
            continue
        if len(lemma) < min_len and lemma not in acronyms:
            continue
        out.append(lemma)
    return out


def clean_title_translated(t: str) -> str:
    """Removes translated title appended in brackets (Scopus artifact)."""
    return RE_TITLE_TRANSLATED.sub("", str(t)).strip()


def normalizar_casing(texto: str) -> str:
    """ALL CAPS -> sentence case."""
    t = str(texto).lower()
    return SENTENCE_INITIAL.sub(lambda m: m.group(1) + m.group(2).upper(), t)


def detectar_mayusculas(serie, umbral: float = 0.80) -> tuple[float, bool]:
    s = serie.astype(str)
    frac = float((s == s.str.upper()).mean())
    return frac, frac > umbral





# -----------------------------------------------------------------------------
# 4. Country normalization map (Scopus-style variants)
# -----------------------------------------------------------------------------
PAIS_MAPA = {
    "usa": "United States", "u.s.a.": "United States",
    "united states of america": "United States",
    "uk": "United Kingdom", "u.k.": "United Kingdom",
    "great britain": "United Kingdom", "england": "United Kingdom",
    "scotland": "United Kingdom", "wales": "United Kingdom",
    "northern ireland": "United Kingdom",
    "p.r. china": "China", "people's republic of china": "China",
    "deutschland": "Germany", "italia": "Italy", "españa": "Spain",
    "holland": "Netherlands", "brasil": "Brazil", "méxico": "Mexico",
    "suisse": "Switzerland", "schweiz": "Switzerland", "sverige": "Sweden",
    "belgique": "Belgium", "danmark": "Denmark", "suomi": "Finland",
    "norge": "Norway", "polska": "Poland", "rusia": "Russia",
    "türkiye": "Turkey", "éire": "Ireland", "viet nam": "Vietnam",
    "perú": "Peru", "egipto": "Egypt", "tailandia": "Thailand",
    "aotearoa": "New Zealand",
}


def normalizar_pais(valor: str | None) -> str | None:
    if valor is None:
        return None
    v = str(valor).strip()
    if not v or v.lower() == "nan":
        return None
    key = v.lower()
    return PAIS_MAPA.get(key, v.title() if v.isupper() else v)


# Canonical set of country names that may appear at the end of a Scopus
# affiliation segment. Used to filter non-country tokens.
PAISES_CANONICOS = {
    "United States", "United Kingdom", "China", "India", "Germany", "France",
    "Canada", "Australia", "Japan", "South Korea", "Italy", "Spain",
    "Netherlands", "Brazil", "Mexico", "Switzerland", "Sweden", "Belgium",
    "Austria", "Denmark", "Finland", "Norway", "Poland", "Russia", "Turkey",
    "Israel", "Singapore", "Taiwan", "Hong Kong", "Ireland", "Portugal",
    "Greece", "Czech Republic", "Czechia", "Hungary", "Romania", "Argentina",
    "Chile", "Colombia", "Peru", "South Africa", "Egypt", "Nigeria", "Kenya",
    "Malaysia", "Thailand", "Vietnam", "Indonesia", "Philippines",
    "United Arab Emirates", "Saudi Arabia", "New Zealand", "Marshall Islands",
    "South Korea", "Pakistan", "Bangladesh", "Iran", "Iran, Islamic Republic of",
    "Morocco", "Tunisia", "Algeria", "Kenya", "Ethiopia", "Ghana",
    "Tanzania", "Uganda", "Zimbabwe", "Sudan", "Sri Lanka", "Nepal",
    "Kazakhstan", "Uzbekistan", "Ukraine", "Croatia", "Serbia", "Slovenia",
    "Slovakia", "Bulgaria", "Lithuania", "Latvia", "Estonia", "Cyprus",
    "Malta", "Luxembourg", "Iceland", "Jordan", "Lebanon", "Qatar", "Kuwait",
    "Bahrain", "Oman", "Yemen", "Iraq", "Syria",
}

_PAIS_LOOKUP = {c.lower(): c for c in PAISES_CANONICOS}
# add variants to lookup
for k, v in PAIS_MAPA.items():
    _PAIS_LOOKUP[k] = v


def _match_country_token(token: str) -> str | None:
    key = token.strip().lower().rstrip(".")
    if key in _PAIS_LOOKUP:
        return _PAIS_LOOKUP[key]
    # try title-case
    if key.title() in PAISES_CANONICOS:
        return key.title()
    return None


def extract_countries_from_text(text: str) -> list[str]:
    """Extract unique country names from a Scopus affiliation string."""
    if not isinstance(text, str) or not text.strip():
        return []
    out: list[str] = []
    for seg in str(text).split(";"):
        seg = seg.strip()
        if not seg:
            continue
        parts = [p.strip() for p in seg.split(",") if p.strip()]
        found = None
        # Country is usually the LAST comma-separated piece; try 1, then 2, then 3.
        for n in (1, 2, 3):
            cand = " ".join(parts[-n:])
            norm = _match_country_token(cand)
            if norm:
                found = norm
                break
        if found and found not in out:
            out.append(found)
    return out


def enrich_with_countries(df: pd.DataFrame,
                         cols_to_try: tuple = (
                             "Affiliations", "Authors with affiliations",
                             "Correspondence Address",
                         )) -> pd.DataFrame:
    """Add `pais_principal` and `paises_todos` columns to df."""
    df = df.copy()

    def para_fila(row):
        countries: list[str] = []
        for col in cols_to_try:
            if col in row and pd.notna(row[col]):
                cs = extract_countries_from_text(str(row[col]))
                if cs:
                    countries = cs
                    break
        return countries

    listas = df.apply(para_fila, axis=1)
    df["paises_todos"] = listas.apply(lambda x: "; ".join(x))
    df["pais_principal"] = listas.apply(lambda x: x[0] if x else None)
    df["pais_normalizado"] = df["pais_principal"].apply(normalizar_pais)
    df["pais_original"] = df["pais_principal"]
    return df


# -----------------------------------------------------------------------------
# 5. Continuous parameters from corpus size (adapted from get_params)
# -----------------------------------------------------------------------------
def etiqueta_escala(n_docs: int) -> str:
    if n_docs < 5_000:
        return "MICRO"
    if n_docs < 30_000:
        return "SMALL"
    if n_docs <= 50_000:
        return "MEDIUM"
    return "LARGE"


def ms_para(mcs: int) -> list[int]:
    return sorted({1, max(1, mcs // 20), max(1, mcs // 10)})


def get_params(n_docs: int) -> dict:
    """Derive LDA parameters continuously from corpus size."""
    n = int(n_docs)
    params = {
        "n_docs": n,
        "escala": etiqueta_escala(n),
        "k_min": 4,
        "k_max": 40,
        "k_step": 1 if n < 30_000 else 2,
        "passes_lda": 30 if n < 30_000 else 20,
        "passes_fijo": 30 if n < 30_000 else 20,
        "no_below": max(2, int(round(n * 0.002))),
        "no_above": 0.50,
        "bigram_min_count": max(3, int(round(n * 0.003))),
        "bigram_threshold": 8,
        "alpha_grid": ["symmetric", "asymmetric", 0.01, 0.1],
        "eta_grid": ["symmetric", 0.01, 0.05, 0.5],
        "coherence_sample": None if n <= 10_000 else 10_000,
    }
    return params


if __name__ == "__main__":
    print("text_processing.py loaded")
    demo = "STUDIES ON COVID-19 AND CO2 SHOW THAT RESULTS ARE ANALYSED."
    stops = build_stopwords()
    print(f"  demo input : {demo}")
    print(f"  demo output: {clean_for_lda(demo, stops)}")
    n = 20_000
    p = get_params(n)
    print(f"  params for {n:,} docs [{p['escala']}]: "
          f"K={p['k_min']}-{p['k_max']}")
