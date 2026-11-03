# -*- coding: utf-8 -*-
"""
pipeline_base.py
================
Shared helpers for every pipeline phase:
- CSV-only I/O (the pipeline never uses xlsx)
- pickle checkpoints for complex objects
- country extraction from Scopus affiliation strings
- logging/formatting helpers
"""
from __future__ import annotations

import json
import os
import pickle
import re
from pathlib import Path
from time import perf_counter

import pandas as pd

from config import (
    COL_AFFILIATIONS, COL_AUTHORS_AFFIL, COL_CORRESPONDENCE,
    COL_DOC_ID, COL_ANIO, COL_PAIS_NORMALIZADO, COL_PAIS_ORIGINAL,
    COL_TEXT, CHECKPOINT_DIR, RESULTS_DIR, VIZ_DIR, MODELS_DIR,
)

# Output sub-folders
CORPUS_AUDIT_DIR = RESULTS_DIR / "corpus"
LDA_RESULTS_DIR = RESULTS_DIR / "lda"
BERTOPIC_RESULTS_DIR = RESULTS_DIR / "bertopic"
TRENDS_DIR = RESULTS_DIR / "trends"


def asegurar_carpetas():
    for d in [
        CHECKPOINT_DIR, RESULTS_DIR, VIZ_DIR, MODELS_DIR,
        CORPUS_AUDIT_DIR, LDA_RESULTS_DIR, BERTOPIC_RESULTS_DIR,
        TRENDS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# CSV I/O (always CSV, never xlsx)
# -----------------------------------------------------------------------------
def leer_csv(ruta, **kwargs) -> pd.DataFrame:
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"File not found: {ruta}")
    df = pd.read_csv(ruta, **kwargs)
    print(f"  Read {ruta.name}: {len(df):,} rows x {df.shape[1]} cols")
    return df


def guardar_csv(df, nombre_archivo, subdir=RESULTS_DIR, index=False,
                encoding="utf-8-sig") -> Path:
    subdir = Path(subdir)
    subdir.mkdir(parents=True, exist_ok=True)
    ruta = subdir / nombre_archivo
    df.to_csv(ruta, index=index, encoding=encoding)
    print(f"  Saved CSV: {ruta} ({len(df):,} rows)")
    return ruta


def guardar_checkpoint(obj, nombre: str) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ruta = CHECKPOINT_DIR / f"{nombre}.pkl"
    with open(ruta, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved checkpoint: {ruta.name}")
    return ruta


def cargar_checkpoint(nombre: str):
    ruta = CHECKPOINT_DIR / f"{nombre}.pkl"
    if not ruta.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ruta}. Run the previous phase first."
        )
    with open(ruta, "rb") as f:
        obj = pickle.load(f)
    print(f"  Loaded checkpoint: {ruta.name}")
    return obj


def checkpoint_existe(nombre: str) -> bool:
    return (CHECKPOINT_DIR / f"{nombre}.pkl").exists()


def guardar_json(obj, nombre: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ruta = RESULTS_DIR / f"{nombre}.json"
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"  Saved JSON: {ruta.name}")
    return ruta


# -----------------------------------------------------------------------------
# Country extraction from Scopus affiliation strings
# -----------------------------------------------------------------------------
# Scopus puts affiliations separated by ';'. Each affiliation segment ends with
# the country (e.g. "..., Seattle, United States"). Each document may have
# several affiliations => several countries.
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
    "Pakistan", "Bangladesh", "Iran", "Morocco", "Tunisia", "Algeria",
    "Ethiopia", "Ghana", "Tanzania", "Uganda", "Zimbabwe", "Sudan",
    "Sri Lanka", "Nepal", "Kazakhstan", "Uzbekistan", "Ukraine", "Croatia",
    "Serbia", "Slovenia", "Slovakia", "Bulgaria", "Lithuania", "Latvia",
    "Estonia", "Cyprus", "Malta", "Luxembourg", "Iceland", "Jordan",
    "Lebanon", "Qatar", "Kuwait", "Bahrain", "Oman", "Yemen", "Iraq",
    "Syria", "Taiwan", "Macau",
}

_PAIS_LOOKUP: dict[str, str] = {c.lower(): c for c in PAISES_CANONICOS}
for k, v in PAIS_MAPA.items():
    _PAIS_LOOKUP[k] = v


def _match_country_token(token: str) -> str | None:
    key = token.strip().lower().rstrip(".")
    if key in _PAIS_LOOKUP:
        return _PAIS_LOOKUP[key]
    if key.title() in PAISES_CANONICOS:
        return key.title()
    return None


def extract_countries_from_text(text: str) -> list[str]:
    """Return the unique list of countries found in a Scopus affiliation string."""
    if not isinstance(text, str) or not text.strip():
        return []
    out: list[str] = []
    for seg in str(text).split(";"):
        seg = seg.strip()
        if not seg:
            continue
        parts = [p.strip() for p in seg.split(",") if p.strip()]
        found = None
        for n in (1, 2, 3):
            cand = " ".join(parts[-n:])
            norm = _match_country_token(cand)
            if norm:
                found = norm
                break
        if found and found not in out:
            out.append(found)
    return out


def enrich_with_countries(df: pd.DataFrame) -> pd.DataFrame:
    """Add `pais_principal` (first country) and `paises_todos` columns."""
    df = df.copy()

    def para_fila(row):
        for col in (COL_AFFILIATIONS, COL_AUTHORS_AFFIL, COL_CORRESPONDENCE):
            if col in row.index and pd.notna(row[col]):
                cs = extract_countries_from_text(str(row[col]))
                if cs:
                    return cs
        return []

    listas = df.apply(para_fila, axis=1)
    df["paises_todos"] = listas.apply(lambda x: "; ".join(x))
    df[COL_PAIS_ORIGINAL] = listas.apply(lambda x: x[0] if x else None)
    df[COL_PAIS_NORMALIZADO] = df[COL_PAIS_ORIGINAL].apply(
        lambda v: PAIS_MAPA.get(str(v).strip().lower(), v) if v else None
    )
    # ensure canonical via PAISES set fallback (already there for known)
    df[COL_PAIS_NORMALIZADO] = df[COL_PAIS_ORIGINAL].apply(
        lambda v: _PAIS_LOOKUP.get(str(v).strip().lower(), v) if v else None
    )
    return df


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _fmt(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m):3d}m {s:05.2f}s"


def print_separador(titulo: str, ancho: int = 66) -> None:
    print("\n" + "=" * ancho)
    print(titulo)
    print("=" * ancho)


if __name__ == "__main__":
    from config import RAW_CORPUS
    import pandas as pd
    df = pd.read_csv(RAW_CORPUS, nrows=5)
    df = enrich_with_countries(df)
    print(df[["Title", "Affiliations", "pais_original",
              "pais_normalizado", "paises_todos"]].head())
