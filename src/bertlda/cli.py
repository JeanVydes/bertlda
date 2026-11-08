# -*- coding: utf-8 -*-
"""
cli.py
======
Command-line interface for the bertlda multi-stage topic-modeling pipeline.

Examples
--------
    bertlda --phase phase0
    bertlda --phase phase1a
    bertlda --phase phase1b
    bertlda --phase phase1c
    bertlda --phase preprocess-corpus --input raw_corpus.csv
    bertlda --list
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

try:
    _DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _DIR = os.getcwd()
sys.path.insert(0, _DIR)

PHASE_REGISTRY = [
    ("phase0", "phase0_audit", "main",
     "Corpus audit: validation, doc_id, countries, stopwords."),
    ("phase1a", "phase1a_lda_diagnostic", "main",
     "LDA diagnosis: tokenization, bigrams, K search."),
    ("phase1b", "phase1b_lda_decision", "main",
     "LDA decision: stability, K choice, alpha/eta search."),
    ("phase1c", "phase1c_lda_final", "main",
     "LDA final model: training, Theta, topics, exports."),
    ("phase2a", "phase2a_embeddings_compare", "main",
     "Compare embedding models (optional, heavy)."),
    ("phase2b", "phase2b_bertopic_grid", "main",
     "BERTopic grid: UMAP + HDBSCAN exploration."),
    ("phase2c", "phase2c_bertopic_final", "main",
     "BERTopic final model from a manually selected grid combination."),
    ("phase3", "phase3_visualizations", "main",
     "Visualizations for LDA and BERTopic."),
    ("phase4", "phase4_trends", "main",
     "Temporal trend analysis (OLS, Mann-Kendall, beta regression)."),
    ("phase5", "phase5_sensitivity", "main",
     "LDA sensitivity: seed stability (Jaccard/ARI) + K neighbourhood."),
    ("phase6", "phase6_report", "main",
     "Deliverables: full config JSON, HJ-Biplot, LDAvis, Methodology_LDA.md."),
    ("preprocess-corpus", "preprocess_corpus", "main",
     "Preprocess a Scopus export corpus (Title+Abstract -> tokens)."),
]


def _run_phase(name: str) -> int:
    mod_name, fn_name = None, "main"
    for reg_name, module, func, _ in PHASE_REGISTRY:
        if reg_name == name:
            mod_name, fn_name = module, func
            break
    if mod_name is None:
        print(f"Unknown phase: {name}")
        return 2
    try:
        mod = importlib.import_module(mod_name)
    except Exception as e:
        print(f"Failed to import {mod_name}: {e}")
        return 1
    fn = getattr(mod, fn_name, None)
    if fn is None:
        print(f"Module {mod_name} has no {fn_name}().")
        return 1
    try:
        fn()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Phase {name} failed: {e}")
        return 1
    return 0


def _list_phases():
    print("Available phases:")
    for name, _, _, desc in PHASE_REGISTRY:
        print(f"  {name:<16} {desc}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="bertlda", description="bertlda multi-stage pipeline (CSV only).")
    parser.add_argument("--phase", metavar="NAME", help="Phase to run (see --list).")
    parser.add_argument("--list", action="store_true", help="List available phases.")
    parser.add_argument("--input", metavar="FILE.csv", help="Input CSV for preprocess-corpus.")
    parser.add_argument("--out-dir", metavar="DIR", default=None, help="Output dir (preprocess-corpus only).")
    args = parser.parse_args(argv)

    if args.list:
        _list_phases()
        return 0
    if args.phase is None:
        _list_phases()
        return 0
    if args.phase == "preprocess-corpus":
        import preprocess_corpus
        path = args.input
        out = args.out_dir or "./outputs"
        if path is None:
            print("--input FILE.csv required for preprocess-corpus")
            return 2
        preprocess_corpus.main(["--preprocess-corpus", path, "--out-dir", out])
        return 0
    return _run_phase(args.phase)


if __name__ == "__main__":
    sys.exit(main())
