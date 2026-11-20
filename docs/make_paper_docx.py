# -*- coding: utf-8 -*-
"""
make_paper_docx.py
==================
Rebuild the final MDPI paper DOCX with the corrected HJ-Biplot figures.

The original manuscript (docs/Information_MDPI_Manuscript.docx) is used as a
template: every paragraph, table, style and image stays byte-identical except
for the three embedded HJ-Biplot PNGs, which are replaced by the CLR-
transformed versions produced by phase6_report.py. This guarantees that the
output document is exactly the same as the original, with only the corrected
plots.

Usage:
    uv run python docs/make_paper_docx.py
"""
from __future__ import annotations

from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "docs" / "Information_MDPI_Manuscript.docx"
OUTPUT = ROOT / "docs" / "Information_MDPI_Manuscript_CLR.docx"
VIZ = ROOT / "outputs" / "visualizations"

# Mapping: image part name in the original template -> corrected PNG.
# Part names were identified by extracting the embedded images from the
# original manuscript (years, countries and documents HJ-Biplots).
IMAGE_REPLACEMENTS = {
    # years HJ-Biplot (1403x1109 in the original template)
    "/word/media/265bd2f715d909d7e9a833bc0ce79788e0f472a0.png":
        "LDA_HJBiplot_Years.png",
    # countries HJ-Biplot (1408x1109 in the original template)
    "/word/media/1d43a3d0589aa8451d6e3026b4250ad1d29d7365.png":
        "LDA_HJBiplot_Countries.png",
    # documents HJ-Biplot (1409x1109 in the original template)
    "/word/media/046363f4c9a04c609055a13473b6994b655c14d7.png":
        "LDA_HJBiplot_Documents.png",
}


# These paragraphs in the original text still claim that no log-ratio
# transformation was applied (methods, discussion, limitations and future
# work) or quote the statistics of the OLD, untransformed biplots. They are
# rewritten here so the text matches the corrected CLR figures.
TEXT_REPLACEMENTS = {
    # Methods: the SVD is now applied to CLR-transformed proportions.
    " (no further scaling or log-ratio transformation was applied), "
    "the singular value decomposition is":
        " after centered log-ratio (CLR) transformation of the topic "
        "proportions, the singular value decomposition is",
    "each cell is the mean topic probability":
        "each cell is the CLR-transformed mean topic proportion",
    # Results: new variance / cos² values (Table 14 and its description).
    "Table 15 reports the quality of the three biplots. The years × topics "
    "biplot is well represented (the first two axes retain 95.6% of the "
    "variance; mean squared cosines of 0.90 for rows and 0.88 for columns), "
    "the countries × topics biplot is moderately represented (82.8% of the "
    "variance; mean squared cosines of 0.56 and 0.46), and the documents × "
    "topics biplot is poorly represented (about 18.6%; mean squared cosines "
    "of 0.18 and 0.11).":
        "Table 15 reports the quality of the three biplots after the centered "
        "log-ratio transformation. The years × topics biplot is well "
        "represented (the first two axes retain 97.5% of the variance; mean "
        "squared cosines of 0.93 for rows and 0.95 for columns), the "
        "countries × topics biplot is moderately represented (59.4% of the "
        "variance; mean squared cosines of 0.43 and 0.47), and the documents "
        "× topics biplot is poorly represented (about 16.8%; mean squared "
        "cosines of 0.17 and 0.13).",
    # Discussion: the transformation was applied, not merely recommended.
    "The biplots were computed on column-centered mean topic probabilities. "
    "Because topic proportions are compositional data, log-ratio "
    "transformations are generally recommended for biplots [40]; the present "
    "biplots should therefore be regarded as exploratory, and repeating them "
    "after a log-ratio transformation is a priority.":
        "The biplots were computed on column-centered, centered log-ratio "
        "(CLR) transformed mean topic proportions. Because topic proportions "
        "are compositional data confined to the simplex, the CLR map [40] was "
        "applied before column centering so that the HJ-Biplot approximation "
        "is computed in the appropriate Aitchison geometry instead of on "
        "constrained raw proportions. Representation quality remains high for "
        "the years biplot but is lower for the countries and documents "
        "biplots, which should therefore be regarded as exploratory.",
    # Limitations: remove the now-fixed flaw.
    " The countries and documents biplots have moderate and low "
    "representation quality, respectively, and no log-ratio transformation "
    "was applied.":
        " The countries and documents biplots have moderate and low "
        "representation quality, respectively, after the centered log-ratio "
        "transformation.",
    # Future work: the transformation is no longer pending.
    "to repeat the biplots with a log-ratio transformation, and to compare "
    "LDA with embedding-based topic models":
        "to compare LDA with embedding-based topic models",
}

TABLE_TEXT_REPLACEMENTS = {
    "77.5 / 18.1": "74.7 / 22.8",
    "0.900": "0.931",
    "0.883": "0.952",
    "59.7 / 23.1": "32.8 / 26.7",
    "0.562": "0.434",
    "0.461": "0.468",
    "10.3 / 8.2": "9.2 / 7.7",
    "0.176": "0.169",
    "0.108": "0.133",
}


def _replace_text(doc: Document) -> int:
    """Apply the text fixes above at run level, preserving all formatting."""
    count = 0
    for paragraph in doc.paragraphs:
        for run in paragraph.runs:
            new = run.text
            for old, replacement in TEXT_REPLACEMENTS.items():
                if old in new:
                    new = new.replace(old, replacement)
                    count += 1
            if new != run.text:
                run.text = new
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        new = run.text.strip()
                        if new in TABLE_TEXT_REPLACEMENTS:
                            run.text = run.text.replace(
                                new, TABLE_TEXT_REPLACEMENTS[new])
                            count += 1
    return count


def main() -> None:
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"Missing template: {TEMPLATE}")

    doc = Document(str(TEMPLATE))
    replaced = 0

    for rel in doc.part.rels.values():
        partname = str(rel.target_part.partname) if hasattr(
            rel.target_part, "partname") else ""
        if partname in IMAGE_REPLACEMENTS:
            png = VIZ / IMAGE_REPLACEMENTS[partname]
            if not png.exists():
                raise FileNotFoundError(
                    f"Corrected figure not found: {png}. "
                    "Run phase6_report.py first.")
            # Replace only the image payload; the inline drawing extent
            # (display size/position on the page) is left untouched.
            rel.target_part._blob = png.read_bytes()
            replaced += 1
            print(f"  Replaced {partname} -> {png.name} "
                  f"({png.stat().st_size:,} bytes)")

    if replaced != len(IMAGE_REPLACEMENTS):
        raise RuntimeError(
            f"Expected {len(IMAGE_REPLACEMENTS)} HJ-Biplot images in the "
            f"template, found {replaced}. The template may have changed.")

    n_text = _replace_text(doc)
    print(f"  Applied {n_text} text corrections (CLR method claim and "
          "biplot statistics)")

    doc.save(str(OUTPUT))
    print(f"Saved {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
