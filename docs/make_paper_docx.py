# -*- coding: utf-8 -*-
"""
make_paper_docx.py
==================
Rebuild the final MDPI paper DOCX with:
  1. The corrected CLR-transformed HJ-Biplot figures (and PRISMA figure).
  2. Text, captions and tables consistent with the CLR plots.
  3. Word OMML equations converted to plain Unicode so they survive PDF export.
  4. The Author Contributions placeholder resolved.

The original manuscript (docs/Information_MDPI_Manuscript.docx) is the template:
styles/layout are preserved; only the items above are edited.

Usage:
    uv run python docs/make_paper_docx.py
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "docs" / "Information_MDPI_Manuscript.docx"
OUTPUT = ROOT / "docs" / "2611.00002.docx"
PDF_OUTPUT = ROOT / "docs" / "2611.00002.pdf"
VIZ = ROOT / "outputs" / "visualizations"

# Image part name in the template -> corrected PNG produced by phase6_report.py
IMAGE_REPLACEMENTS = {
    # PRISMA flow (Figure 1) - regenerated with "Section 2.1" in the caption
    "/word/media/d10d8a461a0a1cb48493713d081ca685c708adaf.png": "PRISMA_Flow.png",
    # years HJ-Biplot (Figure 5)
    "/word/media/265bd2f715d909d7e9a833bc0ce79788e0f472a0.png": "LDA_HJBiplot_Years.png",
    # countries HJ-Biplot (Figure 6)
    "/word/media/1d43a3d0589aa8451d6e3026b4250ad1d29d7365.png": "LDA_HJBiplot_Countries.png",
    # documents HJ-Biplot
    "/word/media/046363f4c9a04c609055a13473b6994b655c14d7.png": "LDA_HJBiplot_Documents.png",
}

TEXT_REPLACEMENTS = {
    # --- Methods: CLR applied instead of "no transformation" -------------
    " (no further scaling or log-ratio transformation was applied), "
    "the singular value decomposition is":
        " after centered log-ratio (CLR) transformation of the topic "
        "proportions, the singular value decomposition is",
    "each cell is the mean topic probability":
        "each cell is the CLR-transformed mean topic proportion",
    # --- Results: Table 15 description numbers ----------------------------
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
    # --- Section 3.5: rewrite both interpretation paragraphs for CLR ------
    "In the years biplot, axis 1 is a temporal gradient that orders the years "
    "chronologically from 2022 (−0.145) to 2026 (+0.117). The early pole is "
    "associated with T1 (human problem-solving), text detection (T19), "
    "generative AI ethics (T3), bias (T8), and training (T14), and the recent "
    "pole with retrieval and knowledge graphs (T20), governance frameworks "
    "(T2), evaluation (T13), agentic systems (T7), ChatGPT response "
    "evaluation (T5), and clinical applications (T9). Axis 2 separates "
    "2023–2024 from the two extreme years and is associated with application "
    "challenges (T23), ChatGPT-based literature reviews (T25), and education "
    "(T27), which is typical of an arch pattern in ordered data. The 2022 "
    "point rests on only 31 documents, and T11 and T17 are poorly represented "
    "(squared cosines of 0.23 and 0.44).":
        "In the CLR-transformed years biplot, axis 1 is a temporal gradient "
        "that orders the years chronologically from 2022 (−4.79) to 2026 "
        "(+1.93). The early pole is associated with T1 (human problem-solving), "
        "multilingual corpora (T24), value alignment (T22), bias (T8) and "
        "training (T14), whereas the recent pole is dominated by ChatGPT "
        "response evaluation (T5, the longest vector), privacy (T4), clinical "
        "applications (T9), agentic systems (T7), retrieval and knowledge "
        "graphs (T20) and robotics/digital twins (T21). Axis 2 separates 2026 "
        "(+2.02) from 2023 (−1.85) and 2024 (−1.06), and is associated with "
        "agentic systems (T7), governance frameworks (T2) and evaluation "
        "(T13) at its positive pole, and with human problem-solving (T1), "
        "literature reviews (T25) and education (T27) at its negative pole. "
        "The 2022 point rests on only 31 documents; its extreme position "
        "should therefore be interpreted with caution.",
    "In the countries biplot, axis 1 is dominated by a single association: "
    "Turkey (+0.241; squared cosine 0.998) with topic T5, the evaluation of "
    "ChatGPT responses, and to a lesser extent T9 and T13. Axis 2 contrasts "
    "China (−0.123; squared cosine 0.915), South Korea, and Japan, which are "
    "aligned with retrieval (T20), training (T14), intelligent management "
    "(T15), and multimodal models (T12), with the United Kingdom, Australia, "
    "Germany, Italy, Canada, Spain, and the United States, which are aligned "
    "with generative AI ethics (T3), human problem-solving (T1), governance "
    "(T2), and law (T26). Most countries other than China and Turkey lie "
    "close to the origin with squared cosines below 0.6 (for example, India "
    "0.32, Japan 0.37, and Spain 0.42), so their relative positions should "
    "not be interpreted.":
        "In the CLR-transformed countries biplot, axis 1 is dominated by China "
        "(+2.44; squared cosine 0.87), aligned with retrieval and knowledge "
        "graphs (T20), training (T14) and multimodal models (T12); South "
        "Korea (+1.16) and Japan (+0.85) lie on the same side. The negative "
        "pole gathers most Western countries (United Kingdom, Australia, "
        "Germany, Italy, Canada, Spain and the United States), aligned with "
        "generative AI ethics (T3), human problem-solving (T1), governance "
        "(T2) and law (T26). Axis 2 is dominated by Turkey (+2.57; squared "
        "cosine 0.97), aligned with ChatGPT response evaluation (T5), "
        "clinical applications (T9) and evaluation (T13), whereas China, "
        "South Korea and Japan lie near the centre of this axis. Most "
        "countries other than China and Turkey lie close to the origin with "
        "squared cosines below 0.6 (for example, the United States 0.18, "
        "India 0.15 and Spain 0.26), so their relative positions should not "
        "be interpreted.",
    # --- Figure 5 caption --------------------------------------------------
    "Figure 5. HJ-Biplot of years × topics (mean topic probability, "
    "column-centered).":
        "Figure 5. HJ-Biplot of years × topics (CLR-transformed topic "
        "proportions, column-centered).",
    # --- Author Contributions placeholder ----------------------------------
    "Author Contributions: [To be completed by the authors using the CRediT "
    "taxonomy, e.g., Conceptualization, J.V.; methodology, J.V.; software, "
    "J.V.; validation, J.V.; formal analysis, J.V.; investigation, J.V.; data "
    "curation, J.V.; writing—original draft preparation, J.V.; writing—review "
    "and editing, J.V.; visualization, J.V.; supervision, [name]. All authors "
    "have read and agreed to the published version of the manuscript.]":
        "Author Contributions: Conceptualization, J.V.; methodology, J.V.; "
        "software, J.V.; validation, J.V.; formal analysis, J.V.; "
        "investigation, J.V.; data curation, J.V.; writing—original draft "
        "preparation, J.V.; writing—review and editing, J.V.; visualization, "
        "J.V.; supervision, J.V. All authors have read and agreed to the "
        "published version of the manuscript.",
    # --- Limitations: the flaw is now fixed --------------------------------
    "Biplot quality. The countries and documents biplots have moderate and "
    "low representation quality, respectively, and no log-ratio "
    "transformation was applied.":
        "Biplot quality. The countries and documents biplots have moderate "
        "and low representation quality, respectively, after the centered "
        "log-ratio transformation.",
    # --- Discussion (Section 4.3): the CLR transformation was APPLIED ------
    "The biplots were computed on column-centered mean topic probabilities. "
    "Because topic proportions are compositional data, log-ratio "
    "transformations are generally recommended for biplots [40]; the present "
    "biplots should therefore be regarded as exploratory, and repeating them "
    "after a log-ratio transformation is a priority.":
        "The biplots were computed on column-centered, centered log-ratio "
        "(CLR) transformed mean topic proportions. Because topic proportions "
        "are compositional data confined to the simplex, the CLR map [40] was "
        "applied before column centering, so the HJ-Biplot approximation is "
        "computed in the appropriate Aitchison geometry rather than on "
        "constrained raw proportions. Representation quality remains high "
        "for the years biplot but is lower for the countries and documents "
        "biplots, which should therefore be regarded as exploratory.",
    # --- Conclusions (Section 5): no longer a pending recommendation -------
    "to repeat the biplots with a log-ratio transformation, and to compare "
    "LDA with embedding-based topic models":
        "to compare LDA with embedding-based topic models",
}

# Table 16 ("Biplot, axis / Negative pole / Positive pole"): overwrite the
# four data rows with the poles read from the new CLR coordinates.
TABLE_15_CELLS = [
    ["Years, axis 1 (74.7%)",
     "2022; T1, T24, T22, T8, T14",
     "2026, 2025; T5, T4, T9, T7, T20, T21"],
    ["Years, axis 2 (22.8%)",
     "2023, 2024; T1, T25, T5, T23, T3, T27",
     "2026, 2022; T7, T2, T13, T15, T21, T20"],
    ["Countries, axis 1 (32.8%)",
     "UK, Australia, Germany, Italy, Canada, Spain, United States; "
     "T3, T1, T2, T26",
     "China; T20, T14, T12, T15, T10, T21"],
    ["Countries, axis 2 (26.7%)",
     "United States, India; T1, T3, T17, T4",
     "Turkey; T5, T9, T13"],
]

TABLE_TEXT_REPLACEMENTS = {
    "77.5 / 18.1": "74.7 / 22.8",
    "0.900": "0.931", "0.883": "0.952",
    "59.7 / 23.1": "32.8 / 26.7",
    "0.562": "0.434", "0.461": "0.468",
    "10.3 / 8.2": "9.2 / 7.7",
    "0.176": "0.169", "0.108": "0.133",
}

# Word OMML equations render blank in LibreOffice PDF export. Replace each
# with plain Unicode text (keyed by the equation number in the paragraph).
EQUATION_TEXT = {
    "(1)": "θ_d ~ Dirichlet(α)",
    "(2)": "z_{d,n} ~ Categorical(θ_d)",
    "(3)": "w_{d,n} ~ Categorical(φ_{z_{d,n}})",
    "(4)": "φ_k ~ Dirichlet(η)",
    "(5)": "NPMI(w_i, w_j) = log[P(w_i,w_j)/(P(w_i)P(w_j))] / (−log P(w_i,w_j))",
    "(6)": "H̃ = −Σ_{k=1}^{K} p_k log p_k / log K",
    "(7)": "J(A,B) = |A ∩ B| / |A ∪ B|",
    "(8)": "X_c = U · D · Vᵀ",
    "(9)": "cos²_i = (a_{i1}² + a_{i2}²) / Σ_k a_{ik}²",
    "(10)": "r(w|t) = λ·log p(w|t) + (1−λ)·log[p(w|t)/p(w)]",
}


def _replace_text(doc: Document) -> int:
    """Apply text fixes at run level, preserving all formatting."""
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

        # Some long replacements (e.g. the Author Contributions placeholder)
        # are split across several runs in the DOCX. They never match a single
        # run above, so apply them at paragraph level as a fallback: put the
        # corrected text in the first run and empty the rest.
        whole = paragraph.text
        fixed = whole
        for old, replacement in TEXT_REPLACEMENTS.items():
            if old in fixed:
                fixed = fixed.replace(old, replacement)
                count += 1
        if fixed != whole and paragraph.runs:
            paragraph.runs[0].text = fixed
            for run in paragraph.runs[1:]:
                run.text = ""
            count += 1
    for table in doc.tables:
        # Detect the poles table by its header.
        header = " | ".join(c.text for c in table.rows[0].cells)
        is_poles_table = "Negative pole" in header and "Positive pole" in header
        if is_poles_table:
            for ri, row_values in enumerate(TABLE_15_CELLS, start=1):
                for ci, value in enumerate(row_values):
                    cell = table.rows[ri].cells[ci]
                    for p in cell.paragraphs:
                        for run in list(p.runs):
                            run._r.getparent().remove(run._r)
                    cell.paragraphs[0].add_run(value)
                    count += 1
            continue
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


def _replace_equations(doc: Document) -> int:
    """Convert OMML equations to plain Unicode text that exports to PDF."""
    ns = {"m": "http://schemas.openxmlformats.org/officeDocument/2006/math"}
    count = 0
    for paragraph in doc.paragraphs:
        if not paragraph._p.findall(".//m:oMath", ns):
            continue
        label = paragraph.text.strip()
        if label not in EQUATION_TEXT:
            continue
        formula = EQUATION_TEXT[label]
        # Remove all OMML elements from the paragraph.
        for omath in paragraph._p.findall(".//m:oMath", ns):
            omath.getparent().remove(omath)
        # Remove leftover text runs (tabs / equation number), then write a
        # single centered run with the formula followed by its number.
        for run in list(paragraph.runs):
            run._r.getparent().remove(run._r)
        run = paragraph.add_run(f"{formula}\t\t{label}")
        run.font.name = "Times New Roman"
        run.font.size = paragraph.style.font.size or None
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
            rel.target_part._blob = png.read_bytes()
            replaced += 1
            print(f"  Replaced {partname} -> {png.name} "
                  f"({png.stat().st_size:,} bytes)")

    if replaced != len(IMAGE_REPLACEMENTS):
        raise RuntimeError(
            f"Expected {len(IMAGE_REPLACEMENTS)} figures in the template, "
            f"found {replaced}. The template may have changed.")

    n_text = _replace_text(doc)
    print(f"  Applied {n_text} text/table corrections")
    n_eq = _replace_equations(doc)
    print(f"  Converted {n_eq} OMML equations to Unicode text")

    doc.save(str(OUTPUT))
    print(f"Saved {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")

    # Export to PDF with LibreOffice (headless) so equations/figures render.
    import subprocess
    subprocess.run(
        ["libreoffice", "--headless", "--convert-to", "pdf",
         "--outdir", str(OUTPUT.parent), str(OUTPUT)],
        check=True, capture_output=True,
    )
    generated_pdf = OUTPUT.with_suffix(".pdf")
    if generated_pdf != PDF_OUTPUT:
        PDF_OUTPUT.write_bytes(generated_pdf.read_bytes())
        generated_pdf.unlink()
    print(f"Saved {PDF_OUTPUT} ({PDF_OUTPUT.stat().st_size:,} bytes)")

    # Keep the site PDF in sync (research.vyde.dev/public/pdfs/2611.00002.pdf)
    site_pdf = (ROOT.parent / "research.vyde.dev" / "public" / "pdfs"
                / "2611.00002.pdf")
    if site_pdf.parent.exists():
        import shutil
        shutil.copy2(PDF_OUTPUT, site_pdf)
        print(f"  Synced site PDF -> {site_pdf}")


if __name__ == "__main__":
    main()
