# -*- coding: utf-8 -*-
"""
make_paper_docx.py
==================
Build the final paper (2611.00002) completely from scratch -- NO template.

All text and tables come from `paper_content.py`; all figures come from
`outputs/visualizations/`. The generator writes the DOCX, exports it to PDF
with LibreOffice, and syncs the PDF to the research site.

Usage:
    uv run python docs/make_paper_docx.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
VIZ = ROOT / "outputs" / "visualizations"
OUTPUT = ROOT / "docs" / "2611.00002.docx"
PDF_OUTPUT = ROOT / "docs" / "2611.00002.pdf"
SITE_PDF = (ROOT.parent / "research.vyde.dev" / "public" / "pdfs"
            / "2611.00002.pdf")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_content import PAPER_ITEMS  # noqa: E402

FONT = "Times New Roman"
ALIGNMENTS = {
    "CENTER (1)": WD_ALIGN_PARAGRAPH.CENTER,
    "RIGHT (2)": WD_ALIGN_PARAGRAPH.RIGHT,
    "LEFT (0)": WD_ALIGN_PARAGRAPH.LEFT,
    "JUSTIFY (3)": WD_ALIGN_PARAGRAPH.JUSTIFY,
}


def _add_run(paragraph, text: str, bold: bool = False,
             italic: bool = False, size: float | None = None) -> None:
    run = paragraph.add_run(text)
    run.font.name = FONT
    if size is not None:
        run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic


def _build_styles(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    for name, size in (("Heading 1", 15), ("Heading 2", 12)):
        st = doc.styles[name]
        st.font.name = FONT
        st.font.size = Pt(size)
        st.font.color.rgb = RGBColor(0, 0, 0)
        st.paragraph_format.space_before = Pt(12)
        st.paragraph_format.space_after = Pt(6)
        st.paragraph_format.keep_with_next = True


def _add_figure(doc: Document, filename: str) -> None:
    png = VIZ / filename
    if not png.exists():
        raise FileNotFoundError(f"Figure not found: {png}")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(png), width=Inches(6.3))


def _add_table(doc: Document, rows: list[list[str]]) -> None:
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Table Grid"
    for ri, row in enumerate(rows):
        for ci, value in enumerate(row):
            cell = table.cell(ri, ci)
            cell.text = ""
            para = cell.paragraphs[0]
            _add_run(para, value, bold=(ri == 0), size=9)
    doc.add_paragraph()


def main() -> None:
    doc = Document()

    # Page setup matching the manuscript (A4, one column).
    for section in doc.sections:
        section.top_margin = Inches(1.04)
        section.bottom_margin = Inches(0.97)
        section.left_margin = Inches(1.18)
        section.right_margin = Inches(1.18)
    _build_styles(doc)

    figures = 0
    for item in PAPER_ITEMS:
        if item["type"] == "p":
            if item.get("img"):
                _add_figure(doc, item["img"])
                figures += 1
                continue
            text = item.get("text", "")
            p = doc.add_paragraph(style=item.get("style") or "Normal")
            align = item.get("align")
            if align in ALIGNMENTS:
                p.alignment = ALIGNMENTS[align]
            if text:
                # Preserve leading tabs before equation numbers.
                _add_run(p, text)
        elif item["type"] == "tbl":
            _add_table(doc, item["rows"])

    doc.save(OUTPUT)
    print(f"Saved {OUTPUT} ({OUTPUT.stat().st_size:,} bytes) "
          f"with {figures} figures, {len(doc.tables)} tables, "
          f"{len(doc.paragraphs)} paragraphs")

    # Export to PDF.
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

    if SITE_PDF.parent.exists():
        shutil.copy2(PDF_OUTPUT, SITE_PDF)
        print(f"  Synced site PDF -> {SITE_PDF}")


if __name__ == "__main__":
    main()
