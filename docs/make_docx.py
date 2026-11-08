# -*- coding: utf-8 -*-
"""
make_docx.py
=============
Convert docs/Methodology_LDA.md into an academic-style Word document
(docs/Methodology_LDA.docx).

Style: Times New Roman, justified body, numbered headings, captioned
figures (embedded), grid tables, references at the end, page footer.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.shared import Inches, Pt, RGBColor
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = Path(__file__).resolve().parent.parent
MD_PATH = ROOT / "docs" / "Methodology_LDA.md"
DOCX_PATH = ROOT / "docs" / "Methodology_LDA.docx"

FONT = "Times New Roman"
ACCENT = RGBColor(0x1F, 0x3A, 0x5F)
GREY = RGBColor(0x44, 0x44, 0x44)


def _set_cell_bg(cell, hexcolor: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hexcolor)
    tc_pr.append(shd)


def _add_page_number(run) -> None:
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    run._r.append(fld)


INLINE_RE = re.compile(
    r"(\*\*.+?\*\*|\*[^*]+?\*|`[^`]+?`|\[[^\]]+?\]\([^)]+?\)|\$[^$]+?\$)"
)


def add_inline(paragraph, text: str, base_size: int = 11,
               italic_all: bool = False) -> None:
    """Add runs parsing **bold**, *italic*, `code`, [txt](url), $math$."""
    if not text:
        return
    parts = INLINE_RE.split(text)
    for part in parts:
        if not part:
            continue
        run = paragraph.add_run()
        run.font.name = FONT
        run.font.size = Pt(base_size)
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            run.text = part[2:-2]
            run.bold = True
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            run.text = part[1:-1]
            run.italic = True
        elif part.startswith("`") and part.endswith("`"):
            run.text = part[1:-1]
            run.font.name = "Consolas"
            run.font.size = Pt(9)
        elif part.startswith("[") and "](" in part:
            inner = part[1:-1]
            txt, url = inner.split("](", 1)
            run.text = txt
            run.underline = True
            run.font.color.rgb = ACCENT
        elif part.startswith("$") and part.endswith("$"):
            run.text = part[1:-1]
            run.italic = True
        else:
            run.text = part
        if italic_all:
            run.italic = True


def is_table_row(line: str) -> bool:
    return line.strip().startswith("|") and line.strip().endswith("|")


def split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def is_sep_row(line: str) -> bool:
    return bool(re.fullmatch(r"\|[\s\-\:\|]+\|", line.strip()))


def main() -> None:
    text = MD_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()

    doc = Document()

    # -- page setup: A4-ish margins -------------------------------------
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.1)
        section.right_margin = Inches(1.1)

    style = doc.styles["Normal"]
    style.font.name = FONT
    style.font.size = Pt(11)
    style.paragraph_format.space_after = Pt(6)
    style.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE

    for name, size, color in (("Heading 1", 14, ACCENT),
                              ("Heading 2", 12, ACCENT),
                              ("Heading 3", 11, ACCENT)):
        hs = doc.styles[name]
        hs.font.name = FONT
        hs.font.size = Pt(size)
        hs.font.bold = True
        hs.font.color.rgb = color
        hs.paragraph_format.space_before = Pt(14 if name == "Heading 1" else 10)
        hs.paragraph_format.space_after = Pt(6)
        hs.paragraph_format.keep_with_next = True

    # -- footer with page number -----------------------------------------
    footer = doc.sections[0].footer
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = fp.add_run("Page ")
    r.font.name = FONT
    r.font.size = Pt(9)
    r.font.color.rgb = GREY
    r2 = fp.add_run()
    r2.font.name = FONT
    r2.font.size = Pt(9)
    r2.font.color.rgb = GREY
    _add_page_number(r2)
    r3 = fp.add_run("  |  Methodology and preliminary results — LDA")
    r3.font.name = FONT
    r3.font.size = Pt(9)
    r3.font.color.rgb = GREY

    figure_no = 0
    i = 0
    title_done = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # --- code fence -------------------------------------------------
        if stripped.startswith("```"):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1
            for cl in code_lines:
                p = doc.add_paragraph()
                p.paragraph_format.space_after = Pt(1)
                p.paragraph_format.left_indent = Inches(0.25)
                run = p.add_run(cl)
                run.font.name = "Consolas"
                run.font.size = Pt(8.5)
            continue

        # --- headings ----------------------------------------------------
        m = re.match(r"^(#{1,3})\s+(.*)$", stripped)
        if m:
            level = len(m.group(1))
            heading = m.group(2).strip()
            if level == 1 and not title_done:
                title_done = True
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(18)
                r = p.add_run(heading)
                r.font.name = FONT
                r.font.size = Pt(16)
                r.bold = True
                r.font.color.rgb = ACCENT
                i += 1
                continue
            doc.add_heading(heading, level=level)
            i += 1
            continue

        # --- image --------------------------------------------------------
        m = re.match(r"^!\[(.*?)\]\((.*?)\)$", stripped)
        if m:
            alt, rel = m.group(1), m.group(2)
            img_path = (MD_PATH.parent / rel).resolve()
            if img_path.exists():
                figure_no += 1
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = p.add_run()
                run.add_picture(str(img_path), width=Inches(max(4.2, min(6.2, 6.2))))
                cap = doc.add_paragraph()
                cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                cr = cap.add_run(f"Figure {figure_no}. {alt}")
                cr.font.name = FONT
                cr.font.size = Pt(9)
                cr.italic = True
                cr.font.color.rgb = GREY
            i += 1
            continue

        # --- table --------------------------------------------------------
        if is_table_row(stripped) and i + 1 < len(lines) \
                and is_sep_row(lines[i + 1]):
            header = split_row(stripped)
            rows = []
            i += 2
            while i < len(lines) and is_table_row(lines[i]):
                rows.append(split_row(lines[i]))
                i += 1
            table = doc.add_table(rows=1, cols=len(header))
            table.style = "Table Grid"
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            table.autofit = True
            hdr = table.rows[0].cells
            for j, h in enumerate(header):
                hdr[j].text = ""
                add_inline(hdr[j].paragraphs[0], h, base_size=9)
                for run in hdr[j].paragraphs[0].runs:
                    run.bold = True
                _set_cell_bg(hdr[j], "DCE6F2")
            for row in rows:
                cells = table.add_row().cells
                for j in range(len(header)):
                    val = row[j] if j < len(row) else ""
                    cells[j].text = ""
                    add_inline(cells[j].paragraphs[0], val, base_size=9)
            # smaller font for tables
            for row in table.rows:
                for cell in row.cells:
                    for p in cell.paragraphs:
                        for r in p.runs:
                            r.font.size = Pt(9)
            doc.add_paragraph()
            continue

        # --- blockquote ----------------------------------------------------
        if stripped.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip("> ").strip())
                i += 1
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.3)
            p.paragraph_format.right_indent = Inches(0.3)
            add_inline(p, " ".join(buf), italic_all=True)
            continue

        # --- horizontal rule ------------------------------------------------
        if stripped == "---":
            i += 1
            continue

        # --- list items ------------------------------------------------------
        m = re.match(r"^\d+\.\s+(.*)$", stripped)
        if m:
            p = doc.add_paragraph(style="List Number")
            add_inline(p, m.group(1))
            i += 1
            continue
        if stripped.startswith("- "):
            p = doc.add_paragraph(style="List Bullet")
            add_inline(p, stripped[2:])
            i += 1
            continue

        # --- blank ------------------------------------------------------------
        if not stripped:
            i += 1
            continue

        # --- metadata line (starts with **Corpus:** etc.) -------------------
        # --- paragraph --------------------------------------------------------
        buf = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i]
            nstr = nxt.strip()
            if not nstr or nstr.startswith(("#", "-", ">", "!", "```")) \
                    or re.match(r"^\d+\.\s", nstr) or is_table_row(nstr):
                break
            buf.append(nstr)
            i += 1
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        add_inline(p, " ".join(buf))

    doc.save(str(DOCX_PATH))
    print(f"Saved {DOCX_PATH} ({DOCX_PATH.stat().st_size:,} bytes, "
          f"{figure_no} figures)")


if __name__ == "__main__":
    main()
