"""
report/build_report.py
Builds the final project report (MMMUT B.Tech format, following the
department's sample final report) from report/content.txt and the figures in
report/figures/.  Microsoft Word is then used (COM) to fill in the table of
contents, list of figures, list of tables and page numbers, and to export PDF.

    python report/build_report.py <logo.png> <out_docx> [<lists.json>]

Without lists.json the contents / figure / table lists are Word TOC fields
(placeholders). report/make_report.py renders a draft with Word, reads the
page of every heading and caption from that PDF, and rebuilds the document
with those lists written out (Word on the build machine could export PDF but
could not save documents).
"""
import json
import os
import re
import sys

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
LOGO, OUT = sys.argv[1], sys.argv[2]
LISTS = json.load(open(sys.argv[3], encoding="utf-8")) if len(sys.argv) > 3 else None

TITLE = "Adaptive AI-Driven Intrusion Detection and Protection System using Federated Learning (A-IDAPS-FL)"
STUDENTS = [("Tarkeshvar Mani Yadav", "2023011078"), ("Vishal Kumar", "2023011085"),
            ("Shashwat Srivastava", "2023021159"), ("Ritesh Pandey", "2023021154")]
SUPERVISOR, SUP_DESIG = "Dr. Raj Kumar", "Assistant Professor"
HOD = "Prof. Rakesh Kumar"
MONTH = "November 2026"
FONT = "Times New Roman"

doc = Document()


# ------------------------------------------------------------------ helpers --
def set_cell_shading(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def page_border(section):
    sectPr = section._sectPr
    pb = OxmlElement("w:pgBorders"); pb.set(qn("w:offsetFrom"), "page")
    for side in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:val"), "single"); el.set(qn("w:sz"), "12"); el.set(qn("w:space"), "24"); el.set(qn("w:color"), "000000")
        pb.append(el)
    sectPr.append(pb)


def page_number_format(section, fmt, start=None):
    sectPr = section._sectPr
    pg = sectPr.find(qn("w:pgNumType"))
    if pg is None:
        pg = OxmlElement("w:pgNumType"); sectPr.append(pg)
    pg.set(qn("w:fmt"), fmt)
    if start is not None:
        pg.set(qn("w:start"), str(start))


def add_field(paragraph, instr, placeholder=""):
    r = paragraph.add_run()
    b = OxmlElement("w:fldChar"); b.set(qn("w:fldCharType"), "begin"); r._r.append(b)
    r2 = paragraph.add_run()
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = instr; r2._r.append(it)
    r3 = paragraph.add_run()
    s = OxmlElement("w:fldChar"); s.set(qn("w:fldCharType"), "separate"); r3._r.append(s)
    paragraph.add_run(placeholder)
    r5 = paragraph.add_run()
    e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), "end"); r5._r.append(e)


def footer_page_number(section, show=True):
    section.footer.is_linked_to_previous = False
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if show:
        add_field(p, "PAGE")


def para(text="", style=None, align=None, size=None, bold=False, italic=False, before=None, after=None,
         spacing=None, keep_next=False):
    p = doc.add_paragraph(style=style)
    if text:
        r = p.add_run(text); r.bold = True if bold else None; r.italic = True if italic else None
        if size:
            r.font.size = Pt(size)
    if align is not None:
        p.alignment = align
    pf = p.paragraph_format
    if before is not None:
        pf.space_before = Pt(before)
    if after is not None:
        pf.space_after = Pt(after)
    if spacing is not None:
        pf.line_spacing = spacing
    if keep_next:
        pf.keep_with_next = True
    return p


def rich(p, text):
    """**bold** spans inside a paragraph."""
    for i, part in enumerate(re.split(r"\*\*", text)):
        if part:
            p.add_run(part).bold = (i % 2 == 1)
    return p


def static_list(key, field_instr):
    """Written-out list (from LISTS) or, for the draft, a Word TOC field."""
    if not LISTS:
        add_field(para(align=WD_ALIGN_PARAGRAPH.LEFT), field_instr, "Right-click and update field.")
        return
    for level, text, page in LISTS[key]:
        p = para(align=WD_ALIGN_PARAGRAPH.LEFT, after=1, spacing=1.0)
        pf = p.paragraph_format
        pf.tab_stops.add_tab_stop(Inches(6.5), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
        if level == 2:
            pf.left_indent = Inches(0.3)
        major = key == "toc" and level == 1 and text[:7] in ("Chapter", "Referen")
        if major:
            pf.space_before = Pt(4)
        r = p.add_run(text); r.bold = major; r.font.size = Pt(11)
        r2 = p.add_run("\t" + page); r2.bold = major; r2.font.size = Pt(11)


def page_break():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def no_borders(table):
    tblPr = table._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{side}"); el.set(qn("w:val"), "nil"); b.append(el)
    tblPr.append(b)


# ------------------------------------------------------------------- styles --
st = doc.styles
normal = st["Normal"]
normal.font.name = FONT; normal.font.size = Pt(12)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.5
normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

for name, size, align, before, after in (("Heading 1", 16, WD_ALIGN_PARAGRAPH.CENTER, 0, 24),
                                         ("Heading 2", 14, WD_ALIGN_PARAGRAPH.LEFT, 18, 6),
                                         ("Heading 3", 12, WD_ALIGN_PARAGRAPH.LEFT, 12, 4)):
    h = st[name]
    h.font.name = FONT; h.font.size = Pt(size); h.font.bold = True; h.font.italic = False
    h.font.color.rgb = RGBColor(0, 0, 0)
    h.element.rPr.rFonts.set(qn("w:eastAsia"), FONT); h.element.rPr.rFonts.set(qn("w:ascii"), FONT)
    h.element.rPr.rFonts.set(qn("w:hAnsi"), FONT)
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        h.element.rPr.rFonts.attrib.pop(qn(a), None)
    pf = h.paragraph_format
    pf.alignment = align; pf.space_before = Pt(before); pf.space_after = Pt(after); pf.keep_with_next = True
    pf.line_spacing = 1.15
st["Heading 1"].font.all_caps = True

front = st.add_style("Front Heading", WD_STYLE_TYPE.PARAGRAPH)
front.base_style = normal
front.font.size = Pt(14); front.font.bold = True; front.font.all_caps = True
front.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
front.paragraph_format.space_after = Pt(18)
ol = OxmlElement("w:outlineLvl"); ol.set(qn("w:val"), "0"); front.element.get_or_add_pPr().append(ol)

for name, bold in (("Figure Caption", False), ("Table Caption", True)):
    c = st.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
    c.base_style = normal
    c.font.size = Pt(11); c.font.bold = bold
    c.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.paragraph_format.space_before = Pt(4); c.paragraph_format.space_after = Pt(10)
    c.paragraph_format.line_spacing = 1.0
st["Table Caption"].paragraph_format.keep_with_next = True

code = st.add_style("Algorithm Code", WD_STYLE_TYPE.PARAGRAPH)
code.base_style = normal
code.font.name = "Consolas"; code.font.size = Pt(9.5)
code.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
code.paragraph_format.line_spacing = 1.0; code.paragraph_format.space_after = Pt(0)

for name in ("List Bullet",):
    lb = st[name]; lb.font.name = FONT; lb.font.size = Pt(12)
    lb.paragraph_format.space_after = Pt(3); lb.paragraph_format.line_spacing = 1.5

# ------------------------------------------------------------ page settings --
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
for m in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, m, Inches(1))
sec.footer_distance = Inches(0.6)
sec.different_first_page_header_footer = True
page_border(sec)
page_number_format(sec, "lowerRoman", 1)
footer_page_number(sec)

# -------------------------------------------------------------------- cover --
p = para(TITLE, align=WD_ALIGN_PARAGRAPH.CENTER, size=18, bold=True, italic=True, after=2, spacing=1.15)
pPr = p._p.get_or_add_pPr(); bdr = OxmlElement("w:pBdr"); bot = OxmlElement("w:bottom")
bot.set(qn("w:val"), "single"); bot.set(qn("w:sz"), "12"); bot.set(qn("w:space"), "4"); bot.set(qn("w:color"), "000000")
bdr.append(bot); pPr.append(bdr)
for t in ("A Project Report Submitted", "In Partial Fulfillment of the Requirements", "for the Degree of"):
    para(t, align=WD_ALIGN_PARAGRAPH.CENTER, italic=True, after=0, before=0, spacing=1.15)
para("BACHELOR OF TECHNOLOGY", align=WD_ALIGN_PARAGRAPH.CENTER, size=14, bold=True, before=8, after=0)
para("in", align=WD_ALIGN_PARAGRAPH.CENTER, size=14, after=0)
para("COMPUTER SCIENCE AND ENGINEERING", align=WD_ALIGN_PARAGRAPH.CENTER, size=14, bold=True, after=0)
para("By", align=WD_ALIGN_PARAGRAPH.CENTER, size=14, after=0)
for n, r in STUDENTS:
    para(f"{n} ({r})", align=WD_ALIGN_PARAGRAPH.CENTER, after=0, spacing=1.15)
para("Project Group G-52  |  Session 2026-27", align=WD_ALIGN_PARAGRAPH.CENTER, after=0, before=4, size=11)
para("Under the supervision of", align=WD_ALIGN_PARAGRAPH.CENTER, before=10, after=0)
para(SUPERVISOR, align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, after=0)
para(f"({SUP_DESIG})", align=WD_ALIGN_PARAGRAPH.CENTER, after=4)
lp = para(align=WD_ALIGN_PARAGRAPH.CENTER, after=4); lp.add_run().add_picture(LOGO, width=Inches(1.7))
for t in ("DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING", "Madan Mohan Malaviya University of Technology, Gorakhpur (U.P.)",
          "INDIA", MONTH):
    para(t, align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, after=0, spacing=1.15)
page_break()

# ---------------------------------------------------------------- copyright --
cp = para(spacing=1.5)
for _ in range(22):
    cp.add_run().add_break()
para("© M. M. M. University of Technology, Gorakhpur (U.P.) – 273010, INDIA", align=WD_ALIGN_PARAGRAPH.CENTER,
     bold=True, after=0)
para("ALL RIGHTS RESERVED", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True)
page_break()

names_rolls = ", ".join(f"{n} ({r})" for n, r in STUDENTS[:-1]) + f" and {STUDENTS[-1][0]} ({STUDENTS[-1][1]})"

# -------------------------------------------------------------- certificate --
para("Certificate", style="Front Heading", before=12)
para(f"Certified that {names_rolls} have carried out the project work presented in this report entitled "
     f"“{TITLE}” for the award of Bachelor of Technology from Madan Mohan Malaviya University of Technology, "
     f"Gorakhpur under my supervision. The report embodies results of original work and studies carried out by the "
     f"students themselves and the contents of the report do not form the basis for the award of any other degree "
     f"to the candidates or to anybody else.")
para("", before=60)
for t, b in (("______________________", False), (SUPERVISOR, True), (SUP_DESIG, False),
             ("Computer Science and Engineering Department", False), ("M.M.M.U.T. Gorakhpur – 273010", False)):
    para(t, align=WD_ALIGN_PARAGRAPH.RIGHT, bold=b, after=0, spacing=1.15)
para("Date: ……………………", before=150, align=WD_ALIGN_PARAGRAPH.LEFT)
page_break()

# -------------------------------------------------------------- declaration --
para("Candidate’s Declaration", style="Front Heading", before=12)
para("We declare that this written submission represents our work and ideas in our own words and where others’ "
     "ideas or words have been included, we have adequately cited and referenced the original sources. We also declare "
     "that we have adhered to all principles of academic honesty and integrity and have not misrepresented or fabricated "
     "or falsified any idea/data/fact/source in our submission. We understand that any violation of the above will be "
     "cause for disciplinary action by the University and can also evoke penal action from the sources which have thus "
     "not been properly cited or from whom proper permission has not been taken when needed.")
para("", before=18)
t = doc.add_table(rows=len(STUDENTS) + 1, cols=3); no_borders(t); t.alignment = WD_TABLE_ALIGNMENT.CENTER
for j, h in enumerate(("Name", "Roll No.", "Signature")):
    c = t.rows[0].cells[j]; c.paragraphs[0].add_run(h).bold = True
for i, (n, r) in enumerate(STUDENTS, 1):
    for j, v in enumerate((n, r, "………………………")):
        t.rows[i].cells[j].paragraphs[0].add_run(v)
for row in t.rows:
    for c in row.cells:
        c.paragraphs[0].paragraph_format.space_after = Pt(10); c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT
para("", before=24)
for tx in ("Computer Science and Engineering Department,", "Madan Mohan Malaviya University of Technology, Gorakhpur (U.P.)",
           "Date: ……………………"):
    para(tx, align=WD_ALIGN_PARAGRAPH.LEFT, after=0)
page_break()

# ----------------------------------------------------------------- approval --
para("Approval Sheet", style="Front Heading", before=12)
para(f"This project report entitled “{TITLE}” by {names_rolls} is approved for the degree of Bachelor of "
     f"Technology in Computer Science and Engineering.")
for role, name in (("Examiner", None), ("Supervisor", SUPERVISOR), ("Head of Department", HOD)):
    para("", before=30)
    para(role, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, after=0)
    para("……………………………………", align=WD_ALIGN_PARAGRAPH.LEFT, after=0, before=24)
    if name:
        para(f"({name})", align=WD_ALIGN_PARAGRAPH.LEFT, after=0)
para("Date: ……………………", before=36, align=WD_ALIGN_PARAGRAPH.LEFT, after=0)
para("Place: MMMUT Gorakhpur", align=WD_ALIGN_PARAGRAPH.LEFT)
page_break()

# ---------------------------------------------------------- acknowledgement --
para("Acknowledgement", style="Front Heading", before=12)
for tx in (
    f"We express our sincere gratitude to our supervisor, {SUPERVISOR}, {SUP_DESIG}, Department of Computer Science and "
    f"Engineering, for his guidance, encouragement and critical review throughout the three phases of this project. "
    f"His questions about how far our results could be trusted shaped much of the final work.",
    f"We are grateful to {HOD}, Head of the Department of Computer Science and Engineering, for providing the facilities "
    f"and support needed to carry out this project, and to all faculty members and staff of the department.",
    "We thank the Canadian Institute for Cybersecurity at the University of New Brunswick and the Communications Security "
    "Establishment for making the CIC-IDS2017 and CSE-CIC-IDS2018 datasets publicly available, and the developers of "
    "Flower, TensorFlow, XGBoost, Scapy, Npcap and Streamlit, on whose open-source work this project is built.",
    "Finally, we thank our families and friends for their constant support."):
    para(tx)
para("", before=24)
para("Team Members", bold=True, align=WD_ALIGN_PARAGRAPH.RIGHT, after=0)
for n, r in STUDENTS:
    para(f"{n} ({r})", align=WD_ALIGN_PARAGRAPH.RIGHT, after=0, spacing=1.15)
page_break()

# ----------------------------------------------------- lists of figs / tabs --
para("List of Figures", style="Front Heading", before=12)
static_list("lof", 'TOC \\h \\z \\t "Figure Caption,1"')
page_break()
para("List of Tables", style="Front Heading", before=12)
static_list("lot", 'TOC \\h \\z \\t "Table Caption,1"')
page_break()

# ----------------------------------------------------------------- abstract --
para("Abstract", style="Front Heading", before=12)
for n, r in STUDENTS:
    para(f"{n} ({r})", align=WD_ALIGN_PARAGRAPH.CENTER, size=13, after=0, spacing=1.15)
para("Under the guidance of", align=WD_ALIGN_PARAGRAPH.CENTER, before=8, after=0)
para(SUPERVISOR, align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, after=14)
para("Machine-learning intrusion detection needs large amounts of labelled network traffic, but collecting the traffic of "
     "many organisations in one place is a privacy problem. This project builds A-IDAPS-FL, an intrusion detection and "
     "protection system whose hybrid model, an autoencoder for anomalies and an XGBoost classifier for known attacks, is "
     "trained with federated learning. Four simulated network nodes, each seeing different attacks, train locally with the "
     "Flower framework and exchange only model parameters over TLS. The server aggregates them, calibrates the decision "
     "thresholds on its own hold-out with a family-aware rule and Platt scaling, and publishes a global model. On "
     "283,074 held-out flows of CIC-IDS2017 the federated model reaches 99.28% accuracy, 99.94% recall and a 0.88% "
     "false-positive rate, against 99.31%, 100% and 0.86% for a centralised model that needed 2.26 million raw flows "
     "uploaded. Each alert is explained with SHAP and a LIME-style surrogate, logged, and, when confirmed, blocked at the "
     "host firewall, which was verified on the real Windows firewall. Testing beyond the dataset found seven defects in the "
     "live capture path, including one that hid port scans; after the fixes and a local baseline, live false alarms fell "
     "from 74.7% to 0.9% of benign packets. It also showed that the supervised model learned the 2017 test network's "
     "attacker machine rather than scanning behaviour, so a behavioural port-scan detector was added, and that the model "
     "detects only 4.2% of the attacks in CSE-CIC-IDS2018. These negative results mark where the dataset accuracy stops "
     "being valid and motivate federated training with more diverse data.")
para("Keywords: intrusion detection, federated learning, autoencoder, XGBoost, calibration, explainable AI, "
     "CIC-IDS2017, CSE-CIC-IDS2018", italic=True, before=6)
page_break()

# --------------------------------------------------------------------- TOC --
para("Table of Contents", style="Front Heading", before=12)
p = para(align=WD_ALIGN_PARAGRAPH.LEFT, after=4)
p.paragraph_format.tab_stops.add_tab_stop(Inches(6.5), WD_TAB_ALIGNMENT.RIGHT)
p.add_run("CONTENTS").bold = True
p.add_run("\tPAGE No.").bold = True
static_list("toc", 'TOC \\o "1-2" \\h \\z \\u')

# ================================================================ chapters ==
body = doc.add_section(WD_SECTION.NEW_PAGE)
body.different_first_page_header_footer = False
page_border(body)
page_number_format(body, "decimal", 1)
footer_page_number(body)


def add_table(caption, rows):
    para(caption, style="Table Caption")
    cols = len(rows[0])
    t = doc.add_table(rows=len(rows), cols=cols)
    t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j in range(cols):
            cell = t.rows[i].cells[j]
            p = cell.paragraphs[0]
            p.paragraph_format.line_spacing = 1.0; p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(row[j] if j < len(row) else ""); r.font.size = Pt(10)
            if i == 0:
                r.bold = True; set_cell_shading(cell, "E7E6E2")
    # fixed column widths proportional to content
    t.autofit = False
    def col_len(j):
        cells = [r[j] if j < len(r) else "" for r in rows]
        longest_text = max(min(len(c), 60) for c in cells)
        longest_word = max((len(w) for c in cells for w in c.split()), default=1)
        return max(longest_text, longest_word * 2 + 4) + 4    # never narrower than its longest word
    lens = [col_len(j) for j in range(cols)]
    total = sum(lens)
    for j in range(cols):
        w = Inches(6.5 * lens[j] / total)
        t.columns[j].width = w
        for row in t.rows:
            row.cells[j].width = w
    # short tables stay on one page; no row splits across pages
    for k, row in enumerate(t.rows):
        trPr = row._tr.get_or_add_trPr(); cs = OxmlElement("w:cantSplit"); cs.set(qn("w:val"), "true"); trPr.append(cs)
        if len(rows) <= 12 and k < len(rows) - 1:
            for c in row.cells:
                for pp in c.paragraphs:
                    pp.paragraph_format.keep_with_next = True
    # repeat header row on page breaks
    trPr = t.rows[0]._tr.get_or_add_trPr(); h = OxmlElement("w:tblHeader"); h.set(qn("w:val"), "true"); trPr.append(h)
    para("", after=2, spacing=1.0)


def add_algo(title, lines):
    para(title, style="Table Caption").runs[0].font.size = Pt(11)
    # keep as a plain bold label, not listed in the List of Tables
    doc.paragraphs[-1].style = normal
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.LEFT
    doc.paragraphs[-1].paragraph_format.keep_with_next = True
    doc.paragraphs[-1].runs[0].bold = True
    t = doc.add_table(rows=1, cols=1); t.style = "Table Grid"; t.autofit = False
    t.columns[0].width = Inches(6.5)
    trPr = t.rows[0]._tr.get_or_add_trPr(); cs = OxmlElement("w:cantSplit"); cs.set(qn("w:val"), "true"); trPr.append(cs)
    cell = t.rows[0].cells[0]; cell.width = Inches(6.5); set_cell_shading(cell, "F6F6F4")
    cell.paragraphs[0].style = code
    for k, ln in enumerate(lines):
        p = cell.paragraphs[0] if k == 0 else cell.add_paragraph(style="Algorithm Code")
        p.add_run(ln.replace(" ", " "))
    para("", after=2, spacing=1.0)


with open(os.path.join(HERE, "content.txt"), encoding="utf-8") as f:
    lines = [ln.rstrip("\n") for ln in f]

i, first_chapter = 0, True
while i < len(lines):
    ln = lines[i]
    if not ln.strip():
        i += 1; continue
    if ln.startswith("#CH "):
        num, title = ln[4:].split("|", 1)
        if not first_chapter:
            page_break()
        first_chapter = False
        doc.add_paragraph(f"Chapter {num}: {title}", style="Heading 1")
    elif ln.startswith("### "):
        doc.add_paragraph(ln[4:], style="Heading 3")
    elif ln.startswith("## "):
        doc.add_paragraph(ln[3:], style="Heading 2")
    elif ln.startswith("FIG|"):
        _, fn, cap, w = ln.split("|")
        p = para(align=WD_ALIGN_PARAGRAPH.CENTER, after=0, spacing=1.0, keep_next=True)
        p.add_run().add_picture(os.path.join(FIG, fn), width=Inches(float(w)))
        para(cap, style="Figure Caption")
    elif ln.startswith("TAB|"):
        cap = ln[4:]; rows = []; i += 1
        while i < len(lines) and lines[i].startswith("|"):
            rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")]); i += 1
        add_table(cap, rows); continue
    elif ln.startswith("ALGO|"):
        title = ln[5:]; body_lines = []; i += 1
        while lines[i] != "ENDALGO":
            body_lines.append(lines[i]); i += 1
        add_algo(title, body_lines)
    elif ln.startswith("EQ|"):
        para(ln[3:], align=WD_ALIGN_PARAGRAPH.CENTER, italic=True, after=2, spacing=1.15)
    elif ln.startswith("- "):
        rich(doc.add_paragraph(style="List Bullet"), ln[2:])
    elif re.match(r"^[a-f]\) ", ln):
        p = rich(para(), ln); p.paragraph_format.left_indent = Inches(0.3); p.paragraph_format.first_line_indent = Inches(-0.3)
        p.paragraph_format.space_after = Pt(2)
    else:
        rich(para(), ln)
    i += 1

# --------------------------------------------------------------- references --
page_break()
doc.add_paragraph("References", style="Heading 1")
REFS = [
    "D. E. Denning, “An intrusion-detection model,” IEEE Transactions on Software Engineering, vol. SE-13, no. 2, pp. 222-232, 1987.",
    "M. Roesch, “Snort: Lightweight intrusion detection for networks,” in Proc. 13th USENIX Conference on System Administration (LISA), 1999, pp. 229-238.",
    "P. García-Teodoro, J. Díaz-Verdejo, G. Maciá-Fernández, and E. Vázquez, “Anomaly-based network intrusion detection: Techniques, systems and challenges,” Computers & Security, vol. 28, no. 1-2, pp. 18-28, 2009.",
    "A. L. Buczak and E. Guven, “A survey of data mining and machine learning methods for cyber security intrusion detection,” IEEE Communications Surveys & Tutorials, vol. 18, no. 2, pp. 1153-1176, 2016.",
    "I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, “Toward generating a new intrusion detection dataset and intrusion traffic characterization,” in Proc. 4th International Conference on Information Systems Security and Privacy (ICISSP), 2018, pp. 108-116.",
    "G. Engelen, V. Rimmer, and W. Joosen, “Troubleshooting an intrusion detection dataset: the CICIDS2017 case study,” in Proc. IEEE Security and Privacy Workshops (SPW), 2021, pp. 7-12.",
    "T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in Proc. 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016, pp. 785-794.",
    "Y. Mirsky, T. Doitshman, Y. Elovici, and A. Shabtai, “Kitsune: An ensemble of autoencoders for online network intrusion detection,” in Proc. Network and Distributed System Security Symposium (NDSS), 2018.",
    "B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, “Communication-efficient learning of deep networks from decentralized data,” in Proc. 20th International Conference on Artificial Intelligence and Statistics (AISTATS), PMLR vol. 54, 2017, pp. 1273-1282.",
    "D. J. Beutel et al., “Flower: A friendly federated learning research framework,” arXiv:2007.14390, 2020.",
    "S. Agrawal et al., “Federated learning for intrusion detection system: Concepts, challenges and future directions,” Computer Communications, vol. 195, pp. 346-361, 2022.",
    "H. B. McMahan, D. Ramage, K. Talwar, and L. Zhang, “Learning differentially private recurrent language models,” in Proc. International Conference on Learning Representations (ICLR), 2018.",
    "I. Mironov, “Rényi differential privacy,” in Proc. IEEE 30th Computer Security Foundations Symposium (CSF), 2017, pp. 263-275.",
    "S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in Advances in Neural Information Processing Systems 30 (NeurIPS), 2017.",
    "M. T. Ribeiro, S. Singh, and C. Guestrin, “‘Why should I trust you?’ Explaining the predictions of any classifier,” in Proc. 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016, pp. 1135-1144.",
    "J. Platt, “Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods,” in Advances in Large Margin Classifiers, A. J. Smola et al., Eds. MIT Press, 1999.",
    "A. Niculescu-Mizil and R. Caruana, “Predicting good probabilities with supervised learning,” in Proc. 22nd International Conference on Machine Learning (ICML), 2005, pp. 625-632.",
    "R. Sommer and V. Paxson, “Outside the closed world: On using machine learning for network intrusion detection,” in Proc. IEEE Symposium on Security and Privacy, 2010, pp. 305-316.",
    "D. Arp et al., “Dos and don’ts of machine learning in computer security,” in Proc. 31st USENIX Security Symposium, 2022, pp. 3971-3988.",
    "R. Geirhos et al., “Shortcut learning in deep neural networks,” Nature Machine Intelligence, vol. 2, pp. 665-673, 2020.",
    "Canadian Institute for Cybersecurity, “Intrusion Detection Evaluation Dataset (CIC-IDS2017),” University of New Brunswick. [Online]. Available: https://www.unb.ca/cic/datasets/ids-2017.html [Accessed: Nov. 2026].",
    "Canadian Institute for Cybersecurity and Communications Security Establishment, “CSE-CIC-IDS2018 on AWS.” [Online]. Available: https://www.unb.ca/cic/datasets/ids-2018.html [Accessed: Nov. 2026].",
    "Nmap Project, “Npcap: Windows packet capture library and driver.” [Online]. Available: https://npcap.com [Accessed: Nov. 2026].",
]
for k, r in enumerate(REFS, 1):
    p = para(f"[{k}]  {r}", align=WD_ALIGN_PARAGRAPH.LEFT, after=4, spacing=1.15)
    p.paragraph_format.left_indent = Inches(0.45); p.paragraph_format.first_line_indent = Inches(-0.45)

os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
doc.save(OUT)
print("saved", OUT)
