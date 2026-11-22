"""
report/make_report.py
End-to-end build of the final report:

  1. build_report.py      -> draft .docx (TOC / figure / table lists as Word fields)
  2. finalize_word.py     -> draft PDF rendered by Word (fields updated)
  3. read the page label (roman / arabic footer number) of every heading and
     caption from that PDF
  4. build_report.py with those lists written out -> final .docx
  5. finalize_word.py     -> final PDF, then re-check every listed page number;
     repeat 3-5 until the numbers are stable

    python report/make_report.py <out_dir>
"""
import json
import os
import re
import subprocess
import sys

import pymupdf
from docx import Document

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OUT_DIR = os.path.abspath(sys.argv[1])
BUILD = os.path.join(OUT_DIR, "build")
NAME = "A-IDAPS-FL_Final_Report"
LOGO = os.path.join(HERE, "figures", "mmmut_logo.png")
os.makedirs(BUILD, exist_ok=True)


def run(args, timeout=300):
    r = subprocess.run([PY, *args], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise SystemExit(f"failed: {args}\n{r.stdout}\n{r.stderr[-2000:]}")
    return r.stdout


def norm(s):
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip().lower()


def targets(docx_path):
    """(list key, level, text) in document order."""
    out = []
    for p in Document(docx_path).paragraphs:
        st, tx = p.style.name, p.text.strip()
        if not tx:
            continue
        if st == "Front Heading":
            out.append(("toc", 1, tx, "front"))
        elif st == "Heading 1":
            out.append(("toc", 1, tx, "body"))
        elif st == "Heading 2":
            out.append(("toc", 2, tx, "body"))
        elif st == "Figure Caption":
            out.append(("lof", 1, tx, "body"))
        elif st == "Table Caption":
            out.append(("lot", 1, tx, "body"))
    return out


def page_labels(pdf_path):
    doc = pymupdf.open(pdf_path)
    pages = []
    for pg in doc:
        h = pg.rect.height
        blocks = [b for b in pg.get_text("blocks") if b[4].strip() and b[1] > h * 0.9]
        label = max(blocks, key=lambda b: b[1])[4].strip() if blocks else ""   # lowest block = footer
        txt = pg.get_text()
        pages.append((label, norm(txt), {norm(l) for l in txt.splitlines()}))
    return pages


def locate(tgts, pages):
    body_start = next(i for i, pg in enumerate(pages) if pg[0] == "1")
    lists = {"toc": [], "lof": [], "lot": []}
    cur_front, cur_body = 0, body_start
    for key, level, text, where in tgts:
        needle = norm(text)[:70]
        rng = range(cur_front, body_start) if where == "front" else range(cur_body, len(pages))
        hit = None
        for i in rng:
            body = pages[i][1]
            if where == "front":                     # page titles: a line of their own
                if needle in pages[i][2]:
                    hit = i; break
            elif needle in body:
                hit = i; break
        if hit is None:
            raise SystemExit(f"not found in PDF: {text!r}")
        if where == "front":
            cur_front = hit
        else:
            cur_body = hit
        lists[key].append([level, text, pages[hit][0]])
    return lists


def word_pdf(docx_path, pdf_path, update_toc=False):
    args = [os.path.join(HERE, "finalize_word.py"), docx_path, pdf_path] + (["--update-toc"] if update_toc else [])
    print(run(args).strip().splitlines()[0])


draft_docx, draft_pdf = os.path.join(BUILD, f"{NAME}_draft.docx"), os.path.join(BUILD, f"{NAME}_draft.pdf")
final_docx, final_pdf = os.path.join(OUT_DIR, f"{NAME}.docx"), os.path.join(OUT_DIR, f"{NAME}.pdf")
lists_json = os.path.join(BUILD, "lists.json")

print("1. draft"); run([os.path.join(HERE, "build_report.py"), LOGO, draft_docx])
print("2. Word render (fields updated)"); word_pdf(draft_docx, draft_pdf, update_toc=True)
tg = targets(draft_docx)
lists = locate(tg, page_labels(draft_pdf))
for it in range(3):
    json.dump(lists, open(lists_json, "w", encoding="utf-8"), indent=1)
    print(f"3-4. final docx (pass {it + 1})"); run([os.path.join(HERE, "build_report.py"), LOGO, final_docx, lists_json])
    print("5. Word render"); word_pdf(final_docx, final_pdf)
    check = locate(targets(final_docx), page_labels(final_pdf))
    diff = [(a[1], a[2], b[2]) for k in lists for a, b in zip(lists[k], check[k]) if a[2] != b[2]]
    if not diff:
        print(f"page numbers verified: {sum(len(v) for v in lists.values())} entries match the final PDF")
        break
    print(f"   {len(diff)} entries moved, e.g. {diff[:3]}; rebuilding")
    lists = check
else:
    raise SystemExit("page numbers did not stabilise")
print("pages:", len(pymupdf.open(final_pdf)), "|", final_docx, "|", final_pdf)
