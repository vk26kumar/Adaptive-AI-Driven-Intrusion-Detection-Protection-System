"""
report/finalize_word.py
Render a .docx to PDF with Microsoft Word (COM): optionally update the TOC fields, export. The document itself is never saved
(on the build machine Word could export PDF but hung on any save).

    python report/finalize_word.py <in.docx> <out.pdf> [--update-toc]

Notes: AutoFitBehavior(wdAutoFitContent) and doc.Fields.Update() hung Word for
more than ten minutes on this document; column widths come from build_report.py.
"""
import os
import sys

import win32com.client as win32

src, pdf = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
update_toc = "--update-toc" in sys.argv

word = win32.DispatchEx("Word.Application")
word.Visible = False
word.DisplayAlerts = 0
try:
    doc = word.Documents.Open(src, False, True, False)   # ConfirmConversions, ReadOnly, AddToRecentFiles
    # tables keep the fixed column widths written by build_report.py
    doc.Repaginate()
    if update_toc:
        for _ in range(2):
            for toc in doc.TablesOfContents:
                toc.Update()
            doc.Repaginate()
    print("pages:", doc.ComputeStatistics(2), flush=True)
    doc.ExportAsFixedFormat(pdf, 17)                     # wdExportFormatPDF
    doc.Close(False)
    print("exported", pdf, flush=True)
finally:
    word.Quit()
