# PDF parsing for datasheets, manuals, and reference PDFs

## Strategy

1. **Detect text vs scan**: open the PDF and try **text extraction** on a sample page. If output is empty or garbage, the file is likely **image-only**—plan **OCR** or obtain a **text** export from the vendor.
2. **Preserve structure**: tables need **table-aware** tools; body text needs **blocks** with **bbox** when aligning columns.
3. **Ground truth**: for critical limits (max voltage, derating), **cross-check** against a **second source** (vendor HTML or printed table)—PDF extraction is **lossy**.

## Libraries (Python)

| Goal | Typical choice | Notes |
|------|----------------|-------|
| Fast text + layout | **[PyMuPDF](https://pymupdf.readthedocs.io/)** (`fitz`) | **`get_text("dict")`** for blocks/lines; good speed |
| Tables in datasheets | **[pdfplumber](https://github.com/jsvine/pdfplumber)** | **`extract_tables()`** tuned with **`table_settings`** |
| Lightweight merge/split/metadata | **[pypdf](https://pypdf.readthedocs.io/)** | Good for **page ranges** and **bookmarks**, weaker on messy tables |
| Scanned pages | **`pdf2image`** + **`pytesseract`** | Tune **DPI** (often 300+); expect **manual cleanup** |

## Datasheet workflows

- **Extract spec tables** (temperature range, absolute maxima) into **structured rows** (CSV/Parquet) with **column headers** normalized once in code.
- **Version** the PDF (filename hash or vendor revision field) and **store** it beside extracted data so results are **reproducible**.

## Reference books and papers

- **Bibliography PDFs**: often **two-column**; use **block sorting** by **y, x** (PyMuPDF dict mode) before regex for **DOI**, **ISBN**, or **citation keys**.
- **Equations and figures**: extraction will not reliably capture math; link **page number** + **figure ID** instead of parsing **LaTeX** from raster.

## Pitfalls

- **Hyphenation** and **line breaks** split words across lines—normalize whitespace and **de-hyphenate** cautiously.
- **Embedded fonts** and **subset encoding** can map glyphs oddly; if strings look wrong, try **`get_text("text")`** vs **dict** mode or another backend.
- **Legal**: scraping **paywalled** PDFs may violate terms; only process files you **have rights** to use in your pipeline.

## Further reading

- PyMuPDF [tutorial / text](https://pymupdf.readthedocs.io/en/latest/tutorial.html)
- pdfplumber [README examples](https://github.com/jsvine/pdfplumber/blob/stable/README.md)
- pypdf [user guide](https://pypdf.readthedocs.io/en/stable/user/introduction.html)
