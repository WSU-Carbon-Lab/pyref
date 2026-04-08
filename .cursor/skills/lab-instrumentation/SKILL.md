---
author: dotagents
name: lab-instrumentation
description: PyVISA and lab instrument control in Python: VISA resource lifecycle, timeouts and terminations, raw TCP/UDP sockets vs VISA, when to introduce hardware abstraction layers, validating user and config input before hardware, testing without hardware, and extracting tables or text from instrument manuals and datasheets (PDF). Triggers on pyvisa, VISA, GPIB, USB-TMC, serial instrument, socket lab, SCPI, datasheet PDF, instrument driver.
---

# Lab instrumentation (PyVISA, sockets, HAL)

## Quick start

1. **Split layers**: one place owns **open/configure/close** and **timeouts**; another owns **SCPI or device protocol** (string formatting, parsing floats/units, error queues). See [reference-pyvisa-visa.md](references/reference-pyvisa-visa.md).
2. **Pick transport deliberately**: **VISA** when the stack already provides enumeration, buffering, and vendor backends; **plain sockets** when you own framing, keep-alive, and binary protocols end to end. See [reference-socket-comms.md](references/reference-socket-comms.md).
3. **Add a HAL when** you have **multiple models** behind one experiment API, **two transports** for the same logical device, or **tests** that need a stable seam. Do not wrap a single serial string in three classes. See [reference-hardware-abstraction.md](references/reference-hardware-abstraction.md).
4. **Validate before I/O**: normalize and bound **numeric setpoints**, **enum-like modes**, and **user-entered strings**; reject unknown commands early with **actionable errors**. See [reference-input-validation-errors.md](references/reference-input-validation-errors.md).
5. **PDFs**: use **text-layer** extraction first; fall back to **table-aware** tools for spec tables; reserve **OCR** for scan-only manuals. See [reference-pdf-datasheets.md](references/reference-pdf-datasheets.md).
6. **Tests**: default **fast** unit tests on parsers and validators; gate **integration** tests; use **fakes** or **recorded transcripts** for CI. See [reference-testing-mocks.md](references/reference-testing-mocks.md).

## Stack synergy

| Resource | Role |
|----------|------|
| **general-python** | uv, context managers, **`python-reviewer`** |
| **numpy-scientific** | Numeric buffers, waveforms, dtype-safe binary payloads |
| **numpy-docstrings** | Public driver and session APIs |

## Reference index

| Topic | File |
|--------|------|
| ResourceManager, sessions, timeouts, terminations, SCPI hygiene | [reference-pyvisa-visa.md](references/reference-pyvisa-visa.md) |
| TCP/UDP sockets, framing, timeouts, binary vs text | [reference-socket-comms.md](references/reference-socket-comms.md) |
| When to build HALs, protocols vs transports, registries | [reference-hardware-abstraction.md](references/reference-hardware-abstraction.md) |
| Ranges, enums, command allowlists, error shape | [reference-input-validation-errors.md](references/reference-input-validation-errors.md) |
| PDF text, tables, datasheets, bibliography PDFs, OCR | [reference-pdf-datasheets.md](references/reference-pdf-datasheets.md) |
| pytest, fakes, simulation, recordings | [reference-testing-mocks.md](references/reference-testing-mocks.md) |

## Dependencies

Add with **`uv add`** (do not hand-edit pins in `pyproject.toml`):

| Need | Typical packages |
|------|------------------|
| VISA in Python | **`pyvisa`**; optional **`PyVISA-py`** for pure-Python backend without vendor IVI |
| Serial-backed VISA | Often **`pyserial`** alongside backend docs |
| PDF text | **`pymupdf`** (fitz), **`pypdf`**, or **`pdfplumber`** (tables) |
| Scanned PDFs | **`pdf2image`** + **`pytesseract`** when OCR is unavoidable |

## Official and canonical docs

- [PyVISA documentation](https://pyvisa.readthedocs.io/en/latest/)
- [Python `socket` module](https://docs.python.org/3/library/socket.html)
- [PyMuPDF documentation](https://pymupdf.readthedocs.io/)
- [pdfplumber](https://github.com/jsvine/pdfplumber)
- [pypdf](https://pypdf.readthedocs.io/)
