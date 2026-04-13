# Validating user and config input before hardware

## Principles

- **Validate at boundaries**: CLI, REST body, **GUI field**, or **YAML/JSON** config—before any **`write`** to hardware.
- **Reject early**: unknown **mode strings**, **out-of-range** voltages, or **wrong units** should raise **typed exceptions** with **what failed**, **allowed range**, and **observed value**—not a raw **`VisaIOError`** after the fact.

## Numeric setpoints

- Parse with **explicit types** (`Decimal` for money-like precision, **`float`** only when the manual specifies float-friendly steps). Compare to **min/max** from **datasheet** or **constants** in code, not magic numbers scattered across callers.
- **Quantize** to instrument resolution when the device **snaps** to steps (e.g. **1 mV** steps); document rounding **toward zero** vs **nearest**.

## Command and mode strings

- Prefer **`Literal`** or **`Enum`** in public APIs instead of **free-form** strings for **SCPI subsystems**.
- If users type SCPI, maintain an **allowlist** or **prefix allowlist**; never pass unchecked input to **`write`** when it can reach **shells** or **file paths** on controllers.

## Resource strings and hosts

- Validate **`host:port`** formats, **IP literals**, and **resource strings** with **regex or urllib** patterns before **`open_resource`** or **`connect`**.
- **Path traversal** matters for **save/recall** instrument setups that use filenames on the controller.

## Error design

- Use **small exception types** (`OutOfRange`, `UnknownMode`, `InstrumentFault`) that carry **context**; map **VISA status** and **socket errno** into them at the **transport edge** so experiment code stays readable.
- For operator-facing messages, separate **log detail** (full traceback internally) from **user text** (one sentence + fix hint).

## Idempotency

- **Repeated “set 5 V”** should be safe; **“arm”** may not be—document which operations are **idempotent** in the HAL docstrings.
