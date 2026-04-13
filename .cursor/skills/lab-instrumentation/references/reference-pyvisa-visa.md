# PyVISA and VISA sessions

## Architecture

- Use a single **`ResourceManager`** (or inject one) per process for address parsing and backend selection; avoid creating managers inside tight loops.
- Open resources with explicit **resource strings** (`TCPIP0::host::inst0::INSTR`, `USB0::...`, `ASRL3::INSTR`, etc.) and keep the **open handle** lifetime obvious: **`with rm.open_resource(...) as inst:`** or a small **`InstrumentSession`** class with **`close()`** in **`__exit__`**.

## Lifecycle vs protocol

- **Lifecycle**: `open_resource`, **`baud_rate`**, **`data_bits`**, **`parity`**, **`stop_bits`** for serial; **`read_termination`** / **`write_termination`**; **`timeout`** (milliseconds in PyVISA). Set these once after open, not per scattered call site.
- **Protocol**: functions or a **`SCPIClient`** that only **`write`**, **`query`**, **`read_raw`**, and parse responses. Never embed **`open_resource`** inside a “send SCPI” helper.

## Timeouts and partial reads

- Every blocking **`read`** must respect **`timeout`**. For binary payloads with known length, prefer **`read_bytes(count)`** (or read until delimiter) instead of unbounded **`read`**.
- After errors, many instruments need **`clear()`** or a **`*CLS`** / device-specific recovery sequence before the bus is trustworthy again—document that path in one place.

## Terminations and encoding

- SCPI text is usually **7-bit ASCII** with **`\\n`** termination. Match **`read_termination`** / **`write_termination`** to the manual; mixed **`\\r\\n`** gear is common on serial.
- For **`read_raw`**, you own **length** and **endianness**; do not apply text terminations.

## Queries and staleness

- Prefer **`query(message)`** for “write then read line” patterns to avoid half-open transactions.
- If the instrument can **async** complete (long sweeps), either poll **`*OPC?`** / status registers per manual or block with a **longer timeout** on the final read—do not chain short reads that race completion.

## Backends

- **NI-VISA** (vendor stack) vs **`PyVISA-py`**: choose per deployment; CI often uses **`PyVISA-py`** or simulation. Document which backend the project expects in **`README`** or config.

## Logging and safety

- Log **resource string** (sanitized if it embeds secrets) and **high-level command names**, not passwords. Avoid logging full **waveform blobs** at info level.
