# Socket communication for lab gear

## When sockets instead of VISA

- **Custom binary framing**, **multicast**, **UDP discovery**, or **non-VISA** Ethernet devices (raw TCP on a port) are easier with **`socket`** or **`asyncio`** streams than with VISA resource strings.
- **VISA** still wins when you need **USB-TMC**, **GPIB**, **serial via VISA**, or a single **enumeration** story across buses.

## Client patterns

- Prefer **`socket.create_connection((host, port), timeout=...)`** over manual **`connect`** when you want **fail-fast** timeouts and **IPv4/IPv6** handling.
- Set **`sock.settimeout(...)`** for subsequent operations unless you use non-blocking **`asyncio`**.
- Use **`contextlib.closing`** or a small wrapper so **`close()`** runs on error paths.

## Framing

- **Line-oriented** text: read with **`recv`** until delimiter, or use **`readline`** on a **`makefile`** wrapper—watch **blocking** if the peer never sends the delimiter.
- **Length-prefixed** binary: read **header** (fixed bytes), parse **length**, then **`recv`** until **`n`** bytes; loop because **`recv`** may return **partial** chunks.
- **Struct pack/unpack** for fixed layouts; document **endianness** (`"<"` vs `">"`) next to the struct format string.

## Timeouts and shutdown

- Distinguish **`ETIMEDOUT`** (no data) from **`ECONNRESET`**. Surface both with **clear exceptions** so operators know whether to retry or re-seat cabling.
- For clean shutdown, **`shutdown(SHUT_RDWR)`** before **`close`** when the protocol expects half-close behavior.

## Threading and async

- **`socket`** objects are not thread-safe for arbitrary interleaved **`send`/`recv`**; one thread per socket or a **queue-driven** writer/reader design.
- **`asyncio`** suits many **slow instruments**; keep **protocol state machines** explicit so partial reads do not corrupt state.

## Security

- **Lab LAN** is not trusted by default: validate **host allowlists**, avoid **command injection** into shells that restart services, and never expose raw socket servers to the open internet without authentication.
