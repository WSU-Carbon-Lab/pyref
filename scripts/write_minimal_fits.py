"""Write a minimal FITS file for tests (stdlib only, no astropy)."""
from pathlib import Path


def card(keyword: str, value: str | int | float | bool) -> bytes:
    if isinstance(value, bool):
        s = f"{keyword:8}= {str(value):>20}"
    elif isinstance(value, int):
        s = f"{keyword:8}= {value:20d}"
    elif isinstance(value, float):
        s = f"{keyword:8}= {value:20.14E}"
    else:
        s = f"{keyword:8}= '{value[:68]:<68}'"
    return (s + " " * (80 - len(s)))[:80].encode("ascii")


def end_card() -> bytes:
    return ("END" + " " * 77).encode("ascii")


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "minimal.fits"
    out.parent.mkdir(parents=True, exist_ok=True)

    primary = b"".join([
        card("SIMPLE", True),
        card("BITPIX", 16),
        card("NAXIS", 0),
        card("DATE", "2024-01-01"),
        card("Beamline Energy", 500.0),
        card("Sample Theta", 1.0),
        card("CCD Theta", 2.0),
        card("Higher Order Suppressor", 0.0),
        card("EPU Polarization", 1.0),
        end_card(),
    ])
    primary_block = primary + b" " * (2880 - len(primary))

    ext = b"".join([
        card("XTENSION", "IMAGE   "),
        card("BITPIX", 16),
        card("NAXIS", 2),
        card("NAXIS1", 2),
        card("NAXIS2", 2),
        card("BZERO", 0),
        end_card(),
    ])
    ext_block = ext + b" " * (2880 - len(ext))

    import struct
    img = struct.pack(">4h", 0, 1, 2, 3)
    img_block = img + b"\x00" * (2880 - 8)

    out.write_bytes(primary_block + ext_block + img_block)
    print(out)


if __name__ == "__main__":
    main()
