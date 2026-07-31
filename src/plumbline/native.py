"""The same lossless JPEG decoder as :mod:`plumbline.reference`, in compiled C.

`reference.py` is the oracle: readable, dependency-free, and slow. `turbo.py`
accelerates it with numba, which costs a JIT pause on first use and ships LLVM
with the application. This module accelerates it with ~350 lines of C11
(`native/plumbline.c`), compiled once at build time into a small shared
library that this file loads through ctypes, and must agree with the oracle
bit for bit — `tests/test_native.py` decodes real discs and random scans
through both and diffs the results.

The split of work is deliberate: Python parses and validates the JPEG headers,
because that runs once per frame and is where all the refusals live; C runs
the three loops that cost the time (byte destuffing, table building, the scan
itself). The C side is handed plain numbers and buffers — no Python C API —
so the library builds with any C compiler and one binary serves every Python
version on a platform.

Unlike `turbo`, colour frames (three components, 1x1 sampling, one
interleaved scan) are decoded here, with per-component predictor state and
per-component Huffman tables, exactly as the oracle does.

The shared library is optional. Where `_plumbline.dll` / `.so` / `.dylib` is
missing — a source checkout that never built it, an architecture without a
wheel — the module still imports and reports ``AVAILABLE = False``, and
`decode` refuses, so a caller falls back (see `plumbline`) rather than
losing the ability to open a file.

House rule, as everywhere in this project: decode correctly or raise
`LosslessJpegError` — never return an image that merely looks decoded. A scan
that ends before the image does is refused, as is one carrying a code its own
Huffman table never defines, as is a restart interval that is not a whole
number of MCU-rows (T.81 §H.1.1). All three are refused by all three decoders,
because the header, table and entropy-segment checks are `reference`'s own,
imported rather than reimplemented here.

That last refusal is what lets the C loops below assume a restart always lands
at column zero, which is what T.81 §H.1.2.1's rule — the whole first line of
every restart interval predicts from Ra — needs in order to be expressible as
a per-row flag rather than a per-sample one.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np

from plumbline.reference import (
    LosslessJpegError,
    check_frame,
    check_scan,
    check_table,
    header,
    scan_slots as _slots,
)

_SUFFIX = {"win32": ".dll", "darwin": ".dylib"}.get(sys.platform, ".so")
_LIBRARY = Path(__file__).with_name("_plumbline" + _SUFFIX)

_ABI = 1
_OK = 0
_TRUNCATED = -2
_BAD_CODE = -6

_lib = None
try:                                       # pragma: no cover - depends on build
    if _LIBRARY.exists():
        _candidate = ctypes.CDLL(str(_LIBRARY))
        _candidate.plumbline_abi.restype = ctypes.c_int32
        _candidate.plumbline_abi.argtypes = []
        if _candidate.plumbline_abi() == _ABI:
            _candidate.plumbline_decode.restype = ctypes.c_int32
            _candidate.plumbline_decode.argtypes = [
                ctypes.c_char_p, ctypes.c_int64,               # scan
                ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,  # w, h, ncomp
                ctypes.c_int32, ctypes.c_int32,                # precision, selector
                ctypes.c_int32, ctypes.c_int32,                # shift, interval
                ctypes.c_int32,                                # ntables
                ctypes.c_void_p, ctypes.c_void_p,              # counts, symbols
                ctypes.c_void_p,                               # symbol_counts
                ctypes.c_void_p, ctypes.c_void_p,              # slot tables/comps
                ctypes.c_void_p, ctypes.c_int32,               # out, wide
            ]
            _lib = _candidate
except Exception:                          # pragma: no cover - a broken binary
    _lib = None

AVAILABLE = _lib is not None




def _tables(info: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deduplicate the scan's Huffman tables and flatten them for the C side.

    The validation is `reference.check_table`, so a malformed table is refused
    with the same message whichever decoder sees it first.
    """
    index_of: dict[int, int] = {}
    counts_flat: list[int] = []
    symbols_flat: list[int] = []
    symbol_counts: list[int] = []

    for identifier in info["table_ids"]:
        if identifier not in info["tables"]:
            raise LosslessJpegError(f"Huffman table {identifier} is missing")
        if identifier in index_of:
            continue
        counts, symbols = info["tables"][identifier]
        check_table(counts, symbols)
        used = symbols[:sum(counts)]
        index_of[identifier] = len(symbol_counts)
        counts_flat.extend(counts)
        symbols_flat.extend(used)
        symbol_counts.append(len(used))

    slot_tables = [index_of[identifier] for identifier in info["table_ids"]]
    return (np.array(counts_flat, dtype=np.uint8),
            np.array(symbols_flat, dtype=np.uint8),
            np.array(symbol_counts, dtype=np.int32),
            np.array(slot_tables, dtype=np.int32))


def decode(frame: bytes) -> np.ndarray:
    """Decode one lossless JPEG frame, identically to `plumbline.reference.decode`.

    Greyscale frames return (height, width); colour frames return
    (height, width, components) in the frame's component order.

    Raises LosslessJpegError for anything this decoder does not handle — the
    shared library missing included — so a caller falls back rather than
    receiving a plausible-looking wrong image.
    """
    if not AVAILABLE:
        raise LosslessJpegError(
            "the native decoder library is not built for this platform")

    info = header(frame)
    order = _slots(info)
    check_frame(info)

    precision = info["precision"]
    height, width = info["height"], info["width"]
    components = info["components"]
    shift = info["point_transform"]
    dtype = np.uint8 if precision <= 8 else np.uint16

    counts, symbols, symbol_counts, slot_tables = _tables(info)
    slot_comps = np.array(order, dtype=np.int32)
    scan = bytes(frame[info["scan_offset"]:])
    check_scan(scan, height * width, info["restart_interval"])

    out = np.empty(height * width * components, dtype=dtype)
    status = _lib.plumbline_decode(
        scan, len(scan),
        width, height, components,
        precision, info["predictor"], shift, info["restart_interval"],
        len(symbol_counts),
        counts.ctypes.data, symbols.ctypes.data, symbol_counts.ctypes.data,
        slot_tables.ctypes.data, slot_comps.ctypes.data,
        out.ctypes.data, 1 if dtype is np.uint16 else 0)

    if status == _TRUNCATED:
        raise LosslessJpegError("the entropy-coded data ends before the image does")
    if status == _BAD_CODE:
        raise LosslessJpegError(
            "the scan contains a code the frame's Huffman tables do not define")
    if status != _OK:
        raise LosslessJpegError(f"the native decoder refused the frame ({status})")

    if shift:
        out <<= np.array(shift, dtype=dtype)
    if components == 1:
        return out.reshape(height, width)
    return out.reshape(height, width, components)
