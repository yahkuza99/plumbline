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
that ends before the image does is refused, exactly as `turbo` refuses
it, where the oracle would quietly invent the tail.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np

from plumbline.reference import MAX_PRECISION, LosslessJpegError, header

_SUFFIX = {"win32": ".dll", "darwin": ".dylib"}.get(sys.platform, ".so")
_LIBRARY = Path(__file__).with_name("_plumbline" + _SUFFIX)

_ABI = 1
_OK = 0
_TRUNCATED = -2

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


def _slots(info: dict) -> list[int]:
    """Map each scan component back to its slot in the frame, as the oracle
    does, so the output planes land in frame order whatever order the scan
    lists them in."""
    components = info.get("components", 0)
    if components < 1:
        raise LosslessJpegError("frame declares no components")
    for horizontal, vertical in info.get("sampling", []):
        if (horizontal, vertical) != (1, 1):
            raise LosslessJpegError(
                f"sampling factor {horizontal}x{vertical} is not supported; "
                "only 1x1 (no subsampling) is")
    if len(info["scan_ids"]) != components:
        raise LosslessJpegError(
            f"scan interleaves {len(info['scan_ids'])} of {components} "
            "components; non-interleaved scans are not supported")

    frame_ids = info["frame_ids"]
    order: list[int] = []
    for scan_id in info["scan_ids"]:
        if scan_id not in frame_ids or frame_ids.index(scan_id) in order:
            raise LosslessJpegError(
                f"scan component {scan_id} does not match the frame")
        order.append(frame_ids.index(scan_id))
    return order


def _validate(info: dict) -> None:
    if not 1 <= info["predictor"] <= 7:
        raise LosslessJpegError(f"predictor {info['predictor']} is not defined")
    precision = info["precision"]
    if not 1 <= precision <= MAX_PRECISION:
        raise LosslessJpegError(f"precision {precision} is out of range")
    if info["point_transform"] >= precision:
        raise LosslessJpegError("point transform is larger than the precision")


def _tables(info: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deduplicate the scan's Huffman tables and flatten them for the C side.

    The validation mirrors what the oracle and `turbo` raise while
    building their lookups, so a malformed table is refused with the same
    message whichever decoder sees it first.
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
        if len(counts) != 16 or sum(counts) > len(symbols):
            raise LosslessJpegError("Huffman table is truncated")
        used = symbols[:sum(counts)]
        for symbol in used:
            if symbol > MAX_PRECISION:
                raise LosslessJpegError(f"SSSS={symbol} is out of range")
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
    _validate(info)

    precision = info["precision"]
    height, width = info["height"], info["width"]
    components = info["components"]
    shift = info["point_transform"]
    dtype = np.uint8 if precision <= 8 else np.uint16

    if height == 0 or width == 0:
        empty = np.zeros((height, width, components), dtype=dtype)
        return empty[:, :, 0] if components == 1 else empty

    counts, symbols, symbol_counts, slot_tables = _tables(info)
    slot_comps = np.array(order, dtype=np.int32)
    scan = bytes(frame[info["scan_offset"]:])

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
    if status != _OK:
        raise LosslessJpegError(f"the native decoder refused the frame ({status})")

    if shift:
        out <<= np.array(shift, dtype=dtype)
    if components == 1:
        return out.reshape(height, width)
    return out.reshape(height, width, components)
