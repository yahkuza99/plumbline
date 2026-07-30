"""The same lossless JPEG decoder as :mod:`plumbline.reference`, some 500x faster.

`reference.py` is the oracle: readable, dependency-free, and slow enough (~0.2 Mpx/s)
that a single mammogram takes half a minute. This module decodes the same frames
at around 90 Mpx/s on one core by compiling the entropy loop with numba, and must
agree with the oracle bit for bit — `tests/test_turbo.py` decodes real discs
and random scans through both and diffs the results.

Three ideas carry the speedup:

* the Huffman code and the mantissa that follows it are decoded in one step. A
  table indexed by the next sixteen bits gives both the difference and the number
  of bits to consume, so the common sample costs one lookup and no branch.
  Where the pair is too long to fuse (code length + SSSS > 16) the mantissa is
  still read in one shot, because it always fits in what the buffer holds.
* the bit buffer lives in a register rather than in an object, so no attribute
  lookup and no Python integer allocation happens per sample.
* byte stuffing is undone in block copies between the 0xFF bytes, which numpy
  finds by vectorised compare. Walking the stream a byte at a time costs five
  times as much and, on a large frame, more than the entropy decoding itself.

Everything else is deliberately identical to the oracle, including the awkward
parts: the wrap applied inside the loop, where it is free, rather than over the
finished image, which costs a pass and the memory to hold it and is anyway a
different answer for predictors that average their neighbours; SSSS = 16
consuming no mantissa bits; a restart marker resetting the predictor *and*
re-aligning the stream to a byte boundary; and the whole first line of every
restart interval predicting from Ra, per T.81 §H.1.2.1, rather than only the
sample the marker precedes.

The samples are written straight out as uint8 or uint16. Decoding into int32 and
converting afterwards, as an intermediate version did, spends a tenth of the
run on a copy nobody asked for.

The two decoders no longer part company anywhere. A truncated scan and a code
the table never defines are refused by both, and the header, table and
entropy-segment checks are `reference`'s own, imported rather than rewritten —
a check that lived in only one of them would be a decoder that disagrees with
the oracle, which is the one thing this project cannot have.

numba is optional. Without it the module still imports and reports
``AVAILABLE = False``, and `decode` refuses, so a caller falls back to the oracle
rather than losing the ability to open a file.
"""

from __future__ import annotations

import numpy as np

from plumbline.reference import (
    MAX_PRECISION,
    LosslessJpegError,
    check_frame,
    check_scan,
    check_table,
    header,
)

try:                                       # pragma: no cover - depends on install
    from numba import njit, prange

    AVAILABLE = True
except Exception:                          # pragma: no cover - numba is optional
    AVAILABLE = False

    def njit(*args, **kwargs):
        """Stand-in that leaves the kernels as (unused) plain Python."""
        if args and callable(args[0]):
            return args[0]

        def decorate(function):
            return function

        return decorate

    prange = range


_WINDOW = 1 << 16         # the decoder peeks sixteen bits at a time
_SSSS_16 = 32768          # T.81 H.1.2.2: the difference for SSSS = 16


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #

def _canonical(counts: list[int], symbols: list[int]) -> tuple[np.ndarray, ...]:
    """Assign a code to every symbol, in the order T.81 Annex C prescribes."""
    codes = np.empty(len(symbols), dtype=np.int64)
    lengths = np.empty(len(symbols), dtype=np.int64)

    code = 0
    index = 0
    for bits in range(1, 17):
        for _ in range(counts[bits - 1]):
            if index >= len(symbols):
                raise LosslessJpegError("Huffman table is truncated")
            if symbols[index] > MAX_PRECISION:
                raise LosslessJpegError(f"SSSS={symbols[index]} is out of range")
            codes[index] = code
            lengths[index] = bits
            index += 1
            code += 1
        code <<= 1

    return codes[:index], lengths[:index], np.array(symbols[:index], dtype=np.int64)


@njit(cache=True)
def _fill(codes, lengths, sizes, consumed, values, length, ssss):
    """One pass over the 2^16 windows, filling both the fused and slow tables."""
    for entry in range(codes.size):
        bits = lengths[entry]
        size = sizes[entry]
        low = codes[entry] << (16 - bits)
        high = low + (1 << (16 - bits))

        total = bits if size == MAX_PRECISION else bits + size
        span = (1 << size) - 1
        half = (1 << (size - 1)) if size > 0 else 0

        for window in range(low, high):
            length[window] = bits
            ssss[window] = size
            if total > 16:
                continue                 # mantissa runs past the window
            consumed[window] = total
            if size == MAX_PRECISION:
                values[window] = _SSSS_16
                continue
            mantissa = (window >> (16 - total)) & span
            values[window] = mantissa - span if mantissa < half else mantissa


def _fuse(counts: list[int], symbols: list[int]) -> tuple[np.ndarray, ...]:
    """Build the fused lookup: window of sixteen bits -> difference, bits used.

    `consumed[w]` is zero where the symbol and its mantissa do not both fit in
    sixteen bits, and where the window matches no code at all; those windows take
    the unfused path, which reads `length[w]` and `ssss[w]` instead.
    """
    consumed = np.zeros(_WINDOW, dtype=np.uint8)
    values = np.zeros(_WINDOW, dtype=np.int32)
    length = np.zeros(_WINDOW, dtype=np.uint8)
    ssss = np.zeros(_WINDOW, dtype=np.uint8)
    _fill(*_canonical(counts, symbols), consumed, values, length, ssss)
    return consumed, values, length, ssss


# --------------------------------------------------------------------------- #
# entropy-coded segment
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _unstuff(source, marks):
    """Strip JPEG byte stuffing; return the payload and the restart offsets.

    `marks` holds every 0xFF in the source, found with a vectorised compare
    outside — byte-at-a-time scanning here costs five times as much. Everything
    between two of them is a block copy. A 0xFF is always the first byte of its
    pair: were the next byte 0xFF too, the scan would already have stopped.
    """
    out = np.empty(source.size, dtype=np.uint8)
    restarts = np.empty(marks.size, dtype=np.int64)
    written = 0
    found = 0
    prev = 0
    total = source.size

    for entry in range(marks.size):
        position = marks[entry]
        following = source[position + 1] if position + 1 < total else 0
        stuffed = following == 0x00
        keep = position + 1 if stuffed else position   # a literal 0xFF in the data
        span = keep - prev
        out[written:written + span] = source[prev:keep]
        written += span
        if not stuffed:
            if not 0xD0 <= following <= 0xD7:
                return out[:written], restarts[:found]   # EOI, or the next marker
            restarts[found] = written
            found += 1
        prev = position + 2

    # `prev` overshoots when the source ends on a lone 0xFF, which counts as
    # data: there is no following byte to skip, and no tail left to copy.
    span = total - prev if prev < total else 0
    out[written:written + span] = source[prev:prev + span]
    return out[:written + span], restarts[:found]


def _destuff(scan: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _unstuff(scan, np.flatnonzero(scan == 0xFF).astype(np.int64))


@njit(inline="always")
def _refill(data, position, buffer, held):
    """Top the bit buffer up past 32 bits. Past the end the stream reads zero."""
    total = data.size
    while held <= 32:
        byte = np.uint64(data[position]) if position < total else np.uint64(0)
        buffer = (buffer << np.uint64(8)) | byte
        position += 1
        held += 8
    return position, buffer, held


@njit(inline="always")
def _difference(data, position, buffer, held, consumed, values, length, ssss):
    """Decode one Huffman symbol and its mantissa into a signed difference."""
    position, buffer, held = _refill(data, position, buffer, held)
    window = np.int64((buffer >> np.uint64(held - 16)) & np.uint64(0xFFFF))

    bits = np.int64(consumed[window])
    if bits != 0:                        # the fused, overwhelmingly common path
        return np.int64(values[window]), position, buffer, held - bits, False

    # Code and mantissa are too long to share one window, but never too long for
    # the buffer: a refill leaves at least 33 bits and the pair takes at most 32.
    # SSSS = 16 cannot arrive here — its code alone always fits — so the mantissa
    # is always the next `size` bits.
    #
    # The one other way to land here is a window matching none of the table's
    # codes, whose entry is zero throughout. A table may legitimately leave code
    # space unclaimed, but bits that fall in it are not bits that table encoded,
    # so say so rather than consume nothing and call the difference zero.
    code_length = np.int64(length[window])
    held -= code_length
    size = np.int64(ssss[window])
    span = (np.int64(1) << size) - 1
    mantissa = np.int64((buffer >> np.uint64(held - size)) & np.uint64(span))
    if mantissa <= (span >> 1):          # T.81 H.1.2.2, the negative half
        mantissa -= span
    return mantissa, position, buffer, held - size, code_length == 0


@njit(inline="always")
def _predicted(selector, ra, rb, rc):
    """T.81 H.1.2.1. The selector is validated before the loop is entered."""
    if selector == 1:
        return ra
    if selector == 2:
        return rb
    if selector == 3:
        return rc
    if selector == 4:
        return ra + rb - rc
    if selector == 5:
        return ra + ((rb - rc) >> 1)
    if selector == 6:
        return rb + ((ra - rc) >> 1)
    return (ra + rb) >> 1                 # selector 7


# --------------------------------------------------------------------------- #
# scan
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _scan(out, width, height, data, restarts, interval, selector,
          consumed, values, length, ssss, default, mask, ends, faults):
    """Decode a whole scan sequentially, writing samples in the output dtype."""
    buffer = np.uint64(0)
    held = np.int64(0)
    position = np.int64(0)
    index = 0
    used = 0
    since = 0
    undefined = False
    # T.81 §H.1.2.1 puts the whole first line of every restart interval back on
    # Ra, not just the sample the marker precedes; see `reference`, which this
    # must agree with sample for sample. `check_frame` has refused any interval
    # that does not start on a row boundary, so this can only rise at column 0.
    ra_line = True

    for row in range(height):
        if row:
            ra_line = False
        for col in range(width):
            restarted = False
            if interval != 0 and since == interval:
                if used < restarts.size:
                    position = restarts[used]
                    buffer = np.uint64(0)
                    held = np.int64(0)
                    used += 1
                since = 0
                restarted = True
                ra_line = True
            since += 1

            diff, position, buffer, held, missing = _difference(
                data, position, buffer, held, consumed, values, length, ssss)
            undefined = undefined or missing

            if restarted or (row == 0 and col == 0):
                prediction = default
            elif col == 0:
                prediction = np.int64(out[index - width])
            elif ra_line:
                prediction = np.int64(out[index - 1])
            else:
                prediction = _predicted(selector,
                                        np.int64(out[index - 1]),
                                        np.int64(out[index - width]),
                                        np.int64(out[index - width - 1]))
            out[index] = (prediction + diff) & mask
            index += 1

    ends[0] = position * 8 - held        # the bit just past the last one used
    faults[0] = 1 if undefined else 0


@njit(cache=True, parallel=True)
def _scan_intervals(out, width, starts, counts, offsets, data,
                    consumed, values, length, ssss, default, mask, ends, faults):
    """Decode restart intervals concurrently.

    Only sound when every interval begins on a row boundary and the predictor is
    Ra, which together mean no sample depends on an interval decoded elsewhere.
    """
    for segment in prange(starts.size):
        buffer = np.uint64(0)
        held = np.int64(0)
        position = np.int64(starts[segment])
        index = offsets[segment]
        col = 0
        undefined = False

        for pixel in range(counts[segment]):
            diff, position, buffer, held, missing = _difference(
                data, position, buffer, held, consumed, values, length, ssss)
            undefined = undefined or missing

            if pixel == 0:
                prediction = default
            elif col == 0:
                prediction = np.int64(out[index - width])
            else:
                prediction = np.int64(out[index - 1])
            out[index] = (prediction + diff) & mask

            index += 1
            col += 1
            if col == width:
                col = 0

        ends[segment] = position * 8 - held
        faults[segment] = 1 if undefined else 0


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def _plan(interval: int, width: int, npix: int, restarts: np.ndarray):
    """Split the scan into restart intervals, or return None if it cannot be."""
    if interval <= 0 or interval % width or npix == 0:
        return None
    segments = -(-npix // interval)
    if restarts.size < segments - 1:       # missing markers: byte offsets unknown
        return None

    offsets = np.arange(segments, dtype=np.int64) * interval
    starts = np.empty(segments, dtype=np.int64)
    starts[0] = 0
    starts[1:] = restarts[:segments - 1]
    counts = np.minimum(interval, npix - offsets)
    return starts, counts, offsets


def _run(out, info, data, restarts, tables, default, mask, parallel):
    """Decode the scan into `out`.

    Returns where each kernel left the bit stream, and whether any of them met
    a window matching none of the table's codes.
    """
    width, height = info["width"], info["height"]
    interval = info["restart_interval"]
    plan = _plan(interval, width, out.size, restarts) \
        if parallel and info["predictor"] == 1 else None

    ends = np.zeros(1 if plan is None else plan[0].size, dtype=np.int64)
    faults = np.zeros(ends.size, dtype=np.int64)
    if plan is None:
        _scan(out, width, height, data, restarts, interval, info["predictor"],
              *tables, default, mask, ends, faults)
    else:
        _scan_intervals(out, width, *plan, data, *tables, default, mask,
                        ends, faults)
    return ends, faults


def _validate(info: dict) -> None:
    components = info.get("components", 0)
    if components != 1:
        raise LosslessJpegError(
            f"{components}-component frames are not supported yet")
    check_frame(info)


def decode(frame: bytes, parallel: bool = False) -> np.ndarray:
    """Decode one lossless JPEG frame, identically to `plumbline.reference.decode`.

    Raises LosslessJpegError for anything this decoder does not handle — numba
    missing included — so a caller falls back rather than receiving a
    plausible-looking wrong image.
    """
    if not AVAILABLE:
        raise LosslessJpegError("numba is not installed: no accelerated decoder")

    info = header(frame)
    _validate(info)

    precision = info["precision"]
    height, width = info["height"], info["width"]
    shift = info["point_transform"]
    dtype = np.uint8 if precision <= 8 else np.uint16

    identifiers = info["table_ids"]
    if not identifiers or identifiers[0] not in info["tables"]:
        raise LosslessJpegError("the scan names a Huffman table the frame "
                                "does not define")
    check_table(*info["tables"][identifiers[0]])
    tables = _fuse(*info["tables"][identifiers[0]])

    check_scan(frame[info["scan_offset"]:], height * width,
               info["restart_interval"])
    data, restarts = _destuff(np.frombuffer(frame, dtype=np.uint8,
                                            offset=info["scan_offset"]))
    out = np.empty(height * width, dtype=dtype)
    ends, faults = _run(out, info, data, restarts, tables,
                        np.int64(1) << (precision - 1 - shift),
                        (np.int64(1) << precision) - 1, parallel)

    # A well-formed scan never asks for a bit the frame does not contain. When it
    # does, the image is truncated and every sample after the break is invented,
    # so refuse instead of handing back an image that merely looks decoded.
    if int(ends.max()) > data.size * 8:
        raise LosslessJpegError("the entropy-coded data ends before the image does")
    if int(faults.max()):
        raise LosslessJpegError(
            "the scan contains a code the frame's Huffman table does not define")

    out = out.reshape(height, width)
    if shift:
        out <<= np.array(shift, dtype=dtype)
    return out


def _smallest(precision: int, restart: bool) -> bytes:
    """A valid two-by-two frame of zeros — the least work that compiles a kernel."""
    out = bytes([0xFF, 0xD8])
    out += bytes([0xFF, 0xC3, 0, 11, precision, 0, 2, 0, 2, 1, 0, 0x11, 0])
    out += bytes([0xFF, 0xC4, 0, 20, 0, 1] + [0] * 15 + [0])   # symbol 0 is "0"
    if restart:
        out += bytes([0xFF, 0xDD, 0, 4, 0, 2])
    out += bytes([0xFF, 0xDA, 0, 8, 1, 0, 0x00, 1, 0, 0])
    out += b"\x00\xFF\xD0\x00" if restart else b"\x00"
    return out + bytes([0xFF, 0xD9])


def warmup(parallel: bool = False) -> None:
    """Compile the kernels ahead of the first real frame.

    numba caches to `__pycache__`, so this is a one-off cost per install unless
    that directory is read-only. Call it off the UI thread if latency matters.
    """
    if not AVAILABLE:
        return
    for precision in (8, MAX_PRECISION):        # one kernel per output dtype
        decode(_smallest(precision, False))
        if parallel:
            decode(_smallest(precision, True), parallel=True)
