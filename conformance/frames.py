"""Build lossless JPEG frames on purpose, including the awkward ones.

Everything here is synthetic. No frame in this suite came from a scanner and
none contains patient data, which is the only reason the corpus can be
published at all — and publishing it is the point. A decoder can only be
shown correct against frames somebody else can also run.

The frames are built rather than collected because the interesting cases are
the ones real discs happen not to contain. A corpus of real images tests the
paths those particular machines took; a corpus built from the specification
tests the paths the specification allows.
"""

from __future__ import annotations

import numpy as np


def marker(code: int, payload: bytes = b"") -> bytes:
    if not payload:
        return bytes([0xFF, code])
    length = len(payload) + 2
    return bytes([0xFF, code, length >> 8, length & 0xFF]) + payload


def huffman_table(counts: list[int], symbols: list[int], identifier: int = 0) -> bytes:
    return marker(0xC4, bytes([identifier]) + bytes(counts) + bytes(symbols))


def greyscale_frame(width: int, height: int, precision: int, scan: bytes,
                    counts: list[int], symbols: list[int], predictor: int = 1,
                    restart_interval: int = 0, point_transform: int = 0) -> bytes:
    """One SOF3 frame, one component."""
    out = marker(0xD8)
    out += marker(0xC3, bytes([precision, height >> 8, height & 0xFF,
                               width >> 8, width & 0xFF, 1, 0, 0x11, 0]))
    out += huffman_table(counts, symbols)
    if restart_interval:
        out += marker(0xDD, bytes([restart_interval >> 8, restart_interval & 0xFF]))
    out += marker(0xDA, bytes([1, 0, 0x00, predictor, 0, point_transform]))
    return out + scan + marker(0xD9)


def colour_frame(width: int, height: int, precision: int, scan: bytes,
                 tables: dict[int, tuple[list[int], list[int]]],
                 components: list[tuple[int, int]],
                 predictor: int = 1, restart_interval: int = 0,
                 point_transform: int = 0) -> bytes:
    """An interleaved multi-component frame, one SOS covering every component.

    `components` is [(component id, Huffman table id)]. Interleaved means the
    samples arrive in component order per pixel, and — the part decoders get
    wrong — a restart interval counts *pixels*, not samples.
    """
    out = marker(0xD8)
    sof = bytes([precision, height >> 8, height & 0xFF,
                 width >> 8, width & 0xFF, len(components)])
    for component_id, _ in components:
        sof += bytes([component_id, 0x11, 0])
    out += marker(0xC3, sof)
    for identifier, (counts, symbols) in tables.items():
        out += huffman_table(counts, symbols, identifier)
    if restart_interval:
        out += marker(0xDD, bytes([restart_interval >> 8, restart_interval & 0xFF]))
    sos = bytes([len(components)])
    for component_id, table_id in components:
        sos += bytes([component_id, table_id << 4])
    sos += bytes([predictor, 0, point_transform])
    out += marker(0xDA, sos)
    return out + scan + marker(0xD9)


def table(shape: str, top: int) -> tuple[list[int], list[int]]:
    """A canonical Huffman table over SSSS 0..top, in three awkward shapes.

    * `staircase` gives SSSS s an (s+1)-bit code, so every symbol from 8 up
      needs more than sixteen bits together with its mantissa and cannot be
      fused into a single lookup. Decoders that assume a one-step table break
      here.
    * `narrow` does the opposite and spends bits fast.
    * `holes` is incomplete: some sixteen-bit windows match no code at all,
      and every decoder has to treat that the same way.
    """
    symbols = list(range(top + 1))
    sizes = [min(s + 1, 16) for s in symbols]
    if len(sizes) > 1:
        sizes[-1] = sizes[-2]                  # makes the Kraft sum exactly one
    if shape == "staircase":
        lengths = sizes
    elif shape == "narrow":
        lengths = sizes[::-1]
    elif shape == "holes":
        lengths = [5] * len(symbols)           # 2^5 slots for at most 17 codes
    else:
        raise ValueError(f"unknown table shape {shape!r}")
    counts = [0] * 16
    for length in lengths:
        counts[length - 1] += 1
    order = sorted(range(len(symbols)), key=lambda i: (lengths[i], i))
    return counts, [symbols[i] for i in order]


def stuffed(raw: bytes) -> bytes:
    """Byte-stuff so a 0xFF inside entropy data is read as data, not a marker."""
    return raw.replace(b"\xFF", b"\xFF\x00")


def scan_for(samples: int, seed: int) -> bytes:
    """Entropy data long enough that no sample can run off the end.

    Pseudo-random rather than an encoded picture: what is being exercised is
    the entropy decoder and the predictor, and random bits reach corners of
    the Huffman table that a smooth photograph never would. The seed makes it
    reproducible on any machine, which matters more here than realism.
    """
    generator = np.random.default_rng(seed)
    return stuffed(generator.integers(0, 256, samples * 4 + 16,
                                      dtype=np.uint8).tobytes())
