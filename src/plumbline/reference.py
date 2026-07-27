"""A lossless JPEG decoder we own outright.

Covers SOF3 — DICOM transfer syntaxes 1.2.840.10008.1.2.4.70 and .57 — which is
what nearly every disc from a Thai hospital arrives in, and the one format whose
only readily available decoder is GPL-3.0 and drags the whole application into
GPL with it.

Lossless JPEG has no DCT and no quantisation. Each sample is predicted from its
neighbours and only the difference is Huffman-coded, which is why a complete
decoder fits in a few hundred lines.

Three details decide whether an implementation is correct, and each was found
here by decoding real files and diffing against two independent decoders:

* a restart marker resets the predictor and re-aligns the stream to a byte
  boundary. Six of eleven real files carry them, usually one per row.
* SSSS = 16 is a special case (T.81 H.1.2.2): the difference is 32768 and no
  mantissa bits follow. Reading sixteen bits instead corrupts the rest of the
  image, because the predictor carries the error forward.
* reconstruction is modulo 2^P. Without the wrap, one overflowing sample drags
  every later sample with it.

Colour frames (three components, 1x1 sampling, one interleaved scan — the shape
every RGB ultrasound disc seen so far takes) are decoded with per-component
predictor state and per-component Huffman tables. Subsampled or non-interleaved
frames are refused, never guessed at.

This module is the readable reference: correct, dependency-free, and slow. It is
the oracle the accelerated path is tested against.
"""

from __future__ import annotations

import numpy as np

SOF3 = 0xC3
DHT = 0xC4
SOI = 0xD8
EOI = 0xD9
SOS = 0xDA
DRI = 0xDD

MAX_PRECISION = 16


class LosslessJpegError(ValueError):
    """The frame is not lossless JPEG, or is malformed."""


class _Huffman:
    """Canonical JPEG table, flattened to a lookup over sixteen peeked bits."""

    __slots__ = ("symbol", "length")

    def __init__(self, counts: list[int], symbols: list[int]):
        self.symbol = np.zeros(1 << 16, dtype=np.uint8)
        self.length = np.zeros(1 << 16, dtype=np.uint8)

        code = 0
        index = 0
        for bits in range(1, 17):
            for _ in range(counts[bits - 1]):
                if index >= len(symbols):
                    raise LosslessJpegError("Huffman table is truncated")
                low = code << (16 - bits)
                high = low + (1 << (16 - bits))
                self.symbol[low:high] = symbols[index]
                self.length[low:high] = bits
                index += 1
                code += 1
            code <<= 1


class _Bits:
    """The entropy-coded segment, with JPEG's byte stuffing already removed."""

    def __init__(self, data: bytes):
        out = bytearray()
        self.restarts: list[int] = []
        i, n = 0, len(data)
        while i < n:
            byte = data[i]
            if byte != 0xFF:
                out.append(byte)
                i += 1
                continue
            following = data[i + 1] if i + 1 < n else 0
            if following == 0x00:
                out.append(0xFF)          # a literal 0xFF in the data
                i += 2
            elif 0xD0 <= following <= 0xD7:
                self.restarts.append(len(out))
                i += 2
            else:
                break                     # EOI, or the next marker
        # Spare bytes so a sixteen-bit peek near the end cannot run off the
        # buffer; nothing real is ever read from them.
        out.extend(b"\x00" * 4)
        self.data = np.frombuffer(bytes(out), dtype=np.uint8)
        self.bit = 0

    def peek16(self) -> int:
        index = self.bit >> 3
        chunk = self.data[index:index + 3]
        value = int(chunk[0]) << 16
        if chunk.size > 1:
            value |= int(chunk[1]) << 8
        if chunk.size > 2:
            value |= int(chunk[2])
        return (value >> (8 - (self.bit & 7))) & 0xFFFF

    def skip(self, count: int) -> None:
        self.bit += count

    def difference(self, size: int) -> int:
        """Turn the next `size` bits back into a signed difference."""
        if size == 0:
            return 0
        if size == MAX_PRECISION:
            return 32768                  # T.81 H.1.2.2 — no bits follow
        value = 0
        for _ in range(size):
            index = self.bit >> 3
            value = (value << 1) | ((int(self.data[index]) >> (7 - (self.bit & 7))) & 1)
            self.bit += 1
        if value < (1 << (size - 1)):
            value -= (1 << size) - 1
        return value

    def seek_byte(self, offset: int) -> None:
        self.bit = offset * 8


def _predict(selector: int, ra: int, rb: int, rc: int) -> int:
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
    if selector == 7:
        return (ra + rb) >> 1
    raise LosslessJpegError(f"predictor {selector} is not defined")


def header(frame: bytes) -> dict:
    """Read the frame's parameters without decoding any pixels."""
    info: dict = {"restart_interval": 0}
    tables: dict[int, tuple[list[int], list[int]]] = {}
    i, n = 0, len(frame)

    while i < n - 1:
        if frame[i] != 0xFF:
            i += 1
            continue
        marker = frame[i + 1]
        i += 2
        if marker in (SOI, EOI) or 0xD0 <= marker <= 0xD7:
            continue
        if i + 1 >= n:
            break
        length = (frame[i] << 8) | frame[i + 1]
        segment = frame[i + 2:i + length]

        if marker == SOF3:
            info["precision"] = segment[0]
            info["height"] = (segment[1] << 8) | segment[2]
            info["width"] = (segment[3] << 8) | segment[4]
            info["components"] = segment[5]
            if len(segment) < 6 + 3 * info["components"]:
                raise LosslessJpegError("SOF3 segment is truncated")
            info["frame_ids"] = []
            info["sampling"] = []
            for c in range(info["components"]):
                base = 6 + c * 3
                info["frame_ids"].append(segment[base])
                info["sampling"].append(
                    (segment[base + 1] >> 4, segment[base + 1] & 0x0F))
        elif marker == DHT:
            pos = 0
            while pos < len(segment):
                identifier = segment[pos] & 0x0F
                counts = list(segment[pos + 1:pos + 17])
                total = sum(counts)
                tables[identifier] = (counts, list(segment[pos + 17:pos + 17 + total]))
                pos += 17 + total
        elif marker == DRI:
            info["restart_interval"] = (segment[0] << 8) | segment[1]
        elif marker == SOS:
            count = segment[0]
            info["scan_ids"] = [segment[1 + c * 2] for c in range(count)]
            info["table_ids"] = [segment[2 + c * 2] >> 4 for c in range(count)]
            info["predictor"] = segment[1 + count * 2]
            info["point_transform"] = segment[3 + count * 2] & 0x0F
            info["scan_offset"] = i + length
            info["tables"] = tables
            return info
        i += length

    raise LosslessJpegError("no SOS marker: not a lossless JPEG frame")


def decode(frame: bytes) -> np.ndarray:
    """Decode one lossless JPEG frame.

    Greyscale frames return (height, width); colour frames return
    (height, width, components) in the frame's component order, matching
    pydicom's pixel_array for interleaved (planar configuration 0) data.

    Each component carries its own predictor state — Ra/Rb/Rc are that
    component's own neighbours, never a neighbouring component — and may use
    its own Huffman table, as the real GE and Hitachi ultrasound discs do.

    Raises LosslessJpegError for anything this decoder does not handle, so a
    caller can fall back rather than receive a plausible-looking wrong image.
    """
    info = header(frame)
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

    # Map each scan component back to its slot in the frame, so the output
    # planes land in frame order whatever order the scan lists them in.
    frame_ids = info["frame_ids"]
    order = []
    for scan_id in info["scan_ids"]:
        if scan_id not in frame_ids or frame_ids.index(scan_id) in order:
            raise LosslessJpegError(
                f"scan component {scan_id} does not match the frame")
        order.append(frame_ids.index(scan_id))

    precision = info["precision"]
    height, width = info["height"], info["width"]
    predictor = info["predictor"]
    shift = info["point_transform"]
    interval = info["restart_interval"]

    built: dict[int, _Huffman] = {}
    tables = []
    for identifier in info["table_ids"]:
        if identifier not in info["tables"]:
            raise LosslessJpegError(f"Huffman table {identifier} is missing")
        if identifier not in built:
            built[identifier] = _Huffman(*info["tables"][identifier])
        tables.append(built[identifier])
    bits = _Bits(frame[info["scan_offset"]:])

    out = np.zeros((height, width, components), dtype=np.int64)
    default = 1 << (precision - 1 - shift)
    modulo = 1 << precision

    restarts = bits.restarts
    used = 0
    since = 0

    for row in range(height):
        for col in range(width):
            # With 1x1 sampling one MCU is one sample of every scan
            # component, so restart intervals count pixels.
            restarted = False
            if interval and since == interval:
                if used < len(restarts):
                    bits.seek_byte(restarts[used])
                    used += 1
                since = 0
                restarted = True
            since += 1

            for scan_slot, comp in enumerate(order):
                table = tables[scan_slot]
                peek = bits.peek16()
                size = int(table.symbol[peek])
                bits.skip(int(table.length[peek]))
                diff = bits.difference(size)

                if restarted or (row == 0 and col == 0):
                    prediction = default
                elif row == 0:
                    prediction = int(out[0, col - 1, comp])
                elif col == 0:
                    prediction = int(out[row - 1, 0, comp])
                else:
                    prediction = _predict(predictor,
                                          int(out[row, col - 1, comp]),
                                          int(out[row - 1, col, comp]),
                                          int(out[row - 1, col - 1, comp]))
                out[row, col, comp] = (prediction + diff) % modulo

    if shift:
        out <<= shift

    if components == 1:
        out = out[:, :, 0]
    return out.astype(np.uint8 if precision <= 8 else np.uint16)
