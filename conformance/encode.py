"""A minimal lossless JPEG encoder, for building the conformance corpus.

This exists for one reason: **a frame with restart markers cannot be built
out of random bytes.** A real encoder ends each interval by padding to a byte
boundary and emitting RST0-RST7 in turn, and only whatever produced the bits
knows where that boundary falls. Without this, every restart case in the
corpus was malformed — see conformance/README.md for how that was found.

It buys a second thing, which matters more. The expected pixels are now the
image that went in, so a case asserts a round trip rather than one
implementation's reading of the specification. Nobody has to take our
decoder's word for anything.

    encoded = encode(image, precision=12, predictor=4, restart_interval=7)
    assert (reference.decode(encoded) == image).all()

**This is not part of the library and must never become part of it.**
Plumbline decodes and refuses to write files. A decoder that is wrong is
visible the moment somebody looks at the image; an encoder that is wrong
corrupts an archive silently, years before anybody opens it again. The
asymmetry is the whole reason the project has the rule it has.

Implements ITU-T T.81 Annex H as the reference decoder reads it: predictors
1-7, precision 2-16, point transform, restart intervals, greyscale and
interleaved multi-component frames.
"""

from __future__ import annotations

import heapq

import numpy as np

MAX_LENGTH = 16


# --------------------------------------------------------------------------- #
# Huffman
# --------------------------------------------------------------------------- #

def huffman_lengths(frequencies: dict[int, int]) -> dict[int, int]:
    """Code lengths for a *complete* prefix code over the symbols used.

    Plain Huffman. The alphabet here is at most 17 symbols over images of a
    few dozen pixels, so lengths never approach the 16-bit ceiling and no
    length-limiting pass is needed — but the result is checked below rather
    than assumed, because a table that overflows would produce a frame no
    decoder should accept and the corpus would be asserting nonsense again.
    """
    if len(frequencies) == 1:
        # A one-symbol alphabet needs a bit to read — a zero-length code would
        # leave the decoder unable to advance — but a lone 1-bit code leaves
        # half the code space unclaimed, and an incomplete table is something
        # decoders are entitled to reject. Pair it with an unused symbol so
        # the table is complete and the frame is unambiguously legal.
        only = next(iter(frequencies))
        spare = 1 if only == 0 else 0
        return {only: 1, spare: 1}

    heap = [(count, index, symbol) for index, (symbol, count)
            in enumerate(sorted(frequencies.items()))]
    heapq.heapify(heap)
    lengths = {symbol: 0 for symbol in frequencies}
    tally = len(heap)

    while len(heap) > 1:
        left_count, _, left = heapq.heappop(heap)
        right_count, _, right = heapq.heappop(heap)
        for group in (left, right):
            for symbol in (group if isinstance(group, tuple) else (group,)):
                lengths[symbol] += 1
        merged = ((left if isinstance(left, tuple) else (left,))
                  + (right if isinstance(right, tuple) else (right,)))
        heapq.heappush(heap, (left_count + right_count, tally, merged))
        tally += 1

    longest = max(lengths.values())
    if longest > MAX_LENGTH:
        raise ValueError(f"code length {longest} exceeds the 16-bit limit")
    return lengths


def canonical_codes(lengths: dict[int, int]) -> tuple[list[int], list[int],
                                                      dict[int, tuple[int, int]]]:
    """(DHT counts, DHT symbols, {symbol: (code, length)}) in canonical order.

    This is the assignment every JPEG decoder rebuilds from a DHT segment:
    shortest codes first, ties broken by symbol value, the code incremented
    for each symbol and shifted left at each new length.
    """
    counts = [0] * MAX_LENGTH
    for length in lengths.values():
        counts[length - 1] += 1

    symbols: list[int] = []
    codes: dict[int, tuple[int, int]] = {}
    code = 0
    for length in range(1, MAX_LENGTH + 1):
        for symbol in sorted(s for s, l in lengths.items() if l == length):
            symbols.append(symbol)
            codes[symbol] = (code, length)
            code += 1
        code <<= 1

    kraft = sum(2 ** (MAX_LENGTH - length) for length in lengths.values())
    if kraft != 2 ** MAX_LENGTH:
        raise ValueError(f"table is not complete (Kraft sum {kraft / 2 ** MAX_LENGTH})")
    return counts, symbols, codes


# --------------------------------------------------------------------------- #
# differences
# --------------------------------------------------------------------------- #

def category(difference: int) -> int:
    """SSSS for a difference — the number of bits its magnitude needs."""
    if difference == 0:
        return 0
    return int(abs(difference)).bit_length()


def mantissa(difference: int, size: int) -> int:
    """The `size` bits that follow SSSS, as the decoder will read them back.

    Positive differences are sent as they are. Negative ones are sent as
    D + 2^size - 1, which lands below 2^(size-1) and is what tells the
    decoder to subtract on the way out.
    """
    if difference > 0:
        return difference
    return difference + (1 << size) - 1


# --------------------------------------------------------------------------- #
# the writer
# --------------------------------------------------------------------------- #

class _Writer:
    """Bits out, most significant first, with 0xFF stuffed as the format wants."""

    def __init__(self) -> None:
        self.out = bytearray()
        self.bits = 0
        self.held = 0

    def write(self, value: int, count: int) -> None:
        for shift in range(count - 1, -1, -1):
            self.held = (self.held << 1) | ((value >> shift) & 1)
            self.bits += 1
            if self.bits == 8:
                self.out.append(self.held)
                if self.held == 0xFF:
                    self.out.append(0x00)          # byte stuffing
                self.bits = 0
                self.held = 0

    def align(self) -> None:
        """Pad to a byte boundary with 1 bits, as the format requires."""
        while self.bits:
            self.write(1, 1)

    def restart(self, index: int) -> None:
        """End an interval: align, then RSTn — and markers are never stuffed."""
        self.align()
        self.out += bytes([0xFF, 0xD0 + (index & 7)])


# --------------------------------------------------------------------------- #

def encode(image: np.ndarray, precision: int, predictor: int = 1,
           restart_interval: int = 0, point_transform: int = 0) -> tuple[bytes, dict]:
    """Encode `image` and return (entropy data, table plan).

    `image` is (height, width) or (height, width, components), holding the
    values *before* the point transform is applied — the decoder shifts them
    left on the way out, so those are what the entropy coder carries.

    The table plan is {component index: (counts, symbols)} ready for a DHT
    segment; each component gets its own table, which is what real
    multi-component frames do.
    """
    if image.ndim == 2:
        image = image[:, :, None]
    height, width, components = image.shape
    modulo = 1 << precision
    default = 1 << (precision - 1 - point_transform)

    if image.min() < 0 or image.max() >= modulo:
        raise ValueError(f"samples must be within [0, {modulo})")

    # ---- pass one: the differences, and how often each category appears ----
    differences = np.zeros((height, width, components), dtype=np.int64)
    frequency: list[dict[int, int]] = [{} for _ in range(components)]
    since = 0

    for row in range(height):
        for col in range(width):
            restarted = False
            if restart_interval and since == restart_interval:
                since = 0
                restarted = True
            since += 1

            for comp in range(components):
                if restarted or (row == 0 and col == 0):
                    prediction = default
                elif row == 0:
                    prediction = int(image[0, col - 1, comp])
                elif col == 0:
                    prediction = int(image[row - 1, 0, comp])
                else:
                    ra = int(image[row, col - 1, comp])
                    rb = int(image[row - 1, col, comp])
                    rc = int(image[row - 1, col - 1, comp])
                    prediction = _predict(predictor, ra, rb, rc)

                # Any representative of the residue class decodes correctly;
                # take the one nearest zero so it needs the fewest bits and
                # never reaches SSSS=16, which carries no mantissa and is
                # exercised by its own case instead.
                difference = (int(image[row, col, comp]) - prediction) % modulo
                if difference > modulo // 2:
                    difference -= modulo

                differences[row, col, comp] = difference
                size = category(difference)
                frequency[comp][size] = frequency[comp].get(size, 0) + 1

    plan, codebooks = {}, []
    for comp in range(components):
        lengths = huffman_lengths(frequency[comp])
        counts, symbols, codes = canonical_codes(lengths)
        plan[comp] = (counts, symbols)
        codebooks.append(codes)

    # ---- pass two: emit ----
    writer = _Writer()
    since = 0
    restarts = 0

    for row in range(height):
        for col in range(width):
            if restart_interval and since == restart_interval:
                writer.restart(restarts)
                restarts += 1
                since = 0
            since += 1

            for comp in range(components):
                difference = int(differences[row, col, comp])
                size = category(difference)
                code, length = codebooks[comp][size]
                writer.write(code, length)
                if size:
                    writer.write(mantissa(difference, size), size)

    writer.align()
    return bytes(writer.out), plan


def _predict(selector: int, ra: int, rb: int, rc: int) -> int:
    """T.81 H.1.2.1 table H.1 — the same seven, written the same way."""
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
    raise ValueError(f"predictor {selector} is not defined")
