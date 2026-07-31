"""A lossless JPEG decoder we own outright.

Covers SOF3 — DICOM transfer syntaxes 1.2.840.10008.1.2.4.70 and .57 — which is
what nearly every disc from a Thai hospital arrives in, and the one format whose
only readily available decoder is GPL-3.0 and drags the whole application into
GPL with it.

Lossless JPEG has no DCT and no quantisation. Each sample is predicted from its
neighbours and only the difference is Huffman-coded, which is why a complete
decoder fits in a few hundred lines.

Four details decide whether an implementation is correct:

* a restart marker resets the predictor and re-aligns the stream to a byte
  boundary. Six of eleven real files carry them, usually one per row.
* a restart interval does not merely reset the prediction *value*. T.81
  H.1.2.1 puts the whole first line of every interval back on the horizontal
  predictor Ra, exactly as at the start of the scan; only the lines after it
  use the predictor the scan header selected. Resetting the value alone is
  invisible under predictor 1 — which is every real frame in our corpus — and
  wrong under 2-7, where the first line of each interval would otherwise
  predict from a row belonging to the previous interval.
* SSSS = 16 is a special case (T.81 H.1.2.2): the difference is 32768 and no
  mantissa bits follow. Reading sixteen bits instead corrupts the rest of the
  image, because the predictor carries the error forward.
* the difference is added to the prediction modulo 2^16 (T.81 H.1.2.1), not
  modulo 2^P. The mask applied below is 2^P, which is the same number for
  every conforming frame — a sample is below 2^P and 2^P divides 2^16, so the
  residue class has one representative in range — and keeps the result inside
  the precision the frame declares. What must not be done is take the
  *difference* modulo 2^P when encoding; see conformance/encode.py.

Colour frames (three components, 1x1 sampling, one interleaved scan — the shape
every RGB ultrasound disc seen so far takes) are decoded with per-component
predictor state and per-component Huffman tables. Subsampled or non-interleaved
frames are refused, never guessed at.

`check_frame`, `check_table` and `check_scan` hold everything a frame must
satisfy before a bit is read. They live here rather than in each decoder
because `turbo` and `native` import them: a check present in only one decoder
is a decoder that disagrees with the oracle about which files exist.

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
RST0 = 0xD0
RST7 = 0xD7

MAX_PRECISION = 16


class LosslessJpegError(ValueError):
    """The frame is not lossless JPEG, or is malformed."""


# --------------------------------------------------------------------------- #
# what a frame must satisfy before a single bit is read
#
# These three live here rather than in each decoder because all three decoders
# must refuse the same frames for the same reasons. `native` and `turbo` call
# them too; a check that exists in only one of them is a decoder that disagrees
# with the oracle, which is the one thing this project cannot have.
# --------------------------------------------------------------------------- #

def check_frame(info: dict) -> None:
    """Refuse header combinations no conforming frame can carry."""
    precision = info["precision"]
    if not 1 <= precision <= MAX_PRECISION:
        raise LosslessJpegError(f"precision {precision} is out of range")
    if not 1 <= info["predictor"] <= 7:
        raise LosslessJpegError(f"predictor {info['predictor']} is not defined")
    if info["point_transform"] >= precision:
        raise LosslessJpegError("point transform is larger than the precision")

    # ITU-T T.81 §B.2.2 table B.2: X, the number of samples per line, runs
    # 1..65535. Zero samples per line is not an empty image — the frame still
    # claims Y lines of them — so there is nothing here to return and nothing
    # to guess at either.
    if info["width"] < 1:
        raise LosslessJpegError("the frame is zero samples wide (T.81 §B.2.2 "
                                "allows X = 1..65535)")
    # Y = 0 is legal in T.81 §B.2.2 only because a DNL marker supplies the line
    # count later. This decoder does not implement DNL, so a frame that defers
    # its height is a frame whose height we do not know.
    if info["height"] < 1:
        raise LosslessJpegError("the frame declares no lines; its height would "
                                "come from a DNL marker, which is not supported")

    # T.81 §H.1.1: "For the lossless processes the restart interval shall be an
    # integer multiple of the number of MCU in an MCU-row", and table B.7 gives
    # Ri for lossless as n x MCUR. With 1x1 sampling one MCU is one pixel, so
    # MCUR is the width.
    #
    # This is refused rather than absorbed because the rule it protects is the
    # one below: §H.1.2.1 puts "the first line" of every restart interval on
    # Ra, and an interval starting halfway along a row has no first line that
    # the specification names. Three readings are possible (the tail of the row
    # it landed in, the next `width` samples, or none at all) and libjpeg
    # reproduces none of them — measured, not assumed. Every reading is
    # therefore a guess, and pixels nobody else computes are exactly what this
    # decoder exists not to return. No frame in the 81,172 real ones this
    # project has decoded uses such an interval; every one that restarts at all
    # restarts once per row.
    interval = info.get("restart_interval", 0)
    if interval and interval % info["width"]:
        raise LosslessJpegError(
            f"the restart interval is {interval} MCU, which is not a whole "
            f"number of {info['width']}-MCU rows; T.81 §H.1.1 requires an "
            "integer multiple of the MCU per row, and where an interval starts "
            "mid-row the predictor rule in §H.1.2.1 has no defined meaning")


def scan_slots(info: dict) -> list[int]:
    """Map each scan component back to its slot in the frame, refusing first.

    This lives here, and `native` and `turbo` import it, because it used to be
    written out twice. `turbo` had a third copy that omitted the subsampling
    check entirely, so it accepted frames the other two refused — in a project
    whose own documentation says all three refuse the same frames for the same
    reasons. A check that exists in one decoder is the check that drifts.
    """
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


def check_table(counts: list[int], symbols: list[int]) -> None:
    """Refuse a Huffman table that cannot be the one the encoder used.

    Three ways a table lies about itself, all of which a permissive decoder
    absorbs into plausible wrong pixels rather than a diagnosis:

    * it promises more symbols than the segment carries;
    * it promises more codes of some length than that length has room for
      (Kraft sum above one). The extra codes have nowhere to live, so the
      assignment every decoder rebuilds from T.81 §B.2.4.2 runs off the end
      and each decoder loses a different symbol;
    * it lists an SSSS above 16, which is not a mantissa width at all.

    What is deliberately *not* checked is SSSS against the frame's precision.
    It looks like it should hold — a 10-bit sample surely cannot differ from
    its prediction by twelve bits — and it does not: differences are taken
    modulo 2^16 rather than modulo 2^P, so any representative of the residue
    class is legal and encoders pick whichever codes shortest. The Hologic
    mammogram in the test data is 10-bit and its table lists SSSS = 12; two
    independent decoders and this one agree on its pixels to the bit. A check
    that rejected it would refuse a real diagnostic image to catch a synthetic
    one, which is the wrong trade in a project whose reason to exist is that
    the alternative decoder was GPL rather than that it was wrong.
    """
    total = sum(counts)
    if len(counts) != 16 or total > len(symbols):
        raise LosslessJpegError("Huffman table is truncated")

    code = 0
    for bits in range(1, 17):
        code += counts[bits - 1]
        if code > (1 << bits):
            raise LosslessJpegError(
                f"Huffman table is over-subscribed: {code} codes of {bits} "
                f"bits or fewer, where only {1 << bits} can exist")
        code <<= 1

    for symbol in symbols[:total]:
        if symbol > MAX_PRECISION:
            raise LosslessJpegError(f"SSSS={symbol} is out of range")


def check_scan(scan: bytes, mcus: int, interval: int) -> None:
    """Refuse an entropy-coded segment that is not the one the encoder wrote.

    Only the 0xFF bytes are looked at, and every 0xFF in the segment is one:
    the byte stuffed after a data 0xFF is 0x00, and no marker code is 0xFF, so
    a 0xFF is never the second byte of a pair except in a run of fill bytes,
    where it opens a pair of its own. That makes the classification below
    position-independent, and one vectorised pass enough.

    That pass costs the accelerated decoders about 6% on the test discs (5 ms
    on an 8.4 MB scan, against ~73 ms to decode it), because it reads every
    byte a second time. `native`'s C destuffing already walks the same 0xFF
    bytes and could report all of this for nothing, and the 6% is the price of
    not writing this logic twice in two languages where the two copies could
    drift. Correctness first is the whole premise; if that ever stops being
    the right trade, this is where to look.
    """
    data = np.frombuffer(scan, dtype=np.uint8)
    marks = np.flatnonzero(data == 0xFF)
    if marks.size:
        # A 0xFF as the very last byte has no partner; the decoders keep it as
        # data, so read its follower as 0x00 and let it pass as stuffing.
        following = np.where(marks + 1 < data.size,
                             data[np.minimum(marks + 1, data.size - 1)], 0)
    else:
        following = marks

    restart = (following >= RST0) & (following <= RST7)
    # A marker of any other kind ends the segment: everything past it belongs
    # to the next one and is none of this scan's business. Fill bytes (0xFF)
    # only ever precede such a marker, so they end it too.
    ends = ~restart & (following >= 0xC0)
    stop = int(np.argmax(ends)) if ends.any() else marks.size

    inside = following[:stop]
    restart = restart[:stop]
    # T.81 §B.1.1.5: the encoder stuffs a zero byte after every 0xFF it writes,
    # so inside the segment a 0xFF is followed by 0x00, by an RSTn, or by the
    # marker that ends it. Anything else means these are not the encoder's
    # bytes, and a decoder that stops reading there invents the rest of the
    # image.
    stray = inside[(inside != 0x00) & ~restart]
    if stray.size:
        raise LosslessJpegError(
            f"0xFF 0x{int(stray[0]):02X} inside the entropy-coded data is "
            "neither byte stuffing nor a marker")

    found = int(restart.sum())
    if found:
        # RSTn carries a modulo-8 counter for exactly one purpose: to let a
        # decoder notice that markers were lost. A jump in the sequence says
        # the bytes between here and the last marker are not the bytes the
        # encoder wrote, whatever they decode to.
        order = (inside[restart].astype(np.int64) - RST0)
        expected = np.arange(found, dtype=np.int64) % 8
        wrong = np.flatnonzero(order != expected)
        if wrong.size:
            index = int(wrong[0])
            raise LosslessJpegError(
                f"restart marker {index} is RST{int(order[index])}, not "
                f"RST{int(expected[index])}: markers must cycle RST0-RST7 in "
                "order, so one out of sequence means some were lost")

    # With 1x1 sampling one MCU is one pixel, so an interval that divides the
    # image into N pieces needs exactly N-1 markers to say where they start.
    # A frame with no DRI declares no interval and therefore needs none.
    required = -(-mcus // interval) - 1 if interval > 0 else 0

    # Exactly, in both directions. This was `found < required`, guarded by
    # `if interval > 0`, and each half let a whole class of frame through:
    #
    # * **Too many markers.** A transcoder rewrites DRI while the entropy
    #   stream keeps its original spacing, so the header and the data disagree
    #   about where the restarts are. The decoder resets the predictor at the
    #   interval the header names and re-syncs at the markers the stream
    #   carries, and the two stop lining up after the first one.
    # * **Markers with no DRI at all.** A header rewrite drops the 0xFFDD
    #   segment, or a frame is lifted out of a multi-frame object where only
    #   the first fragment carried it. With no interval declared the check did
    #   not run, the markers were treated as data, and the padding that aligns
    #   each one to a byte boundary was decoded as entropy.
    #
    # Both were measured at three quarters of the pixels wrong on an 8x4
    # frame, returned as a successful decode by every engine.
    if found != required:
        if found > required:
            raise LosslessJpegError(
                f"the frame declares a restart interval of {interval} but "
                f"carries {found} restart markers where that interval needs "
                f"{required}: the header and the entropy data disagree about "
                "where the restarts are, so neither can be trusted")
        raise LosslessJpegError(
            f"the frame declares a restart interval of {interval} but "
            f"carries {found} of the {required} restart markers that "
            "interval needs")


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


_PEEK_PADDING = 4        # spare bytes so a sixteen-bit peek cannot run off the end


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
        # buffer; nothing real is ever read from them, and `supplied` is what
        # the frame actually carried.
        self.supplied = len(out) * 8
        out.extend(b"\x00" * _PEEK_PADDING)
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
        # Any number of 0xFF bytes may precede a marker (T.81 B.1.1.3). Taking
        # the byte after the first one as the marker code desynchronised the
        # whole parse on a conforming file, and reported it as having no SOS.
        while i + 1 < n and frame[i + 1] == 0xFF:
            i += 1
        marker = frame[i + 1]
        i += 2
        if marker in (SOI, EOI) or 0xD0 <= marker <= 0xD7:
            continue
        if i + 1 >= n:
            break
        length = (frame[i] << 8) | frame[i + 1]
        segment = frame[i + 2:i + length]

        def need(count: int, what: str) -> None:
            """Refuse a segment too short for the field about to be read.

            Each of these used to index first and check afterwards, so a
            truncated header raised IndexError — which callers told to expect
            LosslessJpegError do not catch, and which takes their process with
            it. A malformed file must produce a diagnosis, not a crash.
            """
            if len(segment) < count:
                raise LosslessJpegError(
                    f"{what} segment is truncated: {len(segment)} bytes, "
                    f"needs at least {count}")

        if marker == SOF3:
            need(6, "SOF3")
            info["precision"] = segment[0]
            info["height"] = (segment[1] << 8) | segment[2]
            info["width"] = (segment[3] << 8) | segment[4]
            info["components"] = segment[5]
            need(6 + 3 * info["components"], "SOF3")
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
                if len(segment) < pos + 17:
                    raise LosslessJpegError(
                        "DHT segment ends inside a table's code-length counts")
                identifier = segment[pos] & 0x0F
                counts = list(segment[pos + 1:pos + 17])
                total = sum(counts)
                if len(segment) < pos + 17 + total:
                    raise LosslessJpegError(
                        f"DHT segment declares {total} symbols and carries "
                        f"{len(segment) - pos - 17}")
                tables[identifier] = (counts, list(segment[pos + 17:pos + 17 + total]))
                pos += 17 + total
        elif marker == DRI:
            need(2, "DRI")
            info["restart_interval"] = (segment[0] << 8) | segment[1]
        elif marker == SOS:
            need(1, "SOS")
            count = segment[0]
            need(4 + count * 2, "SOS")
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
    # In this order in all three decoders, so that a frame wrong in more than
    # one way is reported the same way by each of them.
    order = scan_slots(info)
    check_frame(info)

    components = info["components"]
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
            check_table(*info["tables"][identifier])
            built[identifier] = _Huffman(*info["tables"][identifier])
        tables.append(built[identifier])

    check_scan(frame[info["scan_offset"]:], height * width, interval)
    bits = _Bits(frame[info["scan_offset"]:])
    if bits.data.size <= _PEEK_PADDING:        # only the peek padding is left
        raise LosslessJpegError("there is no entropy-coded data in the scan")

    out = np.zeros((height, width, components), dtype=np.int64)
    # What the scan carries is the image after a right shift by Pt, so its
    # samples are P-Pt bits wide, not P. The default prediction on the line
    # below already subtracts the shift; the modulus did not, and the two
    # sitting one line apart disagreed about how wide a sample is.
    #
    # For a conforming frame the two are the same number, because every sample
    # already fits in P-Pt bits and the wider mask never has anything to do.
    # For a frame whose Al field is wrong — one nibble — the wide mask let
    # through samples up to 2^P, which `out <<= shift` then multiplied past the
    # precision the frame declares: a frame calling itself 12-bit came back
    # holding 34,800, which is eight times what 12 bits can express. Nothing
    # downstream can tell that from data, so a viewer windows it as if it were.
    default = 1 << (precision - 1 - shift)
    modulo = 1 << (precision - shift)

    restarts = bits.restarts
    used = 0
    since = 0
    # T.81 §H.1.2.1: "The one-dimensional horizontal predictor (prediction
    # sample Ra) is used for the first line of samples at the start of the scan
    # and at the beginning of each restart interval. The selected predictor is
    # used for all other lines." So this is a property of the whole line, not
    # of the one sample the restart landed on. `check_frame` has already
    # refused any interval that does not begin on a row boundary, so the flag
    # can only be raised at column zero.
    ra_line = True

    for row in range(height):
        if row:
            ra_line = False
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
                ra_line = True
            since += 1

            for scan_slot, comp in enumerate(order):
                table = tables[scan_slot]
                peek = bits.peek16()
                size = int(table.symbol[peek])
                length = int(table.length[peek])
                # A table need not claim the whole code space — encoders list
                # only the SSSS values they used, and T.81 §B.2.4.2 does not
                # require otherwise. But the bits in front of us matching none
                # of the codes it *does* claim means these are not the bits
                # that table encoded. Consuming nothing and calling the
                # difference zero, which is what a table of zeros quietly
                # does, produces a whole image out of a stream we cannot read.
                if length == 0:
                    raise LosslessJpegError(
                        f"the scan contains a code Huffman table "
                        f"{info['table_ids'][scan_slot]} does not define "
                        f"(at bit {bits.bit} of the entropy-coded data)")
                bits.skip(length)
                diff = bits.difference(size)

                # §H.1.2.1, in the order the clause states its cases: the
                # default value opens the first line and every restart
                # interval; Rb opens every other line; Ra carries the first
                # line of the scan and of each interval; the selected
                # predictor carries everything else.
                if restarted or (row == 0 and col == 0):
                    prediction = default
                elif col == 0:
                    prediction = int(out[row - 1, 0, comp])
                elif ra_line:
                    prediction = int(out[row, col - 1, comp])
                else:
                    prediction = _predict(predictor,
                                          int(out[row, col - 1, comp]),
                                          int(out[row - 1, col, comp]),
                                          int(out[row - 1, col - 1, comp]))
                out[row, col, comp] = (prediction + diff) % modulo

    # A well-formed scan never asks for a bit the frame does not contain. When
    # it does, the image is truncated and every sample after the break was read
    # out of the padding above — so refuse, as `turbo` and `native` already do,
    # rather than hand back an image whose tail this decoder invented.
    if bits.bit > bits.supplied:
        raise LosslessJpegError("the entropy-coded data ends before the image does")

    if shift:
        out <<= shift

    if components == 1:
        out = out[:, :, 0]
    return out.astype(np.uint8 if precision <= 8 else np.uint16)
