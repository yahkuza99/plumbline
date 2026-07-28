r"""Tests for the compiled (C) lossless JPEG decoder.

The oracle is `plumbline.reference`, deliberately — not pylibjpeg, not any other
library, and not `turbo`. The native decoder exists only to produce what
the slow one produces, faster, so every test here decodes twice and diffs.

The frames come from the same three places as `tests/test_turbo.py`:
random scans across a range of Huffman tables, precisions, predictors and
restart intervals; real hospital discs when the test data folder is present;
and frames built by hand for the details that decide correctness. On top of
the turbo suite, colour frames are exercised too — this decoder takes the
interleaved multi-component frames that `turbo` refuses.

The two decoders no longer part company anywhere: a truncated scan, and a code
the scan's table never defines, are refused by both. Refusing is always allowed
here — but only together. One decoder returning pixels the other will not is
the failure this suite exists to catch.

The whole module is skipped where the shared library has not been built
(`python native/build.py`).
"""

import os
from pathlib import Path

import numpy as np
import pytest

import plumbline
from plumbline import native
from plumbline.reference import LosslessJpegError, header
from plumbline.reference import decode as decode_slowly

if not native.AVAILABLE:
    pytest.skip("the native library is not built (python native/build.py)",
                allow_module_level=True)

LOSSLESS_SYNTAXES = ("1.2.840.10008.1.2.4.70", "1.2.840.10008.1.2.4.57")


# --------------------------------------------------------------------------- #
# frames built by hand
# --------------------------------------------------------------------------- #

def _marker(code: int, payload: bytes = b"") -> bytes:
    if not payload:
        return bytes([0xFF, code])
    length = len(payload) + 2
    return bytes([0xFF, code, length >> 8, length & 0xFF]) + payload


def _huffman_table(counts: list[int], symbols: list[int], identifier: int = 0) -> bytes:
    return _marker(0xC4, bytes([identifier]) + bytes(counts) + bytes(symbols))


def _frame(width: int, height: int, precision: int, scan: bytes,
           counts: list[int], symbols: list[int], predictor: int = 1,
           restart_interval: int = 0, point_transform: int = 0) -> bytes:
    out = _marker(0xD8)
    out += _marker(0xC3, bytes([precision, height >> 8, height & 0xFF,
                                width >> 8, width & 0xFF, 1, 0, 0x11, 0]))
    out += _huffman_table(counts, symbols)
    if restart_interval:
        out += _marker(0xDD, bytes([restart_interval >> 8, restart_interval & 0xFF]))
    out += _marker(0xDA, bytes([1, 0, 0x00, predictor, 0, point_transform]))
    return out + scan + _marker(0xD9)


def _color_frame(width: int, height: int, precision: int, scan: bytes,
                 tables: dict[int, tuple[list[int], list[int]]],
                 components: list[tuple[int, int]],  # (component id, table id)
                 predictor: int = 1, restart_interval: int = 0,
                 point_transform: int = 0) -> bytes:
    """An interleaved multi-component frame, one SOS covering every component."""
    out = _marker(0xD8)
    sof = bytes([precision, height >> 8, height & 0xFF,
                 width >> 8, width & 0xFF, len(components)])
    for component_id, _ in components:
        sof += bytes([component_id, 0x11, 0])
    out += _marker(0xC3, sof)
    for identifier, (counts, symbols) in tables.items():
        out += _huffman_table(counts, symbols, identifier)
    if restart_interval:
        out += _marker(0xDD, bytes([restart_interval >> 8, restart_interval & 0xFF]))
    sos = bytes([len(components)])
    for component_id, table_id in components:
        sos += bytes([component_id, table_id << 4])
    sos += bytes([predictor, 0, point_transform])
    out += _marker(0xDA, sos)
    return out + scan + _marker(0xD9)


def _table(shape: str, top: int) -> tuple[list[int], list[int]]:
    """A canonical table over SSSS 0..top, in three deliberately awkward shapes.

    `staircase` gives SSSS s an (s+1)-bit code, so every symbol from 8 up needs
    more than sixteen bits together with its mantissa and cannot be fused into
    one lookup. `narrow` does the opposite, and spends bits fast. `holes` is
    incomplete: some sixteen-bit windows match no code at all, which both
    decoders have to treat the same way.
    """
    symbols = list(range(top + 1))
    sizes = [min(s + 1, 16) for s in symbols]
    if len(sizes) > 1:
        sizes[-1] = sizes[-2]                  # makes the Kraft sum exactly one
    if shape == "staircase":
        lengths = sizes
    elif shape == "narrow":
        lengths = sizes[::-1]
    else:
        lengths = [5] * len(symbols)           # 2^5 slots for at most 17 codes
    counts = [0] * 16
    for length in lengths:
        counts[length - 1] += 1
    order = sorted(range(len(symbols)), key=lambda i: (lengths[i], i))
    return counts, [symbols[i] for i in order]


def _stuffed(raw: bytes) -> bytes:
    """Byte-stuff data so a 0xFF inside it is read as data, not as a marker."""
    return raw.replace(b"\xFF", b"\xFF\x00")


def _scan_for(samples: int, seed: int) -> bytes:
    """Random entropy data long enough that no sample can run off the end."""
    generator = np.random.default_rng(seed)
    return _stuffed(generator.integers(0, 256, samples * 4 + 16,
                                       dtype=np.uint8).tobytes())


def _restarted(samples: int, mcus: int, interval: int, seed: int) -> bytes:
    """Random pieces joined by the restart markers a conforming frame carries.

    One marker per interval boundary and no more, cycling RST0-RST7. Both are
    load-bearing: a frame whose markers run out before its intervals do, or
    whose markers skip a number, is one where bytes went missing, and both
    decoders refuse it rather than reading on into the next interval.

    Each piece is long enough to finish the whole image on its own, so a
    difference between the decoders is a difference in how they read the
    stream and never one of them running dry first.
    """
    pieces = [_scan_for(samples, seed + piece)
              for piece in range(-(-mcus // interval))]
    out = pieces[0]
    for index, piece in enumerate(pieces[1:]):
        out += bytes([0xFF, 0xD0 + (index & 7)]) + piece
    return out


def _decoded_twice(frame: bytes) -> tuple:
    """Both decoders' answers: an array each, or the exception type raised."""
    try:
        slow = decode_slowly(frame)
    except Exception as error:                 # the oracle raises bare IndexError
        slow = type(error)
    try:
        fast = native.decode(frame)
    except Exception as error:
        fast = type(error)
    return slow, fast


def _identical(frame: bytes) -> bool:
    slow, fast = _decoded_twice(frame)
    assert isinstance(slow, np.ndarray), f"the oracle refused this frame: {slow}"
    assert isinstance(fast, np.ndarray), f"the native decoder refused this frame: {fast}"
    return (slow.dtype == fast.dtype and slow.shape == fast.shape
            and np.array_equal(slow, fast))


def _refused_together(frame: bytes) -> bool:
    """Both decoders must refuse — agreeing on a refusal is agreeing."""
    slow, fast = _decoded_twice(frame)
    assert not isinstance(slow, np.ndarray), "the oracle returned pixels for this frame"
    assert not isinstance(fast, np.ndarray), "the native decoder returned pixels"
    return slow is LosslessJpegError and fast is LosslessJpegError


# symbol 0 and symbol 16 are the two one-bit codes "0" and "1"
_COUNTS = [2] + [0] * 15
_SYMBOLS = [0, 16]


def _bits_to_bytes(bits: str) -> bytes:
    bits += "0" * (-len(bits) % 8)
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


# --------------------------------------------------------------------------- #
# the library is optional
# --------------------------------------------------------------------------- #

class TestTheLibraryIsOptional:
    def test_it_reports_itself_available_where_the_library_is_built(self):
        assert native.AVAILABLE is True

    def test_it_refuses_when_the_library_is_missing(self, monkeypatch):
        """Without the binary the application must lose speed, never a file."""
        monkeypatch.setattr(native, "AVAILABLE", False)
        with pytest.raises(LosslessJpegError):
            native.decode(_frame(2, 2, 8, b"\x00", _COUNTS, _SYMBOLS))


# --------------------------------------------------------------------------- #
# the details that decide correctness
# --------------------------------------------------------------------------- #

class TestAgreesOnTheHardCases:
    def test_all_zero_differences_give_a_flat_image(self):
        frame = _frame(4, 3, 8, _bits_to_bytes("0" * 12), _COUNTS, _SYMBOLS)
        assert np.all(native.decode(frame) == 1 << 7)
        assert _identical(frame)

    def test_ssss_16_consumes_no_mantissa_bits(self):
        frame = _frame(3, 2, 16, _bits_to_bytes("1" + "0" * 5), _COUNTS, _SYMBOLS)
        assert native.decode(frame)[0, 0] == 0     # (32768 + 32768) mod 2^16
        assert _identical(frame)

    def test_restart_marker_resets_the_prediction(self):
        scan = _bits_to_bytes("0" * 4) + bytes([0xFF, 0xD0]) + _bits_to_bytes("0" * 4)
        frame = _frame(4, 2, 8, scan, _COUNTS, _SYMBOLS, restart_interval=4)
        assert np.all(native.decode(frame) == 1 << 7)
        assert _identical(frame)

    def test_stuffed_ff_is_data_not_a_marker(self):
        frame = _frame(2, 1, 8, bytes([0xFF, 0x00]), _COUNTS, _SYMBOLS)
        assert _identical(frame)

    def test_a_scan_ending_on_a_lone_ff_keeps_that_byte(self):
        """The last byte has no partner to skip, and is data like any other."""
        frame = _frame(2, 1, 8, bytes([0x00, 0xFF]), _COUNTS, _SYMBOLS)
        assert _identical(frame)

    def test_the_output_dtype_follows_the_precision(self):
        counts, symbols = _table("staircase", 8)
        assert native.decode(
            _frame(4, 4, 8, _scan_for(16, 1), counts, symbols)).dtype == np.uint8
        counts, symbols = _table("staircase", 12)
        assert native.decode(
            _frame(4, 4, 12, _scan_for(16, 2), counts, symbols)).dtype == np.uint16

    def test_point_transform_shifts_after_prediction_not_before(self):
        counts, symbols = _table("staircase", 12)
        frame = _frame(5, 4, 12, _scan_for(20, 3), counts, symbols, point_transform=2)
        assert _identical(frame)
        assert native.decode(frame).max() % 4 == 0   # every sample was shifted

    def test_reconstruction_wraps_at_two_to_the_precision(self):
        """Predictor 7 averages neighbours, so a sample that overflowed feeds a
        wrong prediction into the next one — wrapping late is a different
        picture, not just a cast. Agreeing with the oracle on a scan built to
        overflow proves the wrap happens inside the loop."""
        counts, symbols = _table("narrow", 12)
        frame = _frame(6, 6, 12, _scan_for(36, 4), counts, symbols, predictor=7)
        assert _identical(frame)


class TestColourFrames:
    _SMALL = (_COUNTS, _SYMBOLS)

    def test_colour_decodes_to_height_width_components(self):
        scan = _bits_to_bytes("0" * 12)
        frame = _color_frame(2, 2, 8, scan, {0: self._SMALL},
                             [(1, 0), (2, 0), (3, 0)])
        decoded = native.decode(frame)
        assert decoded.shape == (2, 2, 3)
        assert _identical(frame)

    def test_each_component_predicts_from_its_own_samples(self):
        scan = _bits_to_bytes("101010")        # +32768 to one component only
        frame = _color_frame(2, 1, 16, scan, {0: self._SMALL},
                             [(1, 0), (2, 0)])
        assert _identical(frame)

    def test_each_component_uses_its_own_huffman_table(self):
        # The same two symbols, but on two-bit codes, so reading a component
        # with the wrong table desynchronises the stream rather than merely
        # renaming a symbol. Padded to four codes because a table that claims
        # only half its code space is one a random scan falls out of.
        second = ([0, 4] + [0] * 14, [0, 16, 1, 2])
        frame = _color_frame(2, 2, 12, _scan_for(12, 5),
                             {0: self._SMALL, 1: second},
                             [(1, 0), (2, 1), (3, 0)])
        assert _identical(frame)

    def test_scan_order_maps_back_to_frame_order(self):
        """The scan may list components in any order; planes land in frame
        order either way, exactly as the oracle lays them out."""
        scan = _scan_for(8, 6)
        forward = _color_frame(2, 2, 8, scan, {0: self._SMALL},
                               [(1, 0), (2, 0)])
        backward = _color_frame(2, 2, 8, scan, {0: self._SMALL},
                                [(2, 0), (1, 0)])
        assert _identical(forward)
        assert _identical(backward)

    def test_restart_interval_counts_pixels_not_samples(self):
        scan = _bits_to_bytes("0" * 6) + bytes([0xFF, 0xD0]) + _bits_to_bytes("0" * 6)
        frame = _color_frame(2, 2, 8, scan, {0: self._SMALL},
                             [(1, 0), (2, 0), (3, 0)], restart_interval=2)
        assert _identical(frame)

    def test_subsampled_frames_are_refused_not_guessed(self):
        frame = _color_frame(2, 2, 8, _bits_to_bytes("0" * 12), {0: self._SMALL},
                             [(1, 0), (2, 0), (3, 0)])
        frame = frame.replace(bytes([1, 0x11, 0]), bytes([1, 0x22, 0]), 1)
        with pytest.raises(LosslessJpegError):
            native.decode(frame)

    def test_non_interleaved_scans_are_refused_not_guessed(self):
        # three components in the frame, but a scan covering only one
        out = _marker(0xD8)
        out += _marker(0xC3, bytes([8, 0, 2, 0, 2, 3,
                                    1, 0x11, 0, 2, 0x11, 0, 3, 0x11, 0]))
        out += _huffman_table(*self._SMALL)
        out += _marker(0xDA, bytes([1, 1, 0x00, 1, 0, 0]))
        frame = out + _bits_to_bytes("0" * 4) + _marker(0xD9)
        with pytest.raises(LosslessJpegError):
            native.decode(frame)


class TestRefusesRatherThanGuesses:
    def test_garbage_is_refused(self):
        with pytest.raises(LosslessJpegError):
            native.decode(b"not a jpeg at all")

    def test_unknown_predictor_is_refused(self):
        frame = _frame(3, 3, 8, _bits_to_bytes("0" * 9), _COUNTS, _SYMBOLS,
                       predictor=9)
        with pytest.raises(LosslessJpegError):
            native.decode(frame)

    def test_a_truncated_scan_is_refused_not_padded_with_zeros(self):
        """Half a frame is half an image, however plausible the rest looks."""
        counts, symbols = _table("staircase", 12)
        frame = _frame(16, 16, 12, b"\x91\x37", counts, symbols)
        with pytest.raises(LosslessJpegError):
            native.decode(frame)

    def test_a_scan_naming_a_table_the_frame_lacks_is_refused(self):
        frame = _frame(2, 2, 8, b"\x00", _COUNTS, _SYMBOLS)
        frame = frame.replace(bytes([0xFF, 0xDA, 0, 8, 1, 0, 0x00]),
                              bytes([0xFF, 0xDA, 0, 8, 1, 0, 0x10]), 1)
        with pytest.raises(LosslessJpegError):
            native.decode(frame)

    def test_a_frame_with_no_pixels_is_refused_not_returned_empty(self):
        """T.81 §B.2.2 gives X a range of 1..65535: a line of no samples is not
        a small image, it is a frame that does not describe one. Returning an
        empty array says the file held no pixels, which is a different claim
        from the one the caller can act on."""
        with pytest.raises(LosslessJpegError):
            native.decode(_frame(0, 0, 8, b"", _COUNTS, _SYMBOLS))

    def test_a_truncated_huffman_table_is_refused(self):
        counts = [0] * 15 + [17]               # promises seventeen symbols
        frame = _frame(2, 2, 8, b"\x00", counts, [0])
        with pytest.raises(LosslessJpegError):
            native.decode(frame)

    def test_a_symbol_above_sixteen_is_refused(self):
        frame = _frame(2, 2, 8, b"\x00", _COUNTS, [0, 17])
        with pytest.raises(LosslessJpegError):
            native.decode(frame)


# --------------------------------------------------------------------------- #
# random scans
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("shape", ["staircase", "narrow", "holes"])
@pytest.mark.parametrize("precision", [2, 8, 12, 16])
@pytest.mark.parametrize("predictor", [1, 2, 3, 4, 5, 6, 7])
def test_random_scans_decode_identically(shape, precision, predictor):
    counts, symbols = _table(shape, min(precision, 16))
    seed = precision * 64 + predictor * 8 + len(shape)
    for width, height, interval in ((7, 5, 0), (7, 5, 7), (7, 5, 3), (9, 1, 3)):
        for shift in ((0, 1) if precision > 1 else (0,)):
            scan = (_restarted(width * height, width * height, interval, seed)
                    if interval else _scan_for(width * height, seed))
            frame = _frame(width, height, precision, scan, counts, symbols,
                           predictor, interval, shift)
            # `holes` leaves half the code space unclaimed and random bits land
            # in it, so there is no image to agree on — only the refusal, which
            # both decoders must reach.
            agree = _refused_together if shape == "holes" else _identical
            assert agree(frame), (shape, precision, predictor, width, interval, shift)


@pytest.mark.parametrize("shape", ["staircase", "narrow", "holes"])
@pytest.mark.parametrize("precision", [8, 12, 16])
@pytest.mark.parametrize("predictor", [1, 4, 7])
def test_random_colour_scans_decode_identically(shape, precision, predictor):
    counts, symbols = _table(shape, min(precision, 16))
    other = _table("narrow" if shape != "narrow" else "staircase",
                   min(precision, 16))
    seed = precision * 8 + predictor
    for ncomp, interval in ((2, 0), (3, 0), (3, 4), (4, 2)):
        samples = 4 * 4 * ncomp
        # A restart interval counts MCUs, and with 1x1 sampling one MCU is one
        # pixel however many components it carries.
        scan = (_restarted(samples, 4 * 4, interval, seed + ncomp + 1)
                if interval else _scan_for(samples, seed + ncomp))
        components = [(c + 1, 1 if c == 1 else 0) for c in range(ncomp)]
        frame = _color_frame(4, 4, precision, scan,
                             {0: (counts, symbols), 1: other}, components,
                             predictor, interval)
        agree = _refused_together if shape == "holes" else _identical
        assert agree(frame), (shape, precision, predictor, ncomp, interval)


# --------------------------------------------------------------------------- #
# the front door falls back
# --------------------------------------------------------------------------- #

class TestTheFrontDoor:
    def test_it_uses_the_native_decoder_when_built(self):
        assert plumbline.engine() == "native"
        frame = _frame(4, 3, 8, _bits_to_bytes("0" * 12), _COUNTS, _SYMBOLS)
        assert np.array_equal(plumbline.decode(frame), decode_slowly(frame))

    def test_it_falls_back_to_turbo_then_the_oracle(self, monkeypatch):
        from plumbline import turbo
        frame = _frame(4, 3, 8, _bits_to_bytes("0" * 12), _COUNTS, _SYMBOLS)

        monkeypatch.setattr(native, "AVAILABLE", False)
        assert plumbline.engine() in ("turbo", "reference")
        assert np.array_equal(plumbline.decode(frame), decode_slowly(frame))

        monkeypatch.setattr(turbo, "AVAILABLE", False)
        assert plumbline.engine() == "reference"
        assert np.array_equal(plumbline.decode(frame), decode_slowly(frame))

    def test_without_native_a_colour_frame_skips_turbo_for_the_oracle(self, monkeypatch):
        """turbo refuses colour; the front door must not surface that."""
        monkeypatch.setattr(native, "AVAILABLE", False)
        frame = _color_frame(2, 2, 8, _bits_to_bytes("0" * 12),
                             {0: (_COUNTS, _SYMBOLS)}, [(1, 0), (2, 0), (3, 0)])
        assert plumbline.decode(frame).shape == (2, 2, 3)


# --------------------------------------------------------------------------- #
# real discs
# --------------------------------------------------------------------------- #

def _testdata() -> Path | None:
    root = Path(os.environ.get("PLUMBLINE_TESTDATA", r"$PLUMBLINE_TESTDATA"))
    return root if root.is_dir() else None


def _discs() -> list:
    """Discs whose first frame the oracle can afford to decode.

    Unlike the turbo suite, colour discs stay in: this decoder takes them.
    Only the metadata is read while collecting, so listing stays cheap even
    where the folder holds a hundred megabytes of mammogram.
    """
    root = _testdata()
    if root is None:
        return []
    try:
        import pydicom
    except ImportError:
        return []
    # The oracle decodes about 0.2 Msamples a second, so a full disc costs a
    # minute or more. Small ones run on every `pytest`; the rest are opt-in.
    budget = float("inf") if os.environ.get("PLUMBLINE_FULL_TESTDATA") else 3.0e5
    chosen = []
    for path in sorted(root.glob("*.dcm")):
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True)
            if str(dataset.file_meta.TransferSyntaxUID) not in LOSSLESS_SYNTAXES:
                continue
            samples = int(getattr(dataset, "SamplesPerPixel", 1) or 1)
            if int(dataset.Rows) * int(dataset.Columns) * samples > budget:
                continue
        except Exception:
            continue
        chosen.append(pytest.param(path, id=path.stem[:44]))
    return chosen


@pytest.mark.skipif(_testdata() is None, reason="no hospital discs to test against")
@pytest.mark.parametrize("path", _discs())
def test_real_discs_decode_identically(path):
    """Set PLUMBLINE_FULL_TESTDATA=1 to include the large discs — it takes minutes."""
    import pydicom
    from pydicom.encaps import generate_pixel_data_frame

    dataset = pydicom.dcmread(path)
    count = int(getattr(dataset, "NumberOfFrames", 1) or 1)
    frame = next(iter(generate_pixel_data_frame(dataset.PixelData, count)))
    assert _identical(frame)
