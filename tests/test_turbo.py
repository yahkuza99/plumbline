r"""Tests for the accelerated lossless JPEG decoder.

The oracle is `plumbline.reference`, deliberately — not pylibjpeg, not any other
library. The accelerated decoder exists only to produce what the slow one
produces, faster, so every test here decodes twice and diffs.

The frames come from three places:

* random scans across a range of Huffman tables, precisions, predictors and
  restart intervals. Random bits reach paths real files do not: SSSS values no
  encoder emits, codes too long to fuse with their mantissa, tables with holes.
* real hospital discs, when `$PLUMBLINE_TESTDATA` (or
  `$PLUMBLINE_TESTDATA`) is present. Only frames small enough for the slow oracle
  run by default; `PLUMBLINE_FULL_TESTDATA=1` runs every disc, which takes minutes.
* frames built by hand for the three details that decide correctness — the
  restart marker, SSSS = 16, and reconstruction wrapping at 2^P.

The one place the two decoders deliberately part company is a truncated scan.
The oracle reads off the end into its own padding and returns an image whose
tail it invented; the accelerated decoder notices and refuses. Refusing is
always allowed here. Returning pixels the oracle could not produce is not.

The whole module is skipped where numba is not installed.
"""

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("numba", reason="the accelerated decoder needs numba")

from plumbline import turbo                                  # noqa: E402
from plumbline.reference import LosslessJpegError, header              # noqa: E402
from plumbline.reference import decode as decode_slowly                # noqa: E402

MONOCHROME_LOSSLESS = ("1.2.840.10008.1.2.4.70", "1.2.840.10008.1.2.4.57")


# --------------------------------------------------------------------------- #
# frames built by hand
# --------------------------------------------------------------------------- #

def _marker(code: int, payload: bytes = b"") -> bytes:
    if not payload:
        return bytes([0xFF, code])
    length = len(payload) + 2
    return bytes([0xFF, code, length >> 8, length & 0xFF]) + payload


def _frame(width: int, height: int, precision: int, scan: bytes,
           counts: list[int], symbols: list[int], predictor: int = 1,
           restart_interval: int = 0, point_transform: int = 0) -> bytes:
    out = _marker(0xD8)
    out += _marker(0xC3, bytes([precision, height >> 8, height & 0xFF,
                                width >> 8, width & 0xFF, 1, 0, 0x11, 0]))
    out += _marker(0xC4, bytes([0]) + bytes(counts) + bytes(symbols))
    if restart_interval:
        out += _marker(0xDD, bytes([restart_interval >> 8, restart_interval & 0xFF]))
    out += _marker(0xDA, bytes([1, 0, 0x00, predictor, 0, point_transform]))
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


def _scan_for(pixels: int, seed: int) -> bytes:
    """Random entropy data long enough that no sample can run off the end.

    A sample costs at most 32 bits: sixteen for the longest code and sixteen for
    the widest mantissa. Four bytes each, and some slack, keeps every frame here
    complete, so the two decoders must agree on pixels rather than on a refusal.
    """
    generator = np.random.default_rng(seed)
    return _stuffed(generator.integers(0, 256, pixels * 4 + 16, dtype=np.uint8).tobytes())


def _decoded_twice(frame: bytes) -> tuple:
    """Both decoders' answers: an array each, or the exception type raised."""
    try:
        slow = decode_slowly(frame)
    except Exception as error:                 # the oracle raises bare IndexError
        slow = type(error)
    try:
        fast = turbo.decode(frame)
    except Exception as error:
        fast = type(error)
    return slow, fast


def _identical(frame: bytes) -> bool:
    slow, fast = _decoded_twice(frame)
    assert isinstance(slow, np.ndarray), f"the oracle refused this frame: {slow}"
    assert isinstance(fast, np.ndarray), f"the fast decoder refused this frame: {fast}"
    return (slow.dtype == fast.dtype and slow.shape == fast.shape
            and np.array_equal(slow, fast))


# --------------------------------------------------------------------------- #
# numba is optional
# --------------------------------------------------------------------------- #

class TestNumbaIsOptional:
    def test_it_reports_itself_available_where_numba_is_installed(self):
        assert turbo.AVAILABLE is True

    def test_it_imports_and_refuses_when_numba_is_missing(self, monkeypatch):
        """Without numba the application must lose speed, never a file."""
        import builtins
        import importlib
        import sys

        real_import = builtins.__import__

        def without_numba(name, *args, **kwargs):
            if name == "numba" or name.startswith("numba."):
                raise ImportError("numba is not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.delitem(sys.modules, "plumbline.turbo")
        monkeypatch.setattr(builtins, "__import__", without_numba)
        crippled = importlib.import_module("plumbline.turbo")

        assert crippled.AVAILABLE is False
        assert crippled is not turbo
        with pytest.raises(LosslessJpegError):
            crippled.decode(_frame(2, 2, 8, b"\x00", [2] + [0] * 15, [0, 16]))
        crippled.warmup()                      # a no-op, not a crash


# --------------------------------------------------------------------------- #
# the details that decide correctness
# --------------------------------------------------------------------------- #

# symbol 0 and symbol 16 are the two one-bit codes "0" and "1"
_COUNTS = [2] + [0] * 15
_SYMBOLS = [0, 16]


def _bits_to_bytes(bits: str) -> bytes:
    bits += "0" * (-len(bits) % 8)
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


class TestAgreesOnTheHardCases:
    def test_all_zero_differences_give_a_flat_image(self):
        frame = _frame(4, 3, 8, _bits_to_bytes("0" * 12), _COUNTS, _SYMBOLS)
        assert np.all(turbo.decode(frame) == 1 << 7)
        assert _identical(frame)

    def test_ssss_16_consumes_no_mantissa_bits(self):
        frame = _frame(3, 2, 16, _bits_to_bytes("1" + "0" * 5), _COUNTS, _SYMBOLS)
        assert turbo.decode(frame)[0, 0] == 0      # (32768 + 32768) mod 2^16
        assert _identical(frame)

    def test_restart_marker_resets_the_prediction(self):
        scan = _bits_to_bytes("0" * 4) + bytes([0xFF, 0xD0]) + _bits_to_bytes("0" * 4)
        frame = _frame(4, 2, 8, scan, _COUNTS, _SYMBOLS, restart_interval=4)
        assert np.all(turbo.decode(frame) == 1 << 7)
        assert _identical(frame)

    def test_stuffed_ff_is_data_not_a_marker(self):
        frame = _frame(2, 1, 8, bytes([0xFF, 0x00]), _COUNTS, _SYMBOLS)
        assert _identical(frame)

    def test_a_scan_ending_on_a_lone_ff_keeps_that_byte(self):
        """The last byte has no partner to skip, and is data like any other."""
        frame = _frame(2, 1, 8, bytes([0x00, 0xFF]), _COUNTS, _SYMBOLS)
        payload, restarts = turbo._destuff(np.array([0x00, 0xFF], np.uint8))
        assert list(payload) == [0x00, 0xFF] and restarts.size == 0
        assert _identical(frame)

    def test_the_output_dtype_follows_the_precision(self):
        counts, symbols = _table("staircase", 8)
        assert turbo.decode(
            _frame(4, 4, 8, _scan_for(16, 1), counts, symbols)).dtype == np.uint8
        counts, symbols = _table("staircase", 12)
        assert turbo.decode(
            _frame(4, 4, 12, _scan_for(16, 2), counts, symbols)).dtype == np.uint16

    def test_point_transform_shifts_after_prediction_not_before(self):
        counts, symbols = _table("staircase", 12)
        frame = _frame(5, 4, 12, _scan_for(20, 3), counts, symbols, point_transform=2)
        assert _identical(frame)
        assert turbo.decode(frame).max() % 4 == 0    # every sample was shifted

    def test_reconstruction_wraps_inside_the_loop_not_after_it(self):
        """Wrapping the finished image is a different picture, not just a cast.

        Predictor 7 averages two neighbours, so a sample that overflowed feeds a
        wrong prediction into the next one. The two runs below differ only in
        where 2^P is applied, and they disagree — which is why the accelerated
        decoder pays for the mask inside the loop, where it costs nothing.
        """
        counts, symbols = _table("narrow", 12)
        frame = _frame(6, 6, 12, _scan_for(36, 4), counts, symbols, predictor=7)
        info = header(frame)
        tables = turbo._fuse(counts, symbols)
        data, restarts = turbo._destuff(
            np.frombuffer(frame, dtype=np.uint8, offset=info["scan_offset"]))

        def scan_with(mask):
            out = np.empty(36, dtype=np.uint16)
            turbo._scan(out, 6, 6, data, restarts, 0, 7, *tables,
                              np.int64(1 << 11), np.int64(mask), np.zeros(1, np.int64))
            return out

        wrapped = scan_with((1 << 12) - 1)
        afterwards = scan_with((1 << 16) - 1) & ((1 << 12) - 1)
        assert np.array_equal(wrapped.reshape(6, 6), decode_slowly(frame))
        assert not np.array_equal(wrapped, afterwards)


class TestRefusesRatherThanGuesses:
    def test_colour_frames_are_refused_not_mangled(self):
        frame = _frame(2, 2, 8, _bits_to_bytes("0" * 4), _COUNTS, _SYMBOLS)
        frame = frame.replace(bytes([8, 0, 2, 0, 2, 1]), bytes([8, 0, 2, 0, 2, 3]), 1)
        with pytest.raises(LosslessJpegError):
            turbo.decode(frame)

    def test_garbage_is_refused(self):
        with pytest.raises(LosslessJpegError):
            turbo.decode(b"not a jpeg at all")

    def test_unknown_predictor_is_refused(self):
        frame = _frame(3, 3, 8, _bits_to_bytes("0" * 9), _COUNTS, _SYMBOLS,
                       predictor=9)
        with pytest.raises(LosslessJpegError):
            turbo.decode(frame)

    def test_a_truncated_scan_is_refused_not_padded_with_zeros(self):
        """Half a frame is half an image, however plausible the rest looks."""
        counts, symbols = _table("staircase", 12)
        frame = _frame(16, 16, 12, b"\x91\x37", counts, symbols)
        with pytest.raises(LosslessJpegError):
            turbo.decode(frame)

    def test_a_scan_naming_a_table_the_frame_lacks_is_refused(self):
        frame = _frame(2, 2, 8, b"\x00", _COUNTS, _SYMBOLS)
        frame = frame.replace(bytes([0xFF, 0xDA, 0, 8, 1, 0, 0x00]),
                              bytes([0xFF, 0xDA, 0, 8, 1, 0, 0x10]), 1)
        with pytest.raises(LosslessJpegError):
            turbo.decode(frame)

    def test_a_frame_with_no_pixels_is_not_a_crash(self):
        assert turbo.decode(_frame(0, 0, 8, b"", _COUNTS, _SYMBOLS)).shape == (0, 0)


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
            scan = _scan_for(width * height, seed)
            if interval:
                # Real restart markers, each piece long enough to finish the
                # image on its own: the markers may run out before the intervals
                # do, and then both decoders must read straight on.
                pieces = [_scan_for(width * height, seed + piece) for piece in range(9)]
                scan = bytes([0xFF, 0xD0]).join(pieces)
            frame = _frame(width, height, precision, scan, counts, symbols,
                           predictor, interval, shift)
            assert _identical(frame), (shape, precision, predictor, width, interval, shift)


def test_the_tables_under_test_reach_both_decoding_paths():
    """Guard the guard: the fused lookup and its fallback must both be exercised."""
    consumed, _, length, ssss = turbo._fuse(*_table("staircase", 16))
    assert np.any((consumed == 0) & (length > 0))     # too long to fuse
    assert np.any(consumed > 0)                       # fused
    assert np.any(ssss == 16)                         # and the SSSS = 16 case

    consumed, _, length, _ = turbo._fuse(*_table("holes", 16))
    assert np.any(length == 0)                        # windows matching no code


# --------------------------------------------------------------------------- #
# the parallel path
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("width,height", [(7, 5), (16, 4), (3, 11)])
def test_decoding_restart_intervals_concurrently_changes_nothing(width, height):
    counts, symbols = _table("staircase", 12)
    for attempt in range(4):
        pieces = [_scan_for(width, width * height + attempt * 8 + row)
                  for row in range(height + 1)]
        frame = _frame(width, height, 12, bytes([0xFF, 0xD0]).join(pieces),
                       counts, symbols, 1, width, 0)
        concurrent = turbo.decode(frame, parallel=True)
        assert np.array_equal(concurrent, turbo.decode(frame))
        assert np.array_equal(concurrent, decode_slowly(frame))


def test_intervals_that_do_not_start_on_a_row_stay_sequential():
    """Splitting mid-row would need the row above from another thread."""
    markers = np.arange(8, dtype=np.int64)
    assert turbo._plan(3, 7, 21, markers) is None       # 3 is not a row
    assert turbo._plan(7, 7, 21, markers) is not None
    # too few markers to know where each interval begins in the byte stream
    assert turbo._plan(7, 7, 21, markers[:1]) is None


# --------------------------------------------------------------------------- #
# real discs
# --------------------------------------------------------------------------- #

def _testdata() -> Path | None:
    root = Path(os.environ.get("PLUMBLINE_TESTDATA", r"$PLUMBLINE_TESTDATA"))
    return root if root.is_dir() else None


def _discs() -> list:
    """Discs whose first frame this decoder handles, and the oracle can afford.

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
    # The oracle decodes about 0.2 Mpx a second, so a full disc costs a minute
    # or more. One small one runs on every `pytest`; the rest are opt-in.
    budget = float("inf") if os.environ.get("PLUMBLINE_FULL_TESTDATA") else 3.0e5
    chosen = []
    for path in sorted(root.glob("*.dcm")):
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True)
            if str(dataset.file_meta.TransferSyntaxUID) not in MONOCHROME_LOSSLESS:
                continue
            if int(getattr(dataset, "SamplesPerPixel", 1)) != 1:
                continue
            if int(dataset.Rows) * int(dataset.Columns) > budget:
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
