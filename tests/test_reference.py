"""Tests for our own lossless JPEG decoder.

The cases below encode the details that decide whether an implementation of
this format is correct: restart markers, which reset the prediction *and* put
the whole first line of the interval back on Ra (T.81 §H.1.2.1); the SSSS = 16
special case; and where the reconstruction wraps.

The first three were found by decoding real hospital discs and diffing against
two independent decoders. The Ra rule could not have been — every one of the
81,172 real frames this project has decoded uses predictor 1, under which the
correct and incorrect readings compute the same image. It was found by reading
Annex H sentence by sentence, and it is why the tests here quote the clause
rather than describing it.
"""

import numpy as np
import pytest

from plumbline.reference import LosslessJpegError, decode, header


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


# A table where symbol 0 is the single-bit code "0" and symbol 16 is "10".
_COUNTS = [2] + [0] * 15
_SYMBOLS = [0, 16]


def _color_frame(width: int, height: int, precision: int, scan: bytes,
                 tables: dict[int, tuple[list[int], list[int]]],
                 components: list[tuple[int, int]],  # (component id, table id)
                 predictor: int = 1, restart_interval: int = 0,
                 sampling: int = 0x11) -> bytes:
    """An interleaved multi-component frame, one SOS covering every component."""
    out = _marker(0xD8)
    sof = bytes([precision, height >> 8, height & 0xFF,
                 width >> 8, width & 0xFF, len(components)])
    for component_id, _ in components:
        sof += bytes([component_id, sampling, 0])
    out += _marker(0xC3, sof)
    for identifier, (counts, symbols) in tables.items():
        out += _huffman_table(counts, symbols, identifier)
    if restart_interval:
        out += _marker(0xDD, bytes([restart_interval >> 8, restart_interval & 0xFF]))
    sos = bytes([len(components)])
    for component_id, table_id in components:
        sos += bytes([component_id, table_id << 4])
    sos += bytes([predictor, 0, 0])
    out += _marker(0xDA, sos)
    return out + scan + _marker(0xD9)


def _bits_to_bytes(bits: str) -> bytes:
    bits += "0" * (-len(bits) % 8)
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


class TestHeader:
    def test_reads_geometry_and_scan_parameters(self):
        frame = _frame(4, 2, 12, _bits_to_bytes("0" * 8), _COUNTS, _SYMBOLS)
        info = header(frame)
        assert (info["width"], info["height"]) == (4, 2)
        assert info["precision"] == 12
        assert info["predictor"] == 1
        assert info["components"] == 1

    def test_restart_interval_is_read(self):
        frame = _frame(4, 2, 8, _bits_to_bytes("0" * 8), _COUNTS, _SYMBOLS,
                       restart_interval=4)
        assert header(frame)["restart_interval"] == 4

    def test_a_frame_without_a_scan_is_rejected(self):
        with pytest.raises(LosslessJpegError):
            header(_marker(0xD8) + _marker(0xD9))


class TestDecoding:
    def test_all_zero_differences_give_a_flat_image(self):
        # every symbol is 0, so every sample equals its prediction
        frame = _frame(4, 3, 8, _bits_to_bytes("0" * 16), _COUNTS, _SYMBOLS)
        out = decode(frame)
        assert out.shape == (3, 4)
        assert np.all(out == 1 << 7)      # the default prediction, 2^(P-1)

    def test_ssss_16_consumes_no_mantissa_bits(self):
        # symbol 16 then zeros: the first sample is default + 32768 mod 2^16,
        # and every later sample repeats it. Reading sixteen bits after the
        # symbol instead would desynchronise the stream and change the rest.
        frame = _frame(3, 2, 16, _bits_to_bytes("10" + "0" * 20), _COUNTS, _SYMBOLS)
        out = decode(frame)
        assert out[0, 0] == 0             # (32768 + 32768) mod 65536
        assert np.all(out == 0)

    def test_reconstruction_wraps_at_two_to_the_precision(self):
        frame = _frame(2, 1, 16, _bits_to_bytes("10" + "0" * 8), _COUNTS, _SYMBOLS)
        out = decode(frame)
        assert out.dtype == np.uint16
        assert int(out[0, 0]) < (1 << 16)

    def test_restart_marker_resets_the_prediction(self):
        # two rows, a restart between them; both rows must start from the
        # default rather than continuing from the row above
        scan = _bits_to_bytes("0" * 8) + bytes([0xFF, 0xD0]) + _bits_to_bytes("0" * 8)
        frame = _frame(4, 2, 8, scan, _COUNTS, _SYMBOLS, restart_interval=4)
        out = decode(frame)
        assert np.all(out == 1 << 7)

    def test_the_whole_first_line_of_an_interval_predicts_from_ra(self):
        """T.81 §H.1.2.1: "The one-dimensional horizontal predictor (prediction
        sample Ra) is used for the first line of samples at the start of the
        scan and at the beginning of each restart interval. The selected
        predictor is used for all other lines."

        So a restart does not merely reset the prediction *value* for the one
        sample that follows the marker; it puts that whole line back on Ra.
        Under predictor 1 the two readings are the same number, which is why
        this went unnoticed: every one of the 81,172 real frames this project
        has decoded uses predictor 1.

        Four rows of four, predictor 2 (Rb), one restart after two rows. Each
        bit is a sample: "0" is a difference of 0 and "1" is SSSS = 16, a
        difference of 32768 carrying no mantissa.
        """
        scan = (_bits_to_bytes("1000" "1000")
                + bytes([0xFF, 0xD0])
                + _bits_to_bytes("0100" "0000"))
        frame = _frame(4, 4, 16, scan, _COUNTS, _SYMBOLS,
                       predictor=2, restart_interval=8)
        out = decode(frame)

        # Row 2 opens the second interval: column 0 takes the default, and
        # column 1 must predict from Ra — the 32768 just written beside it —
        # so its 32768 difference wraps to 0. Predicting from Rb instead, which
        # is what this decoder used to do, reads row 1 column 1 (a 0) and
        # leaves 32768 there.
        assert out[2, 0] == 32768
        assert out[2, 1] == 0, "the first line of a restart interval must use Ra"

        # Row 3 is the *second* line of that interval, so it is back on the
        # selected predictor: column 1 takes Rb from row 2 (a 0) and stays 0. A
        # decoder that reset every row rather than every interval would put Ra
        # here too and leave 32768.
        assert out[3, 1] == 0, "only the first line of an interval uses Ra"
        assert out.tolist() == [[0, 0, 0, 0]] + [[32768, 0, 0, 0]] * 3

    def test_an_interval_that_is_not_whole_rows_is_refused(self):
        """T.81 §H.1.1 and table B.7 require Ri to be an integer multiple of
        the MCU in an MCU-row. Mid-row there is no "first line of the interval"
        for §H.1.2.1 to name, so every reading of the reset is a guess — and
        guessing is the one thing this decoder does not do."""
        scan = _bits_to_bytes("0" * 8) + bytes([0xFF, 0xD0]) + _bits_to_bytes("0" * 8)
        with pytest.raises(LosslessJpegError, match="whole number"):
            decode(_frame(4, 4, 8, scan, _COUNTS, _SYMBOLS, restart_interval=3))
        # the same frame with an interval of whole rows is fine
        decode(_frame(4, 2, 8, scan, _COUNTS, _SYMBOLS, restart_interval=4))

    def test_stuffed_ff_is_data_not_a_marker(self):
        scan = bytes([0xFF, 0x00]) + _bits_to_bytes("0" * 8)
        frame = _frame(2, 1, 8, scan, _COUNTS, _SYMBOLS)
        decode(frame)                      # must not raise


class TestColour:
    """The interleaved three-component shape real RGB ultrasound discs take."""

    # "0" → difference 0; "1" + one mantissa bit → difference +1 or -1
    _SMALL = ([2] + [0] * 15, [0, 1])
    # its mirror image: "0" → symbol 16, "1" → symbol 0 — decoding with the
    # wrong table produces a visibly different sample
    _MIRROR = ([2] + [0] * 15, [16, 0])

    def test_colour_frame_decodes_to_height_width_components(self):
        # 2x2, three components, every difference zero
        scan = _bits_to_bytes("0" * 12)
        frame = _color_frame(2, 2, 8, scan, {0: self._SMALL},
                             [(1, 0), (2, 0), (3, 0)])
        out = decode(frame)
        assert out.shape == (2, 2, 3)
        assert out.dtype == np.uint8
        assert np.all(out == 1 << 7)

    def test_each_component_predicts_from_its_own_samples(self):
        # pixel 0: red +1, green 0, blue 0 — pixel 1: all zero differences.
        # Red's second sample must follow red's first (129), and must not
        # leak into green or blue, whose neighbours are their own.
        # "11" = the +1 difference (code "1", one mantissa bit "1")
        scan = _bits_to_bytes("11" + "0" + "0" + "0" + "0" + "0")
        frame = _color_frame(2, 1, 8, scan, {0: self._SMALL},
                             [(1, 0), (2, 0), (3, 0)])
        out = decode(frame)
        assert out[0, 0].tolist() == [129, 128, 128]
        assert out[0, 1].tolist() == [129, 128, 128]

    def test_each_component_uses_its_own_huffman_table(self):
        # component 2 uses the mirrored table, where "1" is the zero
        # difference. Decoding its bit with component 1's table instead
        # would read symbol 16 and shift the sample by 32768.
        scan = _bits_to_bytes("0" + "1" + "0")
        frame = _color_frame(1, 1, 16, scan,
                             {0: self._SMALL, 1: self._MIRROR},
                             [(1, 0), (2, 1), (3, 0)])
        out = decode(frame)
        assert out[0, 0].tolist() == [1 << 15, 1 << 15, 1 << 15]

    def test_restart_interval_counts_pixels_not_samples(self):
        # 2x2 with DRI=2: one restart between the rows. Both rows must
        # start every component from the default prediction.
        row = _bits_to_bytes("0" * 6)
        scan = row + bytes([0xFF, 0xD0]) + row
        frame = _color_frame(2, 2, 8, scan, {0: self._SMALL},
                             [(1, 0), (2, 0), (3, 0)], restart_interval=2)
        out = decode(frame)
        assert np.all(out == 1 << 7)

    def test_subsampled_frames_are_refused_not_guessed(self):
        scan = _bits_to_bytes("0" * 12)
        frame = _color_frame(2, 2, 8, scan, {0: self._SMALL},
                             [(1, 0), (2, 0), (3, 0)], sampling=0x21)
        with pytest.raises(LosslessJpegError):
            decode(frame)

    def test_non_interleaved_scans_are_refused_not_guessed(self):
        # three components in the frame, but a scan covering only one
        counts, symbols = self._SMALL
        frame = _marker(0xD8)
        frame += _marker(0xC3, bytes([8, 0, 2, 0, 2, 3,
                                      1, 0x11, 0, 2, 0x11, 0, 3, 0x11, 0]))
        frame += _huffman_table(counts, symbols)
        frame += _marker(0xDA, bytes([1, 1, 0x00, 1, 0, 0]))
        frame += _bits_to_bytes("0" * 12) + _marker(0xD9)
        with pytest.raises(LosslessJpegError):
            decode(frame)


class TestRefusesRatherThanGuesses:
    def test_truncated_colour_headers_are_refused_not_mangled(self):
        # claims three components but carries parameters for only one
        frame = _frame(2, 2, 8, _bits_to_bytes("0" * 8), _COUNTS, _SYMBOLS)
        frame = frame.replace(bytes([8, 0, 2, 0, 2, 1]), bytes([8, 0, 2, 0, 2, 3]), 1)
        with pytest.raises(LosslessJpegError):
            decode(frame)

    def test_scan_component_unknown_to_the_frame_is_refused(self):
        scan = _bits_to_bytes("0" * 12)
        frame = _color_frame(2, 2, 8, scan, {0: TestColour._SMALL},
                             [(1, 0), (2, 0), (3, 0)])
        sos = bytes([3, 1, 0x00, 2, 0x00, 3, 0x00])
        frame = frame.replace(sos, bytes([3, 1, 0x00, 2, 0x00, 9, 0x00]), 1)
        with pytest.raises(LosslessJpegError):
            decode(frame)

    def test_garbage_is_refused(self):
        with pytest.raises(LosslessJpegError):
            decode(b"not a jpeg at all")

    def test_unknown_predictor_is_refused(self):
        frame = _frame(3, 3, 8, _bits_to_bytes("0" * 24), _COUNTS, _SYMBOLS,
                       predictor=9)
        with pytest.raises(LosslessJpegError):
            decode(frame)
