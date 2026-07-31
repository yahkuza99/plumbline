"""Frames that used to decode into a picture that was not the picture.

Every case here was returned as a successful decode, by every engine, with no
exception and no warning — the one outcome this project exists to prevent. They
are grouped in one file because they are the same failure wearing three faces:
a header field and the entropy data disagreed, and nothing compared them.

Each test asserts a refusal rather than a particular set of pixels. What the
right pixels would have been is not knowable — that is the point. When a frame
says two contradictory things about itself, any image is a guess, and a guess
rendered on a diagnostic workstation is indistinguishable from a finding.
"""

import numpy as np
import pytest

from conformance import encode as enc
from conformance import frames as build
from plumbline import native, reference
from plumbline.reference import LosslessJpegError

WIDTH, HEIGHT, PRECISION = 8, 4, 8

ENGINES = [("reference", reference.decode)]
if native.AVAILABLE:
    ENGINES.append(("native", native.decode))
try:                                            # numba does not install everywhere
    from plumbline import turbo
    if turbo.AVAILABLE:
        ENGINES.append(("turbo", turbo.decode))
except Exception:                               # pragma: no cover
    pass

IDS = [name for name, _ in ENGINES]


def _restarting_frame(declared_interval):
    """A scan that restarts once per row, under whatever DRI we choose to claim."""
    image = np.random.default_rng(7).integers(
        0, 1 << PRECISION, (HEIGHT, WIDTH), dtype=np.int64)
    data, plan = enc.encode(image, PRECISION, 1, restart_interval=WIDTH)
    return build.greyscale_frame(WIDTH, HEIGHT, PRECISION, data, *plan[0],
                                 predictor=1,
                                 restart_interval=declared_interval), image


@pytest.mark.parametrize("engine, decode", ENGINES, ids=IDS)
def test_more_restart_markers_than_the_interval_needs(engine, decode):
    """DRI says one thing, the stream does another.

    A transcoder rewrites the restart interval and leaves the entropy data
    alone. The decoder then resets its predictor where the header says and
    re-syncs where the markers are, and after the first one those are different
    places. Measured at 24 of 32 pixels wrong, returned as a success.

    The check was `found < required`, so it only ever looked one way.
    """
    frame, _ = _restarting_frame(WIDTH * 2)     # stream restarts every row
    with pytest.raises(LosslessJpegError, match="disagree"):
        decode(frame)


@pytest.mark.parametrize("engine, decode", ENGINES, ids=IDS)
def test_restart_markers_with_no_dri_at_all(engine, decode):
    """The check was guarded by `if interval > 0`, so this was not checked.

    A header rewrite drops the 0xFFDD segment, or a frame is lifted out of a
    multi-frame object where only the first fragment carried it. The markers
    then look like data, and the padding that aligns each one to a byte
    boundary is decoded as entropy. Same 24 of 32 pixels wrong, same silence.
    """
    frame, _ = _restarting_frame(0)             # no DRI in the frame at all
    with pytest.raises(LosslessJpegError, match="restart marker"):
        decode(frame)


@pytest.mark.parametrize("engine, decode", ENGINES, ids=IDS)
def test_the_conforming_frame_still_decodes(engine, decode):
    """The two checks above must cost nothing to a frame that is fine."""
    frame, image = _restarting_frame(WIDTH)
    assert np.array_equal(np.squeeze(decode(frame)), image)


@pytest.mark.parametrize("engine, decode", ENGINES, ids=IDS)
@pytest.mark.parametrize("precision, shift", [(12, 4), (16, 8), (10, 2), (9, 1)])
def test_point_transform_output_fits_the_precision_declared(
        engine, decode, precision, shift):
    """A sample is P-Pt bits wide, and the mask said P.

    The scan holds the image after a right shift by Pt, so that is how wide its
    samples are — which the default prediction already accounted for and the
    modulus did not. For a conforming frame the two masks give the same number
    and nothing shows. Give the frame a wrong Al field, which is one nibble,
    and the wide mask passes samples that `out <<= shift` then multiplies past
    the precision the frame declares: a frame calling itself 12-bit came back
    holding 34,800, where 12 bits reach 4,095.

    Nothing downstream can tell that from data. A viewer windows it as if it
    were the image.
    """
    ceiling = 1 << precision
    image = np.array([[ceiling - 1, ceiling - 96, ceiling // 2, 0]],
                     dtype=np.int64)
    data, plan = enc.encode(image, precision, 1)
    frame = build.greyscale_frame(4, 1, precision, data, *plan[0],
                                  predictor=1, point_transform=shift)

    try:
        out = np.atleast_1d(np.squeeze(decode(frame)))
    except LosslessJpegError:
        return                                  # refusing is always allowed
    over = [int(v) for v in out if v >= ceiling]
    assert not over, (
        f"{engine} returned {over} from a frame declaring {precision}-bit "
        f"precision, where the largest expressible sample is {ceiling - 1}")


@pytest.mark.parametrize("engine, decode", ENGINES, ids=IDS)
@pytest.mark.parametrize("rows", [1, 2])
def test_an_interval_must_end_at_its_own_restart_marker(engine, decode, rows):
    """One damaged interval, every other row bit-exact, and a clean status.

    A scratched disc drops a byte inside the first interval of a slice. The
    decoder reads that interval wrong, then finds the marker exactly where it
    expected and resynchronises perfectly: the first row-group comes back as
    noise and the remaining rows are correct to the bit. An image that is right
    everywhere but one band is the hardest kind of wrong to notice, because a
    band of noise reads as anatomy.

    The signal was sitting there unread. The encoder pads to the byte boundary
    before the marker, so an interval's last sample has to leave the position
    inside the final byte; a byte too few or too many puts it somewhere else.

    Checked here on every engine — and on the parallel path too, which decodes
    each interval in its own lane and so has to carry its own copy of the test.
    Half the reason this took two attempts is that a check reaching three of
    the six places made `reference` refuse frames `native` accepted.
    """
    width, height, precision = 8, 6, 8
    interval = width * rows
    image = np.random.default_rng(11 + rows).integers(
        0, 1 << precision, (height, width), dtype=np.int64)
    data, plan = enc.encode(image, precision, 1, restart_interval=interval)
    frame = build.greyscale_frame(width, height, precision, data, *plan[0],
                                  predictor=1, restart_interval=interval)

    assert np.array_equal(np.squeeze(decode(frame)), image), (
        f"{engine} cannot decode the undamaged frame; the rest is meaningless")

    # One byte out of the first interval. Every marker stays where it was, so
    # the decoder still lands on all of them — that is the whole problem.
    at = reference.header(frame)["scan_offset"] + 3
    with pytest.raises(LosslessJpegError):
        decode(frame[:at] + frame[at + 1:])


def test_the_parallel_path_checks_intervals_too():
    """turbo decodes each interval in a separate lane, so it needs its own."""
    try:
        from plumbline import turbo
    except Exception:                                   # pragma: no cover
        pytest.skip("numba is not installed")
    if not turbo.AVAILABLE:                             # pragma: no cover
        pytest.skip("numba is not installed")

    width, height, precision = 8, 8, 8
    image = np.random.default_rng(4).integers(
        0, 1 << precision, (height, width), dtype=np.int64)
    data, plan = enc.encode(image, precision, 1, restart_interval=width)
    frame = build.greyscale_frame(width, height, precision, data, *plan[0],
                                  predictor=1, restart_interval=width)

    assert np.array_equal(np.squeeze(turbo.decode(frame, parallel=True)), image)
    at = reference.header(frame)["scan_offset"] + 3
    with pytest.raises(LosslessJpegError):
        turbo.decode(frame[:at] + frame[at + 1:], parallel=True)


def _with_an_ac_table(frame, counts, symbols):
    """The same frame plus a DHT whose class nibble says AC, destination 0."""
    import struct
    permuted = list(symbols)
    permuted[0], permuted[-1] = permuted[-1], permuted[0]
    body = bytes([0x10]) + bytes(counts) + bytes(permuted)
    segment = b"\xff\xc4" + struct.pack(">H", len(body) + 2) + body
    at = frame.index(b"\xff\xda")
    return frame[:at] + segment + frame[at:]


@pytest.mark.parametrize("engine, decode", ENGINES, ids=IDS)
def test_an_ac_table_does_not_overwrite_the_dc_table_the_scan_uses(engine, decode):
    """T.81 B.2.4.2 gives Tc and Th four slots each; we merged them.

    The DHT byte is the class in the high nibble — 0 for DC, 1 for AC — and the
    destination in the low one. Masking the class away meant a table written as
    0x10 landed on the slot the lossless scan actually reads, which is DC 0.
    A frame carrying both came back with every pixel wrong and nothing said.

    Keying on the whole byte leaves the AC table somewhere nothing looks for
    it. That is deliberately not a refusal: a lossless scan selects DC tables
    only, so a file that also carries an AC table it never uses is not
    malformed, and refusing it would cost a readable image for nothing.
    """
    image = np.random.default_rng(5).integers(
        0, 1 << PRECISION, (HEIGHT, WIDTH), dtype=np.int64)
    data, plan = enc.encode(image, PRECISION, 1)
    counts, symbols = plan[0]
    frame = build.greyscale_frame(WIDTH, HEIGHT, PRECISION, data,
                                  counts, symbols, predictor=1)

    decoded = np.squeeze(decode(_with_an_ac_table(frame, counts, symbols)))
    assert np.array_equal(decoded, image), (
        f"{engine} used the AC table for a DC scan: "
        f"{int((decoded != image).sum())} of {image.size} pixels differ")
