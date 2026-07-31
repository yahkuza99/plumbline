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
