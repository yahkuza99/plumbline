"""The compiled core must refuse on its own, not because Python checked first.

`native.decode` validates a frame in Python and only then hands the scan to C.
That is the shipped path and it is sound. But the C says of itself, above
`build_table`, that it is safe for anyone who calls it without that Python in
front — and that was true of the Huffman tables and false of the scan.

Given a frame whose declared restart interval needs more markers than the
stream contains, the scan loops used to reset the predictor at every boundary
while carrying on from wherever the bitstream had reached, and return success
over pixels that are not the image. An audit found it; this reproduces it
through the ABI, the way a caller in another language would meet it.

The point is not that the shipped path was ever at risk. It is that a promise
written in a comment has to be one the code keeps, or whoever builds on it
inherits a guarantee that was never there.
"""

import numpy as np
import pytest

from conformance import encode as enc
from conformance import frames as build
from plumbline import native, reference

if not native.AVAILABLE:
    pytest.skip("the compiled core is not built (python native/build.py)",
                allow_module_level=True)

WIDTH, HEIGHT, PRECISION = 8, 4, 8
PLUMBLINE_OK = 0


def _frame_with_restarts():
    """A conforming frame that restarts once per row, and the image in it."""
    generator = np.random.default_rng(11)
    image = generator.integers(0, 1 << PRECISION, (HEIGHT, WIDTH), dtype=np.int64)
    data, plan = enc.encode(image, PRECISION, 1, restart_interval=WIDTH)
    frame = build.greyscale_frame(WIDTH, HEIGHT, PRECISION, data, *plan[0],
                                  predictor=1, restart_interval=WIDTH)
    return frame, image


def _through_the_abi(frame):
    """Call the C entry point directly, skipping every Python check.

    Mirrors `native.decode` exactly except that `check_scan` and `check_frame`
    are not called — which is what a binding from another language would do.
    """
    info = reference.header(frame)
    counts, symbols, symbol_counts, slot_tables = native._tables(info)
    slot_comps = np.array(native._slots(info), dtype=np.int32)
    scan = bytes(frame[info["scan_offset"]:])
    out = np.empty(info["height"] * info["width"], dtype=np.uint8)

    status = native._lib.plumbline_decode(
        scan, len(scan),
        info["width"], info["height"], info["components"],
        info["precision"], info["predictor"], info["point_transform"],
        info["restart_interval"],
        len(symbol_counts),
        counts.ctypes.data, symbols.ctypes.data, symbol_counts.ctypes.data,
        slot_tables.ctypes.data, slot_comps.ctypes.data,
        out.ctypes.data, 0)
    return status, out.reshape(info["height"], info["width"])


def _without_one_restart_marker(frame):
    """The same frame with a single RSTn spliced out, and nothing else changed.

    Removing the marker *and everything after it* would only prove the decoder
    notices it ran out of data — which it always did. The interesting frame is
    the one that still has all its entropy data and one marker fewer, because
    that is what a decoder can carry on reading and get wrong.
    """
    last = max(frame.rfind(bytes([0xFF, code])) for code in range(0xD0, 0xD8))
    assert last > 0, "the fixture carries no restart marker to remove"
    return frame[:last] + frame[last + 2:]


def test_it_refuses_a_scan_missing_a_restart_marker():
    """Splice one RSTn out; the core must say so rather than return pixels."""
    frame, image = _frame_with_restarts()
    status, pixels = _through_the_abi(_without_one_restart_marker(frame))
    assert status != PLUMBLINE_OK, (
        "the core reported success on a scan with too few restart markers; "
        f"the last row came out as {pixels[-1].tolist()} instead of "
        f"{image[-1].tolist()}")


def test_the_intact_frame_still_decodes():
    """The check must not cost the frames that are fine."""
    frame, image = _frame_with_restarts()
    assert np.array_equal(np.squeeze(native.decode(frame)), image)
    assert np.array_equal(np.squeeze(reference.decode(frame)), image)


def test_python_refuses_it_too():
    """The shipped path was never exposed, and must stay that way."""
    frame, _ = _frame_with_restarts()
    damaged = _without_one_restart_marker(frame)

    for decode in (native.decode, reference.decode):
        with pytest.raises(reference.LosslessJpegError):
            decode(damaged)
