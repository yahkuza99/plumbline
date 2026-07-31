"""Two frames that always refused in the end, and refused the wrong way.

Neither ever returned wrong pixels, so neither broke the rule this project is
built on. What they broke is the other half of the same promise: a decoder
whose stated job is opening files of unknown origin has to fail *small* and
fail *predictably*. One asked the operating system for 32 GB before working out
there was nothing to decode; the other raised an exception the README tells
callers not to expect, on the code path every platform without a wheel runs.
"""

import time
import tracemalloc

import numpy as np
import pytest

from conformance import encode as enc
from conformance import frames as build
from plumbline import native, reference
from plumbline.reference import LosslessJpegError

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


@pytest.mark.parametrize("engine, decode", ENGINES, ids=IDS)
def test_a_tiny_file_cannot_ask_for_a_huge_allocation(engine, decode):
    """Fifty-one bytes claiming 65535x65535 asked for 32 GB before refusing.

    Every sample costs at least one bit — the shortest Huffman code is one bit,
    and SSSS=0 appends no mantissa — so a scan of N bytes cannot carry more
    than 8N samples, whatever is in it. That is a proof rather than a
    heuristic, and checking it before the output buffer is allocated turns a
    32-GB, 37-second refusal into an immediate one.
    """
    sof = bytes([8, 0xFF, 0xFF, 0xFF, 0xFF, 1, 0, 0x11, 0])   # 65535 x 65535
    frame = (build.marker(0xD8) + build.marker(0xC3, sof)
             + build.huffman_table([1] + [0] * 15, [0])
             + build.marker(0xDA, bytes([1, 0, 0x00, 1, 0, 0]))
             + b"\x00\x10" + build.marker(0xD9))
    assert len(frame) < 100, "the point of this test is that the file is tiny"

    # Decode something real first. `turbo` JIT-compiles its kernels on first
    # use, which costs seconds and has nothing to do with what is being timed;
    # without this the test passes or fails on whether some earlier test in the
    # session happened to warm numba up, which is not a property of the code.
    image = np.zeros((2, 2), dtype=np.int64)
    data, plan = enc.encode(image, 8, 1)
    decode(build.greyscale_frame(2, 2, 8, data, *plan[0], predictor=1))

    tracemalloc.start()
    started = time.monotonic()
    try:
        with pytest.raises(LosslessJpegError):
            decode(frame)
    finally:
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    elapsed = time.monotonic() - started

    assert peak < 64 << 20, (
        f"{engine} allocated {peak / 2 ** 30:.2f} GB to refuse {len(frame)} bytes")
    assert elapsed < 5, (
        f"{engine} took {elapsed:.1f}s to refuse {len(frame)} bytes")


@pytest.mark.parametrize("engine, decode", ENGINES, ids=IDS)
def test_every_truncation_raises_the_documented_exception(engine, decode):
    """Cut a valid frame at every offset; only LosslessJpegError may escape.

    The pure-Python decoder raised a bare IndexError at 120 of 196 offsets: its
    four bytes of peek padding assumed a scan roughly as long as the image it
    claims, and a badly truncated one walked off the buffer. A caller who wrote
    `except LosslessJpegError`, which is what the README tells them to write,
    lost their process instead — on the decoder every platform without a wheel
    falls back to.

    Sweeping every offset rather than sampling a few is deliberate. The
    boundary cases are wherever a Huffman code happens to straddle the cut, and
    which offsets those are is a property of the frame, not something to guess.
    """
    image = np.random.default_rng(3).integers(0, 1 << 8, (8, 16), dtype=np.int64)
    data, plan = enc.encode(image, 8, 1)
    frame = build.greyscale_frame(16, 8, 8, data, *plan[0], predictor=1)

    escaped = []
    for cut in range(1, len(frame)):
        try:
            decode(frame[:cut])
        except LosslessJpegError:
            pass
        except Exception as error:                          # noqa: BLE001
            escaped.append((cut, type(error).__name__))

    assert not escaped, (
        f"{engine} let {len(escaped)} of {len(frame) - 1} truncations raise "
        f"something else, e.g. {escaped[:3]}")
