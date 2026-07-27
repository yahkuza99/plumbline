"""Plumbline — a lossless JPEG decoder that is right, or says so.

Lossless JPEG (ITU-T T.81 Annex H, SOF3) is the format most medical images
on hospital discs are stored in. It shares a name with the JPEG everyone
knows and almost nothing else: no DCT, no quantisation, no loss. Pixels are
predicted from their neighbours and the difference is Huffman-coded.

Three implementations of the same decoder live here, fastest first:

* `native`    — compiled C, loaded through ctypes. Needs the platform's
                `_plumbline` library to have been built (`python native/build.py`).
* `turbo`     — numba. Fast after a JIT pause, needs numba, greyscale only.
* `reference` — pure Python. Always present, readable, correct, slow. This is
                the oracle every other implementation is tested against, and
                it is meant to be read: the comments cite the clause of the
                specification each step implements.

A caller should not have to know which of them this machine can run, so this
module picks: the native library when it is built, else numba when it is
installed — falling to the reference for the colour frames numba cannot take
— else the reference.

The choice is by *availability and capability only*. A frame the chosen
decoder refuses (truncated, subsampled, malformed) is refused, never retried
down the chain: the reference would quietly invent the tail of a truncated
scan, and the house rule is to refuse rather than return an image that merely
looks decoded.

    >>> import plumbline
    >>> pixels = plumbline.decode(frame)     # (h, w) or (h, w, components)

The rule this project exists to keep:

    **Decode correctly, or raise. Never return plausible wrong pixels.**

Wrong pixels in a medical image are worse than no pixels, because nobody
looking at them can tell. That is not hypothetical — it is what sent this
project looking in the first place.
"""

from __future__ import annotations

import numpy as np

from plumbline import native, reference, turbo
from plumbline.reference import (  # noqa: F401  (re-exported as the public API)
    LosslessJpegError,
    header,
)

__all__ = ["decode", "engine", "header", "LosslessJpegError", "__version__"]

__version__ = "0.1.0"


def decode(frame: bytes) -> np.ndarray:
    """Decode one lossless-JPEG frame with the best decoder this machine has.

    Greyscale frames return (height, width); colour frames return
    (height, width, components). Raises LosslessJpegError for anything no
    decoder here handles.
    """
    if native.AVAILABLE:
        return native.decode(frame)
    if turbo.AVAILABLE:
        if int(header(frame).get("components", 0)) == 1:
            return turbo.decode(frame)
    return reference.decode(frame)


def engine() -> str:
    """Which decoder `decode` will use — for logs and about boxes."""
    if native.AVAILABLE:
        return "native"
    if turbo.AVAILABLE:
        return "turbo"
    return "reference"
