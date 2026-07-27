"""Measure Plumbline against every other lossless-JPEG decoder installed here.

Two questions, and the second matters more:

    1. How fast is each decoder?
    2. Does each one agree with the reference, bit for bit?

A decoder that is fast and wrong is worth nothing, so speed is reported only
alongside agreement. Where a decoder returns pixels that differ from
`plumbline.reference`, this prints how many samples differ and where the
first one is — because "it decoded without error" is not the same as "it
decoded".

Nothing here is uploaded and nothing is written; frames are read from the
folder named by $PLUMBLINE_TESTDATA and decoded in memory.

    PLUMBLINE_TESTDATA=/path/to/discs python native/compare.py

Decoders are used only if already importable. Missing ones are reported as
missing rather than silently skipped, so a run that compares against nothing
cannot be mistaken for a clean sweep.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

warnings.filterwarnings("ignore")

import numpy as np                                              # noqa: E402

from plumbline import reference                                 # noqa: E402
from plumbline import native, turbo                             # noqa: E402

LOSSLESS = ("1.2.840.10008.1.2.4.57", "1.2.840.10008.1.2.4.70")
REPEATS = 3


# --------------------------------------------------------------------------- #
# the field
# --------------------------------------------------------------------------- #

def _competitors() -> list[tuple[str, object, str]]:
    """(name, decode(frame) -> ndarray, note) for every decoder importable here."""
    field: list[tuple[str, object, str]] = []

    field.append(("plumbline (native)", native.decode if native.AVAILABLE else None,
                  "compiled C, this project"))
    field.append(("plumbline (turbo)", turbo.decode if turbo.AVAILABLE else None,
                  "numba, greyscale only"))

    try:
        from libjpeg import decode as pylibjpeg

        # takes the encoded bytes directly; `reshape=True` is the default
        field.append(("pylibjpeg-libjpeg", pylibjpeg, "GPL-3.0"))
    except Exception:
        field.append(("pylibjpeg-libjpeg", None, "not installed"))

    try:
        from imagecodecs import ljpeg_decode

        field.append(("imagecodecs.ljpeg", ljpeg_decode, "BSD-3"))
    except Exception:
        field.append(("imagecodecs.ljpeg", None, "not installed"))

    try:
        from imagecodecs import jpeg8_decode  # noqa: F401
        from imagecodecs import jpegsof3_decode

        field.append(("imagecodecs.jpegsof3", jpegsof3_decode, "BSD-3"))
    except Exception:
        field.append(("imagecodecs.jpegsof3", None, "not installed"))

    return field


def _frames() -> list[tuple[str, bytes]]:
    root = Path(os.environ.get("PLUMBLINE_TESTDATA", "testdata"))
    if not root.is_dir():
        sys.exit(f"no corpus at {root} — set $PLUMBLINE_TESTDATA")

    import pydicom
    from pydicom.encaps import generate_fragmented_frames

    out = []
    for path in sorted(root.glob("*.dcm")):
        dataset = pydicom.dcmread(path, force=True)
        try:
            syntax = str(dataset.file_meta.TransferSyntaxUID)
        except Exception:
            continue
        if syntax not in LOSSLESS:
            continue
        try:
            fragments = next(iter(generate_fragmented_frames(dataset.PixelData)))
            out.append((path.stem, b"".join(fragments)))
        except Exception:
            continue
    return out


# --------------------------------------------------------------------------- #

def _agreement(mine: np.ndarray, truth: np.ndarray) -> str:
    """How a decoder's output differs from the reference, in one phrase."""
    mine = np.squeeze(mine)
    truth = np.squeeze(truth)
    if mine.shape != truth.shape:
        return f"SHAPE {mine.shape} vs {truth.shape}"
    if np.array_equal(mine, truth):
        return "exact"
    differing = int(np.count_nonzero(mine != truth))
    first = np.argwhere(mine != truth)[0]
    return (f"WRONG {differing}/{truth.size} samples "
            f"({100 * differing / truth.size:.1f}%), first at {tuple(first)}")


def main() -> None:
    frames = _frames()
    if not frames:
        sys.exit("no lossless-JPEG frames found in the corpus")

    field = _competitors()
    print(f"{len(frames)} frames · {REPEATS} repeats · best of each\n")
    for name, fn, note in field:
        print(f"  {name:24s} {'available' if fn else 'MISSING':10s} {note}")
    print()

    rates: dict[str, list[float]] = {name: [] for name, fn, _ in field if fn}
    wrong: list[str] = []

    for label, frame in frames:
        truth = reference.decode(frame)
        pixels = truth.size
        print(f"{label}")
        for name, fn, _ in field:
            if fn is None:
                continue
            best = None
            verdict = ""
            for _ in range(REPEATS):
                started = time.perf_counter()
                try:
                    got = fn(frame)
                except Exception as error:
                    verdict = f"refused ({type(error).__name__})"
                    break
                elapsed = time.perf_counter() - started
                best = elapsed if best is None else min(best, elapsed)
                verdict = _agreement(got, truth)
            if best is None:
                print(f"    {name:24s} {'—':>12s}   {verdict}")
                continue
            rate = pixels / best / 1e6
            rates[name].append(rate)
            flag = "" if verdict == "exact" else "  <-- "
            print(f"    {name:24s} {rate:9.1f} Mpx/s   {flag}{verdict}")
            if verdict != "exact":
                wrong.append(f"{name} on {label}: {verdict}")
        print()

    print("median over frames each decoder accepted:")
    for name in rates:
        if rates[name]:
            print(f"  {name:24s} {statistics.median(rates[name]):9.1f} Mpx/s"
                  f"   ({len(rates[name])} frames)")

    print()
    if wrong:
        print(f"DISAGREEMENTS WITH THE REFERENCE: {len(wrong)}")
        for line in wrong:
            print(f"  {line}")
        print("\nA disagreement is not automatically the other decoder's fault.")
        print("It means one of the two is wrong, and that is worth finding out.")
    else:
        print("every decoder that accepted a frame agreed with the reference exactly.")


if __name__ == "__main__":
    main()
