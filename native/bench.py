"""Benchmark the lossless JPEG decoders against each other on real discs.

Usage:  python native/bench.py [--repeats N] [--budget MPX]

Decodes the first frame of every plumbline disc in the test data folder
(`$PLUMBLINE_TESTDATA`, or $PLUMBLINE_TESTDATA) through every
decoder that is available on this machine, several times each, and reports the
best throughput in megapixels a second — best-of, not mean, because the
question is what the decoder can do, not what the machine was doing meanwhile.

The oracle only runs within a small pixel budget: at ~0.2 Mpx/s a mammogram
would take minutes and prove nothing new about speed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from plumbline import reference, native, turbo  # noqa: E402


def _frames() -> list[tuple[str, bytes, int]]:
    import pydicom
    from pydicom.encaps import generate_pixel_data_frame

    root = Path(os.environ.get("PLUMBLINE_TESTDATA",
                               r"$PLUMBLINE_TESTDATA"))
    plumbline = ("1.2.840.10008.1.2.4.70", "1.2.840.10008.1.2.4.57")
    out = []
    for path in sorted(root.glob("*.dcm")):
        try:
            dataset = pydicom.dcmread(path)
            if str(dataset.file_meta.TransferSyntaxUID) not in plumbline:
                continue
            count = int(getattr(dataset, "NumberOfFrames", 1) or 1)
            frame = next(iter(generate_pixel_data_frame(dataset.PixelData, count)))
            pixels = int(dataset.Rows) * int(dataset.Columns)
            out.append((path.stem, frame, pixels))
        except Exception as error:
            print(f"  (skipping {path.name}: {error})")
    return out


def _best(decode, frame: bytes, repeats: int) -> float | None:
    try:
        decode(frame)                       # warm the caches (and the JIT)
    except Exception:
        return None
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        decode(frame)
        best = min(best, time.perf_counter() - start)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--budget", type=float, default=0.5,
                        help="oracle budget in megapixels (default 0.5)")
    arguments = parser.parse_args()

    engines = [("native", native.decode, native.AVAILABLE),
               ("turbo", turbo.decode, turbo.AVAILABLE),
               ("oracle", reference.decode, True)]
    print("engines:", ", ".join(name for name, _, up in engines if up))

    if turbo.AVAILABLE:
        started = time.perf_counter()
        turbo.warmup()
        print(f"numba warmup: {time.perf_counter() - started:.1f}s")

    totals: dict[str, list[float]] = {}
    for stem, frame, pixels in _frames():
        mpx = pixels / 1e6
        line = f"{stem[:44]:<44} {mpx:7.2f} Mpx"
        for name, decode, available in engines:
            if not available:
                continue
            if name == "oracle" and mpx > arguments.budget:
                line += f"   {name}: (skipped)"
                continue
            seconds = _best(decode, frame, arguments.repeats)
            if seconds is None:
                line += f"   {name}: refused"
                continue
            line += f"   {name}: {mpx / seconds:8.1f} Mpx/s"
            totals.setdefault(name, []).append(mpx / seconds)
        print(line)

    print()
    for name, rates in totals.items():
        arr = np.array(rates)
        print(f"{name:>8}: median {np.median(arr):8.1f} Mpx/s   "
              f"min {arr.min():8.1f}   max {arr.max():8.1f}   "
              f"over {arr.size} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
