"""Every frame of every plumbline disc, decoded twice and diffed.

Usage:  python native/verify.py

Greyscale frames are checked against `turbo`, which is itself proven
bit-exact against the pure-Python oracle over these same discs; colour frames
go straight to the oracle, since turbo refuses them. The pytest suite already
does this for first frames and (under PLUMBLINE_FULL_TESTDATA=1) full discs — this
script is the belt to those braces: every frame of every disc, one line each.

Exits non-zero on the first mismatch. A mismatch is a released-blocking bug,
never something to average away.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
import pydicom  # noqa: E402
from pydicom.encaps import generate_pixel_data_frame  # noqa: E402

from plumbline import reference, native, turbo  # noqa: E402

LOSSLESS = ("1.2.840.10008.1.2.4.70", "1.2.840.10008.1.2.4.57")


def main() -> int:
    if not native.AVAILABLE:
        print("the native library is not built (python native/build.py)")
        return 2
    if turbo.AVAILABLE:
        turbo.warmup()

    root = Path(os.environ.get("PLUMBLINE_TESTDATA",
                               r"$PLUMBLINE_TESTDATA"))
    frames = 0
    files = 0
    for path in sorted(root.glob("*.dcm")):
        try:
            dataset = pydicom.dcmread(path)
            if str(dataset.file_meta.TransferSyntaxUID) not in LOSSLESS:
                continue
        except Exception:
            continue
        count = int(getattr(dataset, "NumberOfFrames", 1) or 1)
        samples = int(getattr(dataset, "SamplesPerPixel", 1) or 1)
        files += 1
        reference = "oracle"
        for index, frame in enumerate(
                generate_pixel_data_frame(dataset.PixelData, count)):
            mine = native.decode(frame)
            if samples == 1 and turbo.AVAILABLE:
                other = turbo.decode(frame)
                reference = "turbo"
            else:
                other = reference.decode(frame)
                reference = "oracle"
            if not (mine.dtype == other.dtype and mine.shape == other.shape
                    and np.array_equal(mine, other)):
                print(f"MISMATCH {path.name} frame {index} against {reference}")
                return 1
            frames += 1
        print(f"ok {path.name}: {count} frame(s) against {reference}")

    print(f"all identical: {frames} frames across {files} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
