"""Write the conformance corpus and its manifest.

Each case is a lossless JPEG frame plus the pixels it must decode to. The
pixels are stored as a SHA-256 of the array bytes together with the dtype and
shape, so a third party can check their own decoder without installing
Plumbline and without downloading a gigabyte of expected images.

The expected values come from `plumbline.reference`, the pure-Python
implementation, which is written to be read against ITU-T T.81 clause by
clause. That is an honest statement of what this corpus is: not an
independent authority, but one careful reading of the specification, written
down so that anyone can disagree with it *specifically* — cite the case and
the clause, and the argument is about the standard rather than about whose
decoder is nicer.

    python conformance/generate.py

Regenerating must be reproducible: same seeds, same frames, same hashes. If a
regeneration changes a hash, either the reference implementation changed
behaviour or the generator did, and both of those are things somebody needs
to look at rather than commit.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

import numpy as np                                              # noqa: E402

from conformance import frames as build                         # noqa: E402
from plumbline import reference                                 # noqa: E402

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"
MANIFEST = HERE / "manifest.json"

# Small on purpose: every case must be readable as a hex dump when it fails,
# and the whole corpus has to stay something people will actually clone.
WIDTH, HEIGHT = 7, 5


def digest(pixels: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest()


def cases():
    """Every case in the corpus, as (name, frame bytes, why it is here)."""
    seed = 0

    # ---- greyscale: the full cross-product of the things that interact ----
    for shape in ("staircase", "narrow", "holes"):
        for precision in (8, 12, 16):
            for predictor in range(1, 8):
                for restart in (0, 1, 3):
                    seed += 1
                    top = min(precision, 16)
                    counts, symbols = build.table(shape, top)
                    frame = build.greyscale_frame(
                        WIDTH, HEIGHT, precision,
                        build.scan_for(WIDTH * HEIGHT, seed),
                        counts, symbols,
                        predictor=predictor,
                        restart_interval=restart * WIDTH,
                    )
                    yield (
                        f"grey_{shape}_p{precision}_pred{predictor}_ri{restart}",
                        frame,
                        f"{shape} table, {precision}-bit, predictor {predictor}, "
                        + (f"restart every {restart} row(s)" if restart else "no restarts"),
                    )

    # ---- point transform: samples are shifted left after reconstruction ----
    for point_transform in (1, 3):
        for precision in (8, 12):
            seed += 1
            counts, symbols = build.table("narrow", precision)
            yield (
                f"grey_pt{point_transform}_p{precision}",
                build.greyscale_frame(WIDTH, HEIGHT, precision,
                                      build.scan_for(WIDTH * HEIGHT, seed),
                                      counts, symbols, point_transform=point_transform),
                f"point transform {point_transform}, {precision}-bit",
            )

    # ---- colour: interleaved, and the restart interval counts pixels ----
    for components in (2, 3, 4):
        for shared_table in (True, False):
            for restart in (0, 1):
                seed += 1
                if shared_table:
                    tables = {0: build.table("narrow", 8)}
                    plan = [(i + 1, 0) for i in range(components)]
                else:
                    tables = {i: build.table(("narrow", "staircase", "holes")[i % 3], 8)
                              for i in range(components)}
                    plan = [(i + 1, i) for i in range(components)]
                yield (
                    f"colour_c{components}_{'shared' if shared_table else 'per'}_ri{restart}",
                    build.colour_frame(WIDTH, HEIGHT, 8,
                                       build.scan_for(WIDTH * HEIGHT * components, seed),
                                       tables, plan, restart_interval=restart * WIDTH),
                    f"{components} interleaved components, "
                    + ("one table" if shared_table else "a table each")
                    + (", restarts every row" if restart else ""),
                )

    # ---- the special case that desynchronises a scan if it is missed ----
    # SSSS = 16 means difference 32768 with NO bits following (T.81 H.1.2.2).
    seed += 1
    counts = [2] + [0] * 15
    yield (
        "grey_ssss16_no_mantissa",
        build.greyscale_frame(WIDTH, HEIGHT, 16, build.scan_for(WIDTH * HEIGHT, seed),
                              counts, [0, 16]),
        "SSSS=16: difference 32768 and no bits follow (T.81 H.1.2.2)",
    )


def main() -> None:
    CORPUS.mkdir(exist_ok=True)
    for stale in CORPUS.glob("*.jpg"):
        stale.unlink()

    manifest, refused = [], 0
    for name, frame, why in cases():
        try:
            pixels = reference.decode(frame)
        except Exception as error:
            # A case the reference itself refuses cannot state an expected
            # result, so it is not a conformance case. Counting them keeps an
            # accidentally empty corpus from looking like a healthy one.
            refused += 1
            print(f"  reference refused {name}: {type(error).__name__}: {error}")
            continue

        (CORPUS / f"{name}.jpg").write_bytes(frame)
        manifest.append({
            "name": name,
            "file": f"corpus/{name}.jpg",
            "why": why,
            "shape": list(pixels.shape),
            "dtype": str(pixels.dtype),
            "sha256": digest(pixels),
        })

    MANIFEST.write_text(json.dumps(
        {
            "format": "ITU-T T.81 Annex H lossless JPEG (SOF3)",
            "transfer_syntaxes": ["1.2.840.10008.1.2.4.57", "1.2.840.10008.1.2.4.70"],
            "expected_values_from": "plumbline.reference (pure Python)",
            "contains_patient_data": False,
            "cases": manifest,
        },
        indent=2) + "\n", encoding="utf-8")

    print(f"\n{len(manifest)} cases written to {CORPUS}")
    if refused:
        print(f"{refused} generated frames were refused by the reference and left out")


if __name__ == "__main__":
    main()
