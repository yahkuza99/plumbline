"""Run any lossless JPEG decoder against the conformance corpus.

    python conformance/run.py                          # every decoder installed
    python conformance/run.py --decoder plumbline
    python conformance/run.py --decoder mypackage:decode_frame

The decoder is any callable taking the encoded frame as `bytes` and returning
a NumPy array. Nothing here imports Plumbline unless you ask for it, so this
script is usable by a project that has never heard of us — which is the point
of publishing it.

Three outcomes, and the difference between the last two is the whole reason
this exists:

    exact       the pixels match the expected values
    refused     the decoder raised — allowed, and reported honestly as a gap
    WRONG       the decoder returned pixels that are not the right pixels

A decoder that refuses a case is telling the truth about its limits. A
decoder that returns the wrong pixels is not, and nobody downstream can tell.
Only WRONG sets a non-zero exit status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# decoders
# --------------------------------------------------------------------------- #

def _builtin_decoders() -> dict[str, object]:
    """Whatever is importable here. Missing ones are reported, not hidden."""
    found: dict[str, object] = {}

    try:
        sys.path.insert(0, str(HERE.parent / "src"))
        import plumbline

        found["plumbline"] = plumbline.decode
        from plumbline import reference

        found["plumbline.reference"] = reference.decode
    except Exception:
        pass

    try:
        from libjpeg import decode as pylibjpeg

        found["pylibjpeg"] = pylibjpeg
    except Exception:
        pass

    try:
        from imagecodecs import ljpeg_decode

        found["imagecodecs.ljpeg"] = ljpeg_decode
    except Exception:
        pass

    try:
        from imagecodecs import jpegsof3_decode

        found["imagecodecs.jpegsof3"] = jpegsof3_decode
    except Exception:
        pass

    return found


def _load(spec: str):
    """`module:function` — anything importable on this machine."""
    if ":" not in spec:
        raise SystemExit(f"--decoder must be a builtin name or module:function, got {spec!r}")
    module_name, _, function_name = spec.partition(":")
    module = __import__(module_name, fromlist=[function_name])
    return getattr(module, function_name)


# --------------------------------------------------------------------------- #

def check(decode, case: dict) -> tuple[str, str]:
    """(outcome, detail) for one case."""
    frame = (HERE / case["file"]).read_bytes()
    try:
        got = decode(frame)
    except Exception as error:
        return "refused", type(error).__name__

    got = np.squeeze(np.asarray(got))
    expected_shape = tuple(d for d in case["shape"] if d != 1)
    if got.shape != expected_shape:
        return "WRONG", f"shape {got.shape}, expected {expected_shape}"

    # Compare values, not the memory layout: a decoder returning int32 where
    # uint16 was expected is unusual but not incorrect, so long as every
    # sample is right. Only the numbers decide.
    expected = np.dtype(case["dtype"])
    if hashlib.sha256(np.ascontiguousarray(
            got.astype(expected, copy=False)).tobytes()).hexdigest() == case["sha256"]:
        note = "" if got.dtype == expected else f"(dtype {got.dtype}, expected {expected})"
        return "exact", note

    return "WRONG", "pixel values differ"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder", action="append", default=[],
                        help="builtin name, or module:function. Repeatable. "
                             "Default: every decoder installed here.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="list every case, not just the failures")
    args = parser.parse_args()

    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    cases = manifest["cases"]

    builtin = _builtin_decoders()
    if args.decoder:
        chosen = {name: (builtin[name] if name in builtin else _load(name))
                  for name in args.decoder}
    else:
        chosen = builtin
        for name in ("plumbline", "pylibjpeg", "imagecodecs.ljpeg", "imagecodecs.jpegsof3"):
            if name not in builtin:
                print(f"  {name}: not installed here")
        print()

    if not chosen:
        raise SystemExit("no decoders to test")

    print(f"{len(cases)} cases · expected values from {manifest['expected_values_from']}\n")

    worst = 0
    for name, decode in chosen.items():
        tally = {"exact": 0, "refused": 0, "WRONG": 0}
        problems: list[str] = []
        for case in cases:
            outcome, detail = check(decode, case)
            tally[outcome] += 1
            if outcome == "WRONG":
                problems.append(f"    {case['name']}: {detail}   [{case['why']}]")
            elif args.verbose:
                print(f"    {case['name']:44s} {outcome} {detail}")

        verdict = ("no wrong pixels" if tally["WRONG"] == 0
                   else f"*** {tally['WRONG']} CASES DECODED WRONG ***")
        print(f"{name}")
        print(f"  exact {tally['exact']:4d}   refused {tally['refused']:4d}   "
              f"wrong {tally['WRONG']:4d}   {verdict}")
        for line in problems[:20]:
            print(line)
        if len(problems) > 20:
            print(f"    … and {len(problems) - 20} more")
        print()
        worst = max(worst, tally["WRONG"])

    if worst:
        print("A wrong result is not automatically the decoder's fault — it means")
        print("this corpus and that decoder disagree, and one of them is wrong.")
        print("Every case cites the part of the format it covers; please argue")
        print("with the specification rather than with the number.")
    sys.exit(1 if worst else 0)


if __name__ == "__main__":
    main()
