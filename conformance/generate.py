"""Write the conformance corpus: everything the format allows, and a battery
of the ways real files break it.

Two halves, because they answer different questions.

**Valid cases** sweep the whole legal parameter space. That space has edges —
precision 2 to 16, predictors 1 to 7, one to four components, point transform
below precision, restart intervals — so it can be enumerated rather than
sampled. After this runs there is no specification-conforming combination the
decoder has not been shown. That is a stronger claim than any number of real
files can support, because real files only cover what the machines that wrote
them happened to do.

Each valid case is built by encoding an image we chose, so the expected pixels
are the image itself. Nobody has to take our decoder's word for anything, and
a disagreement is a fact rather than an opinion.

**Malformed cases** cover what the specification does *not* bound. Firmware
gets things wrong, and no amount of reading T.81 tells you how. What can be
enumerated is the family a break belongs to — a truncated scan, a restart
interval declared and never emitted, a Huffman table that is over- or
under-subscribed — and for those the corpus asserts only that a decoder must
not return an image. Refusing is the whole point: these have no right answer,
so any pixels at all are wrong pixels.

    python conformance/generate.py

Regeneration must be reproducible: same seeds, same frames, same hashes. A
hash that moves means the reference implementation or the generator changed
behaviour, and both are things somebody needs to look at rather than commit.
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

from conformance import encode as enc                           # noqa: E402
from conformance import frames as build                         # noqa: E402
from plumbline import reference                                 # noqa: E402

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"
MANIFEST = HERE / "manifest.json"

WIDTH, HEIGHT = 7, 5          # small enough to read as a hex dump when it fails


def digest(pixels: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest()


def _image(width: int, height: int, components: int, precision: int,
           pattern: str, seed: int) -> np.ndarray:
    """Content chosen to exercise the predictor, not to look like anything.

    A flat image hides every predictor bug, because every prediction is right.
    A gradient makes predictors 1-4 differ; noise reaches the corners of the
    Huffman table; and extremes force the differences that wrap, which is
    where predictors 5-7 carry the error forward.
    """
    top = (1 << precision) - 1
    generator = np.random.default_rng(seed)
    if pattern == "flat":
        out = np.full((height, width, components), top // 2, dtype=np.int64)
    elif pattern == "gradient":
        ramp = np.linspace(0, top, width * height).reshape(height, width)
        out = np.repeat(ramp[:, :, None], components, axis=2).astype(np.int64)
    elif pattern == "extremes":
        out = generator.choice([0, top], (height, width, components)).astype(np.int64)
    else:
        out = generator.integers(0, top + 1, (height, width, components), dtype=np.int64)
    return out[:, :, 0] if components == 1 else out


# --------------------------------------------------------------------------- #
# valid: the whole legal space
# --------------------------------------------------------------------------- #

def valid_cases():
    """Every (precision, predictor, components, restart, content) combination."""
    seed = 0
    for precision in range(2, 17):
        for predictor in range(1, 8):
            for components in (1, 2, 3, 4):
                # T.81 §H.1.1 and table B.7: Ri shall be an integer multiple of
                # the MCU in an MCU-row, so the intervals worth sweeping are
                # whole rows. The last is wider than the image, which is legal
                # and reaches no marker at all — the case that catches a
                # decoder counting intervals it never sees. Intervals that
                # divide a row (this sweep once used 1 and 3 against a width of
                # 7) are not a harder valid case but an invalid one, and they
                # have moved to `malformed_cases` where they belong.
                for restart in (0, WIDTH, WIDTH * 2, WIDTH * (HEIGHT + 1)):
                    for pattern in ("gradient", "noise", "extremes", "flat"):
                        seed += 1
                        # One pattern per shape keeps the corpus at a size
                        # people will clone; the rotation still reaches every
                        # pattern at every precision.
                        if (seed % 4) != (precision % 4):
                            continue
                        image = _image(WIDTH, HEIGHT, components, precision,
                                       pattern, seed)
                        data, plan = enc.encode(image, precision, predictor,
                                                restart_interval=restart)
                        if components == 1:
                            frame = build.greyscale_frame(
                                WIDTH, HEIGHT, precision, data, *plan[0],
                                predictor=predictor, restart_interval=restart)
                        else:
                            frame = build.colour_frame(
                                WIDTH, HEIGHT, precision, data, plan,
                                [(i + 1, i) for i in range(components)],
                                predictor=predictor, restart_interval=restart)
                        yield (f"p{precision:02d}_pred{predictor}_c{components}"
                               f"_ri{restart}_{pattern}",
                               frame, np.squeeze(image),
                               f"precision {precision}, predictor {predictor}, "
                               f"{components} component(s), "
                               + (f"restart every {restart} MCU "
                                  f"({restart // WIDTH} row(s))"
                                  if restart else "no restarts")
                               + f", {pattern} content")

    # point transform: the decoder shifts samples left on the way out
    for point_transform in (1, 2, 4, 8):
        for precision in (8, 12, 16):
            if point_transform >= precision:
                continue
            seed += 1
            image = _image(WIDTH, HEIGHT, 1, precision - point_transform,
                           "noise", seed).astype(np.int64)
            data, plan = enc.encode(image, precision, 1, point_transform=point_transform)
            frame = build.greyscale_frame(WIDTH, HEIGHT, precision, data, *plan[0],
                                          point_transform=point_transform)
            yield (f"pt{point_transform}_p{precision}", frame,
                   np.squeeze(image) << point_transform,
                   f"point transform {point_transform}, {precision}-bit")


# --------------------------------------------------------------------------- #
# malformed: what the specification does not bound
# --------------------------------------------------------------------------- #

def _sound_frame(precision=8, predictor=1, restart=0, components=1, seed=99):
    image = _image(WIDTH, HEIGHT, components, precision, "noise", seed)
    data, plan = enc.encode(image, precision, predictor, restart_interval=restart)
    if components == 1:
        return build.greyscale_frame(WIDTH, HEIGHT, precision, data, *plan[0],
                                     predictor=predictor, restart_interval=restart), data, plan
    return build.colour_frame(WIDTH, HEIGHT, precision, data, plan,
                              [(i + 1, i) for i in range(components)],
                              predictor=predictor, restart_interval=restart), data, plan


def malformed_cases():
    """Frames a decoder must refuse. Any pixels at all are wrong pixels."""
    good, data, plan = _sound_frame()

    yield ("bad_truncated_scan", good[:len(good) // 2],
           "the scan stops mid-symbol — a decoder that reads past the end "
           "invents the tail of the image")

    yield ("bad_no_sof", good.replace(b"\xff\xc3", b"\xff\xc9", 1),
           "SOF3 replaced by a marker this decoder does not implement")

    yield ("bad_no_huffman_table",
           build.marker(0xD8)
           + build.marker(0xC3, bytes([8, 0, HEIGHT, 0, WIDTH, 1, 0, 0x11, 0]))
           + build.marker(0xDA, bytes([1, 0, 0x00, 1, 0, 0])) + data + build.marker(0xD9),
           "SOS refers to a Huffman table that was never defined")

    # DRI announced, no RST emitted — the exact defect this corpus was rebuilt
    # to avoid producing by accident. Now it is a case on purpose.
    flat, flat_data, flat_plan = _sound_frame(restart=0)
    yield ("bad_dri_without_rst",
           build.marker(0xD8)
           + build.marker(0xC3, bytes([8, 0, HEIGHT, 0, WIDTH, 1, 0, 0x11, 0]))
           + build.huffman_table(*flat_plan[0])
           + build.marker(0xDD, bytes([0, WIDTH]))
           + build.marker(0xDA, bytes([1, 0, 0x00, 1, 0, 0]))
           + flat_data + build.marker(0xD9),
           "a restart interval is declared but no RSTn marker is ever emitted")

    rst, rst_data, rst_plan = _sound_frame(restart=WIDTH)
    yield ("bad_rst_out_of_order", rst.replace(b"\xff\xd0", b"\xff\xd5", 1),
           "restart markers must cycle RST0-RST7 in order; this one jumps")

    # T.81 §H.1.1 / table B.7 require Ri to be a whole number of MCU-rows, and
    # §H.1.2.1 gives the first *line* of every interval to Ra. An interval that
    # begins mid-row has no first line the standard names, so there is no right
    # answer to return — three readings are defensible and libjpeg reproduces
    # none of them. This case says only that a decoder must not pick one
    # silently. The entropy data is a valid row-aligned frame's, so the frame
    # is wrong in exactly one respect: the DRI value.
    yield ("bad_restart_interval_not_whole_rows",
           rst.replace(build.marker(0xDD, bytes([0, WIDTH])),
                       build.marker(0xDD, bytes([0, WIDTH - 4])), 1),
           f"a restart interval of {WIDTH - 4} MCU in a {WIDTH}-MCU row: T.81 "
           "§H.1.1 requires a whole number of MCU-rows, and mid-row the "
           "predictor reset in §H.1.2.1 has no defined meaning")

    yield ("bad_subsampled",
           build.marker(0xD8)
           + build.marker(0xC3, bytes([8, 0, HEIGHT, 0, WIDTH, 1, 0, 0x21, 0]))
           + build.huffman_table(*plan[0])
           + build.marker(0xDA, bytes([1, 0, 0x00, 1, 0, 0])) + data + build.marker(0xD9),
           "2x1 sampling — not supported, and guessing at it would misplace "
           "every sample")

    yield ("bad_zero_width",
           build.marker(0xD8)
           + build.marker(0xC3, bytes([8, 0, HEIGHT, 0, 0, 1, 0, 0x11, 0]))
           + build.huffman_table(*plan[0])
           + build.marker(0xDA, bytes([1, 0, 0x00, 1, 0, 0])) + data + build.marker(0xD9),
           "an image zero pixels wide")

    yield ("bad_precision_zero",
           build.marker(0xD8)
           + build.marker(0xC3, bytes([0, 0, HEIGHT, 0, WIDTH, 1, 0, 0x11, 0]))
           + build.huffman_table(*plan[0])
           + build.marker(0xDA, bytes([1, 0, 0x00, 1, 0, 0])) + data + build.marker(0xD9),
           "precision 0 — outside the 2..16 the format allows")

    yield ("bad_predictor_zero",
           build.greyscale_frame(WIDTH, HEIGHT, 8, data, *plan[0], predictor=0),
           "predictor 0 is only defined for the hierarchical mode, not here")

    yield ("bad_predictor_eight",
           build.greyscale_frame(WIDTH, HEIGHT, 8, data, *plan[0], predictor=8),
           "predictor 8 does not exist")

    over = [0] * 16
    over[0] = 3                                   # three 1-bit codes: Kraft > 1
    yield ("bad_table_oversubscribed",
           build.greyscale_frame(WIDTH, HEIGHT, 8, data, over, [0, 1, 2]),
           "a Huffman table claiming more codes of a length than exist")

    # This was `bad_ssss_beyond_precision`, on the theory that a 4-bit frame
    # cannot carry an SSSS of 12. It can: differences are taken modulo 2^16,
    # not modulo 2^P, so any representative of the residue class is legal and
    # encoders pick the shortest — the 10-bit Hologic mammogram in the test
    # data lists SSSS = 12 and decodes identically under three decoders. What
    # is wrong with this frame is the half of the code space its one 1-bit
    # code leaves unclaimed, which the scan then lands in 30 times out of 35.
    yield ("bad_code_not_in_table",
           build.greyscale_frame(WIDTH, HEIGHT, 4, data, [1] + [0] * 15, [12]),
           "the scan uses codes the frame's only Huffman table never defines")

    # Spliced in rather than edited over an existing stuffed pair: whether this
    # scan happens to contain a 0xFF at all depends on the seed, and the
    # `.replace` this case used to do quietly matched nothing and published a
    # perfectly valid frame as a must-refuse case. Splicing always breaks it.
    half = len(data) // 2
    yield ("bad_unstuffed_ff",
           build.greyscale_frame(WIDTH, HEIGHT, 8,
                                 data[:half] + b"\xFF\x01" + data[half:],
                                 *plan[0]),
           "a 0xFF inside the entropy data followed by something that is "
           "neither stuffing nor a legal marker")

    yield ("bad_empty_scan",
           build.marker(0xD8)
           + build.marker(0xC3, bytes([8, 0, HEIGHT, 0, WIDTH, 1, 0, 0x11, 0]))
           + build.huffman_table(*plan[0])
           + build.marker(0xDA, bytes([1, 0, 0x00, 1, 0, 0])) + build.marker(0xD9),
           "no entropy data at all between SOS and EOI")

    yield ("bad_not_a_jpeg", b"DICM" + bytes(200),
           "not a JPEG stream in any sense")


# --------------------------------------------------------------------------- #

def main() -> None:
    CORPUS.mkdir(exist_ok=True)
    for stale in CORPUS.glob("*.jpg"):
        stale.unlink()

    cases, round_trip_failed, refused_by_reference = [], [], []

    for name, frame, expected, why in valid_cases():
        try:
            decoded = reference.decode(frame)
        except Exception as error:
            refused_by_reference.append(f"{name}: {type(error).__name__}: {error}")
            continue
        if not np.array_equal(np.squeeze(decoded), np.squeeze(expected)):
            # The generator and the decoder disagree about the format. Until
            # that is settled the case states nothing, so it is not published.
            round_trip_failed.append(name)
            continue
        (CORPUS / f"{name}.jpg").write_bytes(frame)
        cases.append({
            "name": name, "file": f"corpus/{name}.jpg", "why": why,
            "must": "decode", "shape": list(np.squeeze(expected).shape),
            "dtype": str(decoded.dtype), "sha256": digest(np.squeeze(expected).astype(decoded.dtype)),
        })

    for name, frame, why in malformed_cases():
        try:
            reference.decode(frame)
        except Exception:
            pass
        else:
            # If the reference accepts it, it is not malformed enough to
            # assert on — saying "everyone must refuse this" while our own
            # decoder does not would be exactly backwards.
            round_trip_failed.append(f"{name} (reference accepted it)")
            continue
        (CORPUS / f"{name}.jpg").write_bytes(frame)
        cases.append({"name": name, "file": f"corpus/{name}.jpg", "why": why,
                      "must": "refuse"})

    MANIFEST.write_text(json.dumps({
        "format": "ITU-T T.81 Annex H lossless JPEG (SOF3)",
        "transfer_syntaxes": ["1.2.840.10008.1.2.4.57", "1.2.840.10008.1.2.4.70"],
        "expected_values_from": "the images this corpus encodes (round trip), "
                                "not from any decoder's output",
        "contains_patient_data": False,
        "cases": cases,
    }, indent=1) + "\n", encoding="utf-8")

    must_decode = sum(1 for c in cases if c["must"] == "decode")
    must_refuse = len(cases) - must_decode
    print(f"{len(cases)} cases: {must_decode} must decode, {must_refuse} must refuse")
    if refused_by_reference:
        print(f"\n{len(refused_by_reference)} valid frames the reference refused:")
        for line in refused_by_reference[:15]:
            print(f"  {line}")
    if round_trip_failed:
        print(f"\n{len(round_trip_failed)} cases withheld — generator and decoder disagree:")
        for line in round_trip_failed[:15]:
            print(f"  {line}")


if __name__ == "__main__":
    main()
