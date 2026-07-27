# Plumbline

**A lossless JPEG decoder that is right, or says so.**

Plumbline decodes ITU-T T.81 Annex H lossless JPEG — the format behind DICOM
transfer syntaxes `1.2.840.10008.1.2.4.57` and `.70`, which is what most
medical images on hospital discs are actually stored in. It exists because a
decoder can be wrong without anyone noticing, and a medical image that looks
fine and isn't is worse than no image at all. So the rule here is
**decode correctly or raise — never return plausible wrong pixels** — and
every claim below names the evidence behind it. It is **Apache-2.0**, with no
copyleft anywhere in its dependency tree, because the software that most needs
a correct lossless-JPEG decoder is commercial imaging software and device
firmware that cannot touch GPL. It also happens to be about 25× faster than
what pydicom reaches for today.

```python
import plumbline

pixels = plumbline.decode(frame)   # (h, w) greyscale, or (h, w, components)
                                   # raises LosslessJpegError rather than guess
```

## Measured, not asserted

Median decode rate over 11 frames from real hospital discs, 8 manufacturers,
best of 3 runs each, on one AMD Zen 3 core (Windows 11, Python 3.12,
NumPy 1.26):

| decoder | median | frames accepted | agrees with the reference | licence |
|---|---|---|---|---|
| **plumbline (native C)** | **117.9 Mpx/s** | 11/11 | exact | Apache-2.0 |
| plumbline (numba fallback) | 82.2 Mpx/s | 9/11 (no colour) | exact | Apache-2.0 |
| pylibjpeg-libjpeg 2.4.0 | 4.6 Mpx/s | 11/11 | exact | **GPL-3.0** |
| plumbline (pure-Python reference) | ~0.15 Mpx/s | 11/11 | is the reference | Apache-2.0 |

Reproduce it on your own files — nothing is uploaded, nothing is written:

```sh
python native/build.py
PLUMBLINE_TESTDATA=/path/to/your/discs python native/compare.py
```

`compare.py` benchmarks **and diffs against the reference implementation**,
because a decoder that is fast and wrong is worth nothing. It reports every
decoder installed on your machine, and prints "not installed" rather than
skipping quietly, so a run that compared against nothing cannot be mistaken
for a clean sweep.

Numbers move ±10% run to run on a laptop. Report the median, say what you
measured on, and treat any single figure — including ours — with suspicion
until you have re-run it.

## Install

```sh
pip install plumbline
```

Wheels ship the compiled core, so there is nothing to build and no C toolchain
needed. If no wheel matches your platform, Plumbline still works: it falls
back to numba (`pip install plumbline[turbo]`) and then to the pure-Python
reference. **You lose speed, never a file.**

To build the core yourself:

```sh
python native/build.py     # finds cl, gcc, clang or zig cc automatically
```

## Scope

**Does:** lossless JPEG (SOF₃), precision 2–16, predictors 1–7, restart
markers, point transform, greyscale and interleaved colour.

**Does not, and refuses rather than guesses:** encoding · chroma subsampling ·
JPEG-LS · JPEG 2000 · baseline JPEG · arithmetic coding · parsing DICOM
itself. Plumbline takes a JPEG frame and returns pixels. Use pydicom or GDCM
for the container.

Keeping that surface small is the whole reason one person can make a
correctness promise about it. Feature requests that widen it will be declined
with thanks — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Correctness

11/11 real discs bit-exact at full size, every frame · colour bit-exact ·
a synthetic sweep across table shape × precision × predictor × restart
interval × point transform · **zero silent disagreements** · 261 tests.

Everything is verified against `plumbline.reference`, the pure-Python
implementation in this repository — never against another library. Testing a
decoder against a second decoder only proves they share assumptions. The
reference is written to be read against the specification, clause by clause,
and its comments cite those clauses.

**[CORRECTNESS.md](CORRECTNESS.md)** has the full claim, the evidence, the
three places in the specification where decoders go wrong, and — just as
importantly — what this project does *not* claim.

## Status

Maintained by one person with a day job. Issues get a first reply within about
a week; a bug with a frame that reproduces it gets one much faster. There is
no roadmap beyond what is in the issues, and the scope above is closed.

If a frame decodes wrongly, **that report is worth more to this project than a
pull request** — it is the one thing we cannot generate ourselves. Strip the
patient data first; the JPEG frame is all that is needed, and nothing carrying
a DICOM preamble will be accepted.

## Licence

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
Copyright 2026 Rungroj T.

**Not a medical device.** Not submitted to any regulator, not for diagnostic
use. If you build it into a regulated product, its verification is yours to
do; CORRECTNESS.md exists to make that easier, not to do it for you.
