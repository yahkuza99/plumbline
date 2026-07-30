# Plumbline

**A lossless JPEG decoder that is right, or says so.**

A decoder can be wrong without anyone noticing. It returns an array, the array
renders, the image looks like an image — and it is not the image that was
recorded. In medical imaging that is the worst failure mode there is, because
nothing about it looks like a failure.

Plumbline decodes ITU-T T.81 Annex H lossless JPEG — DICOM transfer syntaxes
`1.2.840.10008.1.2.4.57` and `.70`, the format most hospital discs actually
use. It has one rule, and every design decision is subordinate to it:

> ### Decode correctly, or raise.
> ### Never return plausible wrong pixels.

```python
import plumbline

pixels = plumbline.decode(frame)   # (h, w), or (h, w, components) for colour
                                   # LosslessJpegError rather than a guess
```

Apache-2.0, with no copyleft anywhere in the dependency tree — because the
software that most needs a correct lossless-JPEG decoder is commercial imaging
software and device firmware, which cannot touch GPL. NumPy is the only
runtime dependency.

---

## Evidence

Every number here was measured on this machine and can be re-measured on
yours. None of it is asserted.

### Against real hospital discs

| | |
|---:|:---|
| **61,921** | frames decoded — **0 refused, 0 crashed** |
| **26.1 billion** | pixels |
| **105** | scanner configurations, from **27** manufacturers |
| **185 / 0** | covering sample re-decoded with the pure-Python reference: exact / differing |

Comparing every frame against the reference is not possible — it runs at about
0.15 Mpx/s, so 26 billion pixels would take a day. So every frame goes through
the shipping decoder, which catches refusals, crashes and hangs; then one frame
from each distinct parameter combination is re-decoded with the reference and
compared bit for bit.

### Against the specification

**1,707 conformance cases**, all passing: 1,691 that must decode and 16 that
must be refused.

The valid cases sweep the *entire legal parameter space* — precision 2 to 16,
predictors 1 to 7, one to four components, point transform, restart intervals.
That space has edges, so it can be enumerated rather than sampled. There is no
conforming combination this decoder has not been shown, which is a stronger
claim than any quantity of real files can support.

Each is built by encoding an image we chose, so the expected pixels are the
image itself — a round trip, not a decoder's opinion of one.

The 16 refusal cases cover what the specification does *not* bound: a
truncated scan, a table the scan names but nobody defined, an interval
declared and never emitted, restart markers out of sequence. Those have no
right answer, so any pixels at all are wrong pixels.

### Speed

Median over 11 frames from real discs, best of three, one AMD Zen 3 core:

| decoder | median | accepted | agrees | licence |
|---|---:|---:|---|---|
| **plumbline** (compiled C) | **117.9 Mpx/s** | 11/11 | exact | Apache-2.0 |
| plumbline (numba fallback) | 82.2 Mpx/s | 9/11 | exact | Apache-2.0 |
| pylibjpeg-libjpeg 2.4.0 | 4.6 Mpx/s | 11/11 | exact | **GPL-3.0** |
| plumbline (reference) | ~0.15 Mpx/s | 11/11 | *is* the reference | Apache-2.0 |

Speed is listed last on purpose. It is the least interesting thing about this
project.

### Reproduce all of it

```sh
python native/build.py
python -m pytest tests/ -q
python conformance/run.py                      # every decoder installed here
PLUMBLINE_TESTDATA=/path/to/your/discs python native/compare.py
```

`compare.py` measures **and diffs against the reference at the same time**,
because a decoder that is fast and wrong is worth nothing. It names every
decoder it could not import rather than skipping quietly, so a run that
compared against nothing cannot be mistaken for a clean sweep.

Nothing is uploaded. Nothing is written. Numbers move ±10% on a laptop —
report the median, say what you measured on, and distrust any single figure,
including ours, until you have run it yourself.

---

## What this does not cover

The limits matter more than the totals, so they are stated first rather than
buried.

**Every predictor observed in all 61,921 real frames was 1.** The other six are
implemented and swept synthetically, but no scanner here has exercised them.
That is the sharpest limit in this document — and it is exactly where two bugs
hid until the conformance corpus was built.

**Those discs all come from hospitals in one country.** The scanner mix is what
those hospitals bought. Konica Minolta and Shimadzu appear nowhere in the
corpus. Eight frames from US research collections are committed to widen it,
adding PET — but they are predictor 1 as well, so they broaden vendor and
modality coverage, not the coverage that matters most.

**"No known silent failures on the corpus we have"** is a much weaker statement
than "correct". It is also the strongest statement anyone can honestly make
about a decoder.

**[CORRECTNESS.md](CORRECTNESS.md)** — the full claim, the evidence behind each
part of it, the three places in the specification where decoders go wrong, and
what this project does not claim.

---

## Install

```sh
pip install plumbline
```

Wheels carry the compiled core — no build step, no C toolchain. Where no wheel
matches your platform it falls back to numba (`plumbline[turbo]`), then to the
pure-Python reference. **You lose speed, never a file.**

Build the core yourself with `python native/build.py`, which finds `cl`, `gcc`,
`clang` or `zig cc` on its own.

---

## Scope

**Does:** lossless JPEG (SOF₃) · precision 2–16 · predictors 1–7 · restart
markers · point transform · greyscale and interleaved colour.

**Does not, and refuses rather than guesses:** encoding · chroma subsampling ·
JPEG-LS · JPEG 2000 · baseline JPEG · arithmetic coding · parsing DICOM itself.

Plumbline takes a JPEG frame and returns pixels. Use pydicom or GDCM for the
container. Keeping that surface small is the only reason one person can make a
correctness promise about it at all, so requests to widen it are declined with
thanks — see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Three implementations, one answer

| | |
|---|---|
| `native` | compiled C, loaded through ctypes. Never touches the Python C API, so one binary serves every Python version on a platform. |
| `turbo` | numba. Greyscale only. For platforms with no wheel. |
| `reference` | pure Python. Slow, and the most important file here — written to be read against the specification clause by clause, with the clause numbers in the comments. |

All three must produce the same bits or refuse. That is the entire contract
between them, and the shared validation is written once and imported, not
reimplemented — because a check that lives in one engine is the check that
drifts.

But agreement is the wrong oracle for truth. **Three implementations that
share a misreading agree perfectly and are wrong together** — which is exactly
what happened, twice, before the first release. So the conformance corpus is
also run against decoders that share no code with this one, and the
specification is quoted wherever it is easy to misread.

---

## Status

Maintained by one person with a day job. A first reply within about a week; a
bug with a frame that reproduces it, much faster. The scope above is closed and
there is no roadmap beyond the open issues.

**If a frame decodes wrongly, that report is worth more than a pull request.**
It is the one thing this project cannot generate for itself. Strip the patient
data first — the bare JPEG frame is all that is needed, and nothing carrying a
DICOM preamble will be accepted.

---

## Licence

Apache-2.0 — [LICENSE](LICENSE), [NOTICE](NOTICE). Copyright 2026 Rungroj T.

**Not a medical device.** Not submitted to any regulator, not for diagnostic
use. If you build it into a regulated product, its verification is yours to do.
CORRECTNESS.md exists to make that easier, not to do it for you.
